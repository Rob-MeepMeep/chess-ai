"""
mcts.py — Batched Monte Carlo Tree Search guided by the neural network.

Each simulation does four things:
  1. SELECT   — walk the tree using UCB until an unexpanded node
  2. EXPAND   — ask the network for priors over all legal moves at that node
  3. EVALUATE — ask the network's value head: who is winning here?
  4. BACKUP   — propagate the value back up the path, flipping sign at each
                level (what's good for me is bad for my opponent)

BATCHING (two levels):
  Within one tree: BATCH_SIMS simulations run to their leaves, then all
  leaves are evaluated in one network call (with VIRTUAL LOSS keeping the
  parallel simulations on different branches).

  Across trees: the search is split into three steps — gather_leaves(),
  evaluate(), apply_results() — so a training loop running many games at
  once can pool every game's leaves into a single large network call.
  16 games × 8-leaf waves = a 128-position batch, which is what a modern
  GPU actually wants. search() below is just these three steps in a loop,
  so single-game callers (eval, the API server) work exactly as before.

CPU/GPU DISCIPLINE:
  Exactly one host→device transfer (the encoded batch in) and one
  device→host transfer (logits + values out) per network call. All prior
  math — mirroring, masking, softmax — happens on CPU numpy afterwards.
  The previous version called .item() once per legal move per leaf
  (~18,000 device round-trips per move at 600 sims), which starved the GPU.
"""

import numpy as np
import torch
import chess

from chessai.encoder import encode
from chessai.moves  import get_mirror_indices_np, move_to_index, index_to_move

C_PUCT          = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS   = 0.25
BATCH_SIMS      = 32     # leaves evaluated per network call within ONE tree.
                         # Virtual loss keeps quality reasonable up to ~64;
                         # bigger batches should come from more parallel games,
                         # not bigger waves (less distortion of a single search).
VIRTUAL_LOSS    = 1.0    # penalty applied to in-flight nodes


class MCTSNode:
    __slots__ = ("N", "W", "P", "parent", "children", "is_expanded")

    def __init__(self, prior: float, parent: "MCTSNode" = None):
        self.N           = 0
        self.W           = 0.0
        self.P           = prior
        self.parent      = parent
        self.children: dict[int, "MCTSNode"] = {}
        self.is_expanded = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def ucb(self, c_puct: float) -> float:
        n_parent = self.parent.N if self.parent else 1
        return self.Q + c_puct * self.P * (n_parent ** 0.5) / (1 + self.N)


class SearchState:
    """
    One in-progress search: the tree plus enough bookkeeping to pause after
    gather_leaves() and resume in apply_results(). This is what lets many
    games interleave their searches through shared network calls.
    """
    __slots__ = ("root", "board", "history", "n_target", "sims_done",
                 "noise_pending", "_wave", "_expand_meta")

    def __init__(self, board: chess.Board, history: list,
                 n_simulations: int, add_noise: bool):
        self.root          = MCTSNode(prior=1.0)
        self.board         = board
        self.history       = list(history)[:3]
        self.n_target      = n_simulations
        self.sims_done     = 0
        self.noise_pending = add_noise   # applied once the root is expanded
        self._wave         = None        # in-flight (path, board, history) triples
        self._expand_meta  = None        # node_id → (node, board) awaiting priors

    @property
    def done(self) -> bool:
        return self.sims_done >= self.n_target


class MCTS:

    def __init__(self, network: torch.nn.Module, device: torch.device,
                 c_puct: float = C_PUCT, batch_sims: int = BATCH_SIMS):
        self.network    = network
        self.device     = device
        self.c_puct     = c_puct
        self.batch_sims = batch_sims
        # bf16 inference roughly halves inference time on CUDA/ROCm.
        # Training stays fp32 — this only affects self-play evaluation.
        self.autocast   = (device.type == "cuda")
        self.network.eval()

    # ------------------------------------------------------------------
    # Single-tree convenience API (eval, API server, snapshots)
    # ------------------------------------------------------------------

    def search(self, board: chess.Board, history: list,
               n_simulations: int, add_noise: bool = False) -> tuple:
        """
        Run a complete search on one tree.
        Returns (policy, root): policy is a (4096,) tensor of normalised
        visit counts; root exposes child Q-values (used for resign checks).
        """
        state = self.begin_search(board, history, n_simulations, add_noise)
        while not state.done:
            batch = self.gather_leaves(state)
            if batch is None:
                self.apply_results(state, None, None)
            else:
                logits, values = self.evaluate(batch)
                self.apply_results(state, logits, values)
        return self.extract_policy(state), state.root

    # ------------------------------------------------------------------
    # Step-driven API — the lockstep training loop drives these directly
    # ------------------------------------------------------------------

    def begin_search(self, board: chess.Board, history: list,
                     n_simulations: int, add_noise: bool = False) -> SearchState:
        """Start a search. The root is expanded by the first wave (batched
        with everything else) rather than by a separate network call."""
        return SearchState(board, history, n_simulations, add_noise)

    def gather_leaves(self, state: SearchState):
        """
        Select the next wave of leaves and return their encoded positions as
        one CPU tensor ready for the network — or None if every leaf in the
        wave is terminal (apply_results must still be called to back up).
        """
        # First wave: the root itself is the only leaf, so a full wave would
        # just burn sims re-evaluating it. Spend exactly one.
        if not state.root.is_expanded:
            wave = 1
        else:
            wave = min(self.batch_sims, state.n_target - state.sims_done)

        wave_data = []
        for _ in range(wave):
            path, sim_board, sim_hist = self._select_to_leaf(
                state.root, state.board.copy(), list(state.history)
            )
            wave_data.append((path, sim_board, sim_hist))
            self._apply_virtual_loss(path)

        # Deduplicate shared leaves; skip terminal ones (they need no network)
        to_expand: dict[int, tuple] = {}
        for path, sim_board, sim_hist in wave_data:
            leaf    = path[-1]
            node_id = id(leaf)
            if (not sim_board.is_game_over()
                    and not leaf.is_expanded
                    and node_id not in to_expand):
                to_expand[node_id] = (leaf, sim_board, sim_hist)

        state._wave        = wave_data
        state._expand_meta = to_expand

        if not to_expand:
            return None
        return torch.stack([encode([b] + h) for _, b, h in to_expand.values()])

    def evaluate(self, batch: torch.Tensor) -> tuple:
        """
        One network call for a batch of encoded positions (from one tree or
        many). Returns (logits, values) as float32 numpy arrays — everything
        downstream is CPU work.
        """
        with torch.inference_mode():
            x = batch.to(self.device)
            if self.autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, values = self.network(x)
            else:
                logits, values = self.network(x)
        return (logits.float().cpu().numpy(),
                values.float().cpu().numpy())

    def apply_results(self, state: SearchState,
                      logits: np.ndarray, values: np.ndarray) -> None:
        """
        Expand the leaves gathered by gather_leaves() using the network's
        output rows, then remove virtual loss and back up all wave paths.
        """
        node_values: dict[int, float] = {}
        if state._expand_meta:
            for i, (node_id, (leaf, sim_board, _)) in enumerate(
                    state._expand_meta.items()):
                self._expand_from_logits(leaf, sim_board, logits[i])
                node_values[node_id] = float(values[i, 0])

        for path, sim_board, _ in state._wave:
            self._remove_virtual_loss(path)
            leaf = path[-1]
            if sim_board.is_game_over():
                value = self._terminal_value(sim_board)
            else:
                value = node_values[id(leaf)]
            self._backup(path, value)

        state.sims_done   += len(state._wave)
        state._wave        = None
        state._expand_meta = None

        # Root just got expanded — apply exploration noise before the real waves
        if state.noise_pending and state.root.is_expanded:
            self._add_dirichlet_noise(state.root)
            state.noise_pending = False

    def extract_policy(self, state: SearchState) -> torch.Tensor:
        """Normalised visit counts as a (4096,) tensor."""
        policy = torch.zeros(4096)
        for idx, child in state.root.children.items():
            policy[idx] = child.N
        total = policy.sum()
        if total > 0:
            return policy / total
        # Degenerate case (n_simulations too small to visit any child):
        # fall back to the network priors so callers always get a distribution.
        for idx, child in state.root.children.items():
            policy[idx] = child.P
        total = policy.sum()
        return policy / total if total > 0 else policy

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_to_leaf(self, root: MCTSNode, sim_board: chess.Board,
                        sim_history: list) -> tuple:
        """Walk the tree to a leaf using UCB. Returns (path, board, history)."""
        node = root
        path = [node]

        while node.is_expanded and not sim_board.is_game_over():
            move_idx, node = self._select(node)
            move = index_to_move(move_idx, sim_board)
            sim_history = ([sim_board.copy()] + sim_history)[:3]
            sim_board.push(move)
            path.append(node)

        return path, sim_board, sim_history

    def _select(self, node: MCTSNode) -> tuple:
        best_idx = max(node.children,
                       key=lambda idx: node.children[idx].ucb(self.c_puct))
        return best_idx, node.children[best_idx]

    def _expand_from_logits(self, node: MCTSNode, board: chess.Board,
                            logits: np.ndarray) -> None:
        """
        Create children from one row of network output. Legal moves are
        generated ONCE; priors come from a softmax over just those entries
        (equivalent to masking illegal moves to -inf, without touching the
        other ~4060 logits). For black, the network sees a mirrored board,
        so we read its output through the mirror table instead of building
        a full mirrored copy.
        """
        legal = list(board.legal_moves)
        idxs  = np.fromiter((move_to_index(m) for m in legal),
                            dtype=np.int64, count=len(legal))
        take  = get_mirror_indices_np()[idxs] if board.turn == chess.BLACK else idxs

        lg = logits[take]
        lg = lg - lg.max()            # stability: softmax is shift-invariant
        priors = np.exp(lg)
        priors /= priors.sum()

        for idx, prior in zip(idxs, priors):
            node.children[int(idx)] = MCTSNode(prior=float(prior), parent=node)
        node.is_expanded = True

    def _apply_virtual_loss(self, path: list) -> None:
        for node in path:
            node.N += 1
            node.W -= VIRTUAL_LOSS

    def _remove_virtual_loss(self, path: list) -> None:
        for node in path:
            node.N -= 1
            node.W += VIRTUAL_LOSS

    def _backup(self, path: list, value: float) -> None:
        for node in reversed(path):
            value = -value   # flip first: value arrives as leaf player's perspective;
            node.N += 1      # each node stores value from the perspective of the player
            node.W += value  # who chose this action (the parent), so flip before storing

    def _terminal_value(self, board: chess.Board) -> float:
        return 0.0 if board.result() == "1/2-1/2" else -1.0

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(root.children))
        for child, n in zip(root.children.values(), noise):
            child.P = (1 - DIRICHLET_EPS) * child.P + DIRICHLET_EPS * n
