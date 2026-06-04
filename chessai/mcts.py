"""
mcts.py — Batched Monte Carlo Tree Search guided by the neural network.

Each simulation does four things:
  1. SELECT   — walk the tree using UCB until an unexpanded node
  2. EXPAND   — ask the network for priors over all legal moves at that node
  3. EVALUATE — ask the network's value head: who is winning here?
  4. BACKUP   — propagate the value back up the path, flipping sign at each
                level (what's good for me is bad for my opponent)

BATCHING:
  Instead of running simulations one at a time (50 separate network calls),
  we run BATCH_SIMS simulations in parallel up to their leaf nodes, then
  evaluate all leaves in one network call. Reduces MPS round-trips from 50
  to ~7 (50 / BATCH_SIMS), keeping the GPU busy.

VIRTUAL LOSS:
  To prevent all parallel simulations selecting the same path, we apply a
  temporary penalty to in-flight nodes so other simulations choose different
  branches. Removed when the real value is backed up.

Expected speedup over sequential MCTS: 2–4× on MPS.
"""

import numpy as np
import torch
import chess

from chessai.encoder import encode
from chessai.moves  import move_to_index, index_to_move, legal_move_mask

C_PUCT          = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS   = 0.25
BATCH_SIMS      = 8      # leaves evaluated per network call
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


class MCTS:

    def __init__(self, network: torch.nn.Module, device: torch.device,
                 c_puct: float = C_PUCT):
        self.network = network
        self.device  = device
        self.c_puct  = c_puct
        self.network.eval()

    def search(self, board: chess.Board, history: list,
               n_simulations: int, add_noise: bool = False) -> torch.Tensor:
        """
        Run n_simulations from the current position using batched leaf evaluation.
        Returns a policy tensor of shape (4096,), normalised visit counts.
        """
        root = MCTSNode(prior=1.0)
        self._expand_node(root, board, history)

        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        sims_done = 0
        while sims_done < n_simulations:
            wave = min(BATCH_SIMS, n_simulations - sims_done)

            # --- 1. Select leaves for this wave ---
            wave_data = []   # list of (path, sim_board, sim_history)
            for _ in range(wave):
                path, sim_board, sim_hist = self._select_to_leaf(
                    root, board.copy(), list(history)
                )
                wave_data.append((path, sim_board, sim_hist))
                self._apply_virtual_loss(path)

            # --- 2. Batch-expand unique non-terminal leaves ---
            # leaf node = path[-1]; use id() to deduplicate shared leaves
            node_values: dict[int, float] = {}
            to_expand: dict[int, tuple]   = {}   # node_id → (node, board, hist)

            for path, sim_board, sim_hist in wave_data:
                leaf    = path[-1]
                node_id = id(leaf)
                if (not sim_board.is_game_over()
                        and not leaf.is_expanded
                        and node_id not in to_expand):
                    to_expand[node_id] = (leaf, sim_board, sim_hist)

            if to_expand:
                node_ids = list(to_expand.keys())
                leaves   = [to_expand[nid][0] for nid in node_ids]
                boards   = [to_expand[nid][1] for nid in node_ids]
                hists    = [to_expand[nid][2] for nid in node_ids]

                # Single batched network call
                batch_tensor = torch.cat(
                    [encode([b] + h).unsqueeze(0) for b, h in zip(boards, hists)],
                    dim=0
                ).to(self.device)

                with torch.no_grad():
                    logits_batch, value_batch = self.network(batch_tensor)

                for i, (node_id, node, sim_board) in enumerate(
                        zip(node_ids, leaves, boards)):
                    logits = logits_batch[i].clone()
                    mask   = legal_move_mask(sim_board).to(self.device)
                    logits[~mask] = float("-inf")
                    priors = torch.softmax(logits, dim=0)

                    for move in sim_board.legal_moves:
                        idx = move_to_index(move)
                        node.children[idx] = MCTSNode(
                            prior=priors[idx].item(), parent=node
                        )
                    node.is_expanded   = True
                    node_values[node_id] = value_batch[i].item()

            # --- 3. Remove virtual loss and backup ---
            for path, sim_board, sim_hist in wave_data:
                self._remove_virtual_loss(path)
                leaf    = path[-1]
                node_id = id(leaf)

                if sim_board.is_game_over():
                    value = self._terminal_value(sim_board)
                else:
                    value = node_values[node_id]

                self._backup(path, value)

            sims_done += wave

        # Normalise visit counts → policy
        policy = torch.zeros(4096)
        for idx, child in root.children.items():
            policy[idx] = child.N
        total = policy.sum()
        if total > 0:
            policy = policy / total
        return policy

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

    def _expand_node(self, node: MCTSNode, board: chess.Board,
                     history: list) -> None:
        """Synchronous single-node expansion — used for the root only."""
        encoded = encode([board] + list(history)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, _ = self.network(encoded)
        policy_logits = policy_logits.squeeze(0)
        mask = legal_move_mask(board).to(self.device)
        policy_logits[~mask] = float("-inf")
        priors = torch.softmax(policy_logits, dim=0)
        for move in board.legal_moves:
            idx = move_to_index(move)
            node.children[idx] = MCTSNode(prior=priors[idx].item(), parent=node)
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
