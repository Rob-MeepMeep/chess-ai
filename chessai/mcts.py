"""
mcts.py — Monte Carlo Tree Search guided by the neural network.

Each simulation does four things:
  1. SELECT   — walk the tree using UCB until an unexpanded node
  2. EXPAND   — ask the network for priors over all legal moves at that node
  3. EVALUATE — ask the network's value head: who is winning here?
  4. BACKUP   — propagate the value back up the path, flipping sign at each
                level (what's good for me is bad for my opponent)

After N simulations the visit counts are normalised to a policy. More visited
= the network and the search together think this move is better.

UCB formula (controls the select step):
  UCB(s, a) = Q(s,a)  +  c_puct × P(s,a) × √N(s) / (1 + N(s,a))
               exploit          explore

  Q is the average value seen from this action so far.
  The exploration term is large when a move has a high prior but few visits —
  it shrinks as the node accumulates visits, letting Q take over.
"""

import numpy as np
import torch
import chess

from chessai.encoder import encode
from chessai.moves  import move_to_index, index_to_move, legal_move_mask

C_PUCT          = 1.5    # exploration constant — higher = more exploration
DIRICHLET_ALPHA = 0.3    # concentration for root noise (AlphaZero chess value)
DIRICHLET_EPS   = 0.25   # fraction of prior replaced by noise at the root


class MCTSNode:
    """
    One node in the search tree — one board position.

    N : visit count
    W : total value accumulated across visits (sum, not average)
    P : prior probability from the network (its first guess, before search)
    Q = W / N — average value (computed on demand)
    """

    __slots__ = ("N", "W", "P", "parent", "children", "is_expanded")

    def __init__(self, prior: float, parent: "MCTSNode | None" = None):
        self.N          = 0
        self.W          = 0.0
        self.P          = prior
        self.parent     = parent
        self.children: dict[int, "MCTSNode"] = {}
        self.is_expanded = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def ucb(self, c_puct: float) -> float:
        # Parent's visit count drives the exploration term —
        # as the parent accumulates visits, less-visited children look more attractive
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
        Run n_simulations from the current position.

        board   : current chess.Board (not modified)
        history : list of recent chess.Board states, most recent first
                  (does not include the current board — encoder prepends it)
        add_noise : True during training to encourage exploration at the root;
                    False during evaluation for pure exploitation

        Returns: policy tensor of shape (4096,), normalised visit counts.
        """
        # Expand root before simulations so all children have priors
        root = MCTSNode(prior=1.0)
        self._expand(root, board, history)

        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        for _ in range(n_simulations):
            node        = root
            sim_board   = board.copy()
            sim_history = list(history)
            path        = [node]

            # --- 1. SELECT ---
            # Descend the tree, choosing the highest-UCB child at each level,
            # until we reach an unexpanded node or a terminal position.
            while node.is_expanded and not sim_board.is_game_over():
                move_idx, node = self._select(node)
                move = index_to_move(move_idx, sim_board)
                # Prepend board BEFORE the move so history stays in order
                sim_history = [sim_board.copy()] + sim_history
                sim_board.push(move)
                path.append(node)

            # --- 2 + 3. EXPAND + EVALUATE ---
            if sim_board.is_game_over():
                value = self._terminal_value(sim_board)
            else:
                value = self._expand(node, sim_board, sim_history)

            # --- 4. BACKUP ---
            self._backup(path, value)

        # Normalise visit counts → policy
        policy = torch.zeros(4096)
        for idx, child in root.children.items():
            policy[idx] = child.N
        total = policy.sum()
        if total > 0:
            policy = policy / total
        return policy

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        """Return the child with the highest UCB score."""
        best_idx = max(node.children,
                       key=lambda idx: node.children[idx].ucb(self.c_puct))
        return best_idx, node.children[best_idx]

    def _expand(self, node: MCTSNode, board: chess.Board,
                history: list) -> float:
        """
        Call the network on this position.
        Create a child node for each legal move, initialised with network priors.
        Returns the value estimate for this position (from the current player's view).
        """
        encoded = encode([board] + list(history)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.network(encoded)

        # Mask illegal moves to -inf, then softmax over legal ones
        policy_logits = policy_logits.squeeze(0)
        mask = legal_move_mask(board).to(self.device)
        policy_logits[~mask] = float("-inf")
        priors = torch.softmax(policy_logits, dim=0)

        for move in board.legal_moves:
            idx = move_to_index(move)
            node.children[idx] = MCTSNode(prior=priors[idx].item(), parent=node)

        node.is_expanded = True
        return value.item()

    def _backup(self, path: list, value: float) -> None:
        """
        Walk back up the path, adding the value at each node.
        Flip the sign at every level — the two-player flip.
        """
        for node in reversed(path):
            node.N += 1
            node.W += value
            value   = -value

    def _terminal_value(self, board: chess.Board) -> float:
        """
        Value for a finished game, from the perspective of the player to move.
        After checkmate the player to move was just mated — they lost: -1.
        Any draw: 0.
        """
        result = board.result()
        if result == "1/2-1/2":
            return 0.0
        return -1.0   # checkmate: player to move lost

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        """
        Mix Dirichlet noise into the root's priors.
        Prevents the search from always exploring the same lines early in training
        when the network's priors are still unreliable.
        """
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(root.children))
        for child, n in zip(root.children.values(), noise):
            child.P = (1 - DIRICHLET_EPS) * child.P + DIRICHLET_EPS * n
