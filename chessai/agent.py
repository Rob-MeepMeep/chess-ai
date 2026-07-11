"""
agent.py — ChessAgent: the network and MCTS combined into one object.

Two responsibilities:

  choose_move() — runs MCTS, picks a move from the resulting policy.
                  Stochastic during training (sample from visit distribution),
                  deterministic during evaluation (argmax).

  train()       — gradient update on a batch from the replay buffer.
                  Two losses combined: policy (cross-entropy with MCTS target)
                  and value (MSE against game outcome).

No target network — MCTS provides the training stability that the target
network provided in DQN. The search always looks ahead, so a single
overconfident network prediction can't corrupt the training signal.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

from chessai.model   import ChessNet
from chessai.mcts    import MCTS
from chessai.moves   import index_to_move
from chessai.encoder import encode

LR           = 1e-4
WEIGHT_DECAY = 1e-4   # L2 regularisation — prevents overfitting on a deeper network
GRAD_CLIP    = 1.0
TEMP_MOVES   = 30     # plies — sample stochastically for the first 30 half-moves (opening),
                      # then switch to greedy (argmax) so endgame lines are always decisive


class ChessAgent:

    def __init__(self, device: torch.device,
                 n_simulations: int = 200,
                 lr: float = LR):
        self.device       = device
        self.n_simulations = n_simulations
        self.steps        = 0

        self.network   = ChessNet().to(device)
        self.mcts      = MCTS(self.network, device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=lr, weight_decay=WEIGHT_DECAY
        )

    # ------------------------------------------------------------------
    # Playing
    # ------------------------------------------------------------------

    def choose_move(self, board: chess.Board, history: list,
                    greedy: bool = False,
                    n_simulations: int = None,
                    add_noise: bool = None) -> tuple:
        """
        Run MCTS and return (uci_move, policy, value_after).

        greedy=False  — stochastic sampling for the first TEMP_MOVES plies (opening
                        diversity), then argmax thereafter (decisive endgame signal).
        greedy=True   — always argmax (evaluation / no training noise).
        add_noise     — override Dirichlet noise at MCTS root. Defaults to not greedy.
                        Set True with greedy=True for diverse-but-strong eval games.

        Returns the chosen UCI move string, the full policy tensor (4096,) for
        the GameBuffer, and value_after: the search's estimate of the position
        AFTER the chosen move, from the perspective of the player then to move.
        This is the same quantity the old per-move get_value() call measured,
        but backed by a full search instead of one raw network guess — and it
        costs nothing extra, so the resign check no longer needs its own
        network call every ply.
        """
        n = n_simulations or self.n_simulations
        if add_noise is None:
            add_noise = not greedy

        policy, root = self.mcts.search(board, history, n, add_noise=add_noise)
        return self.pick_move(board, policy, root, greedy=greedy)

    def pick_move(self, board: chess.Board, policy: torch.Tensor,
                  root, greedy: bool = False) -> tuple:
        """
        Turn a finished search (policy + root) into a move. Split out from
        choose_move so the lockstep training loop, which drives the search
        steps itself, can share the exact same selection logic.
        """
        ply = len(board.move_stack)
        if greedy or ply >= TEMP_MOVES:
            # Greedy: take the most-visited move
            move_idx = policy.argmax().item()
        else:
            # Stochastic: sample proportional to visit counts
            move_idx = torch.multinomial(policy, num_samples=1).item()

        move  = index_to_move(move_idx, board)
        child = root.children.get(move_idx)
        # child.Q is from the mover's perspective; after the move it is the
        # opponent's turn, so their view of the position is the negation.
        value_after = -child.Q if (child is not None and child.N > 0) else 0.0
        return move.uci(), policy, value_after

    def get_value(self, board: chess.Board, history: list) -> float:
        """Single forward pass value estimate. Used by regression logging."""
        encoded = encode([board] + list(history)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            _, value = self.network(encoded)
        return value.item()

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def train(self, batch: tuple) -> float:
        """
        One gradient update on a sampled batch.

        batch: (states, policies, outcomes) — stacked tensors from ReplayBuffer.sample()
          states:   (B, 54, 8, 8)
          policies: (B, 4096)   — MCTS visit-count targets
          outcomes: (B, 1)      — game result from each position's perspective

        Returns the scalar loss value for logging.
        """
        states, target_policies, target_values = batch
        states          = states.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values   = target_values.to(self.device)

        self.network.train()
        pred_logits, pred_values = self.network(states)

        # Policy loss: cross-entropy with soft target (MCTS distribution)
        # = -sum(target × log_softmax(logits)) averaged over the batch
        log_probs   = F.log_softmax(pred_logits, dim=1)
        policy_loss = -(target_policies * log_probs).sum(dim=1).mean()

        # Value loss: MSE against game outcome
        value_loss = F.mse_loss(pred_values, target_values)

        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), GRAD_CLIP)
        self.optimizer.step()

        self.steps += 1
        self.network.eval()

        return loss.item()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        torch.save({
            "network":   self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps":     self.steps,
        }, tmp)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps = ckpt["steps"]
        self.network.eval()
