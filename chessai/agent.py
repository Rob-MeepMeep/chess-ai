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

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

from chessai.model  import ChessNet
from chessai.mcts   import MCTS
from chessai.moves  import index_to_move

LR           = 1e-4
WEIGHT_DECAY = 1e-4   # L2 regularisation — prevents overfitting on a deeper network
GRAD_CLIP    = 1.0


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
                    n_simulations: int = None) -> tuple:
        """
        Run MCTS and return (uci_move, policy).

        greedy=False  — sample from the visit-count distribution (training)
        greedy=True   — take the most-visited move (evaluation)

        Returns the chosen UCI move string and the full policy tensor (4096,)
        so the training loop can store it in the GameBuffer.
        """
        n = n_simulations or self.n_simulations
        add_noise = not greedy   # Dirichlet noise only during training

        policy = self.mcts.search(board, history, n, add_noise=add_noise)

        if greedy:
            move_idx = policy.argmax().item()
        else:
            # Sample proportional to visit counts
            move_idx = torch.multinomial(policy, num_samples=1).item()

        move = index_to_move(move_idx, board)
        return move.uci(), policy

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def train(self, batch: tuple) -> float:
        """
        One gradient update on a sampled batch.

        batch: (states, policies, outcomes) — stacked tensors from ReplayBuffer.sample()
          states:   (B, 55, 8, 8)
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
        torch.save({
            "network":   self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps":     self.steps,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps = ckpt["steps"]
        self.network.eval()
