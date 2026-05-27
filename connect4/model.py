"""
model.py — HAL-3000's neural network (CNN with policy + value heads).

Architecture overview:
    Input:   2 × 6 × 7 tensor  (two binary channels: my pieces, opponent's pieces)
    Conv:    Two convolutional layers extract spatial patterns
    Flatten: Into a single feature vector
    Heads:   Two separate output heads from the same features:
        Policy head → 7 Q-values (one per column) — used for move selection
        Value head  → 1 scalar in [-1, 1]          — how good is this position?

Why CNN instead of fully-connected?
    Fully-connected layers treat every cell as unrelated to every other cell.
    A CNN slides a small window (kernel) across the board, looking for local
    patterns — three in a row, diagonal threats — at every position.
    The same pattern-detector works anywhere on the board. This is called
    weight sharing and makes the network much better at spatial games.

Why two output heads?
    The policy head is the same as Phase 1 Q-values — pick the best column.
    The value head rates the overall position, not individual moves.
    We don't use it yet, but MCTS (Monte Carlo Tree Search, used in chess)
    needs it. Adding it now is a few lines; retrofitting it later breaks
    the entire training loop.

Why two input channels?
    Raw board has values 0 (empty), 1 (player 1), 2 (player 2).
    A CNN works better with clean binary signals, so we split into:
        Channel 0: cells where MY pieces are  (1 = yes, 0 = no)
        Channel 1: cells where OPPONENT'S pieces are
    This is the same encoding AlphaZero uses.
"""

import torch
import torch.nn as nn


def encode_state(state, player):
    """
    Convert a flat 42-tuple from env into a (2, 6, 7) tensor.

    state:  42-tuple of ints (0=empty, 1=player1, 2=player2)
    player: 1 or 2 — which player we are right now

    Returns a float tensor with shape (2, 6, 7):
        [0] — 1.0 where my pieces are, 0.0 elsewhere
        [1] — 1.0 where opponent's pieces are, 0.0 elsewhere
    """
    opponent = 3 - player  # if I'm 1, opponent is 2; if I'm 2, opponent is 1
    board = torch.tensor(state, dtype=torch.float32).reshape(6, 7)
    my_channel  = (board == player).float()    # binary mask for my pieces
    opp_channel = (board == opponent).float()  # binary mask for opponent's pieces
    return torch.stack([my_channel, opp_channel])  # shape: (2, 6, 7)


class ConnectFourNet(nn.Module):
    """
    CNN with two output heads for Connect Four.

    Input shape:  (batch, 2, 6, 7)
    Output:
        policy — (batch, 7)  Q-values per column
        value  — (batch, 1)  position score in [-1, 1]
    """

    def __init__(self):
        super().__init__()

        # Convolutional backbone — extracts spatial features from the board.
        # kernel_size=3, padding=1 keeps the spatial dimensions unchanged
        # so a 6×7 board stays 6×7 through both conv layers.
        self.conv = nn.Sequential(
            # 2 input channels → 32 feature maps
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # 32 → 64 feature maps
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # After conv layers: 64 channels × 6 rows × 7 cols = 2,688 values
        conv_flat_size = 64 * 6 * 7

        # Policy head — produces a Q-value for each of the 7 columns.
        # Same role as the output of the old fully-connected network.
        self.policy_head = nn.Sequential(
            nn.Linear(conv_flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, 7),  # no activation — raw Q-values
        )

        # Value head — produces a single score for the whole position.
        # Tanh squashes the output to [-1, 1]:
        #     +1.0 = certain win for me
        #     -1.0 = certain loss
        #      0.0 = roughly equal
        self.value_head = nn.Sequential(
            nn.Linear(conv_flat_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        x: tensor of shape (batch, 2, 6, 7)

        Returns:
            policy — (batch, 7)  Q-values, one per column
            value  — (batch, 1)  position evaluation
        """
        features = self.conv(x)               # (batch, 64, 6, 7)
        features = features.flatten(start_dim=1)  # (batch, 2688)

        policy = self.policy_head(features)   # (batch, 7)
        value  = self.value_head(features)    # (batch, 1)

        return policy, value

    def predict(self, state, player, device):
        """
        Get Q-values and position score for a single board state.
        Used during gameplay and evaluation — not during batch training.

        state:  42-tuple from env
        player: 1 or 2 — which player HAL is right now
        device: torch.device (MPS or CPU)

        Returns:
            q_values — list of 7 floats (Q-value per column)
            value    — float in [-1, 1] (position score)
        """
        with torch.no_grad():
            # Encode and add batch dimension: (2, 6, 7) → (1, 2, 6, 7)
            x = encode_state(state, player).unsqueeze(0).to(device)
            policy, value = self.forward(x)
            return policy.squeeze(0).tolist(), value.item()
