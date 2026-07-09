"""
model.py — Residual neural network for chess.

Architecture (AlphaZero-style, scaled for MacBook Pro M5 Pro):

  Input: (batch, 54, 8, 8) — encoded board state from encoder.py

  Input conv:    Conv2d(54→160, 3×3) + BatchNorm + ReLU
  Residual ×10:  two conv layers with a skip connection (see ResidualBlock)
  Policy head:   Conv2d(128→2, 1×1) + BatchNorm + ReLU → flatten → Linear(128, 4096)
  Value head:    Conv2d(128→1, 1×1) + BatchNorm + ReLU → flatten → Linear(64, 256) → Linear(1) → Tanh

The policy head outputs raw logits (4096 values, one per possible move).
Illegal moves are masked to -inf by the agent before argmax or softmax.

The value head outputs a scalar in [-1, 1]: positive = current player is winning.
This is what MCTS uses to evaluate leaf nodes without playing the game to completion.

AlphaZero uses 20 residual blocks and 256 filters. We use 10 blocks and 160 filters —
about 3× smaller — running on a MacBook Pro M5 Pro with 24GB unified memory.
"""

import torch
import torch.nn as nn

from chessai.encoder import N_PLANES   # 54

N_CHANNELS = 256
N_BLOCKS   = 20
N_MOVES    = 4096   # 64 × 64 flat move encoding


class ResidualBlock(nn.Module):
    """
    Two conv layers with a skip connection.
    The skip connection lets gradients flow straight back to early layers,
    which is what makes it practical to stack 8+ blocks without instability.

    bias=False throughout because BatchNorm already includes a learnable
    offset (beta), making the conv bias redundant.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))   # skip connection


class ChessNet(nn.Module):

    def __init__(self, n_blocks: int = N_BLOCKS, channels: int = N_CHANNELS):
        super().__init__()

        # Project from 54 input planes to the working channel width
        self.input_conv = nn.Sequential(
            nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # Residual tower — where spatial pattern recognition happens
        self.tower = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(n_blocks)]
        )

        # Policy head — compress to 2 channels, flatten, project to 4096 logits
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),   # 1×1 conv = channel reduction
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, N_MOVES),
        )

        # Value head — compress to 1 channel, flatten, two-layer MLP → scalar
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),   # constrain to [-1, 1] — consistent with +1 win / -1 loss
        )

    def forward(self, x: torch.Tensor):
        """
        x: (batch, 54, 8, 8)
        Returns: (policy_logits, value)
          policy_logits: (batch, 4096) — raw, not softmaxed
          value:         (batch, 1)    — in [-1, 1]
        """
        features = self.tower(self.input_conv(x))
        return self.policy_head(features), self.value_head(features)
