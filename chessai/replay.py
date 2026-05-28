"""
replay.py — Trajectory storage for AlphaZero-style training.

Two classes work together:

  GameBuffer  — accumulates positions from one game as it's being played.
                Outcome unknown until game end, so we store whose turn it was.

  ReplayBuffer — the main circular store. Holds completed (state, policy,
                 outcome) triples across many games. Training samples batches
                 from here.

The outcome flip: we always store outcome from the perspective of the player
to move at that position. If white won and it was white's turn → +1.
If white won and it was black's turn → -1. Same two-player logic as MCTS backup.
"""

import random
from typing import Optional
import chess
import torch
from collections import deque


class ReplayBuffer:
    """
    Circular buffer of (encoded_state, mcts_policy, outcome) triples.
    Oldest entries are dropped automatically when capacity is reached.
    """

    def __init__(self, capacity: int = 50_000):
        self._buffer: deque = deque(maxlen=capacity)

    def extend(self, positions: list) -> None:
        """Add a list of (state, policy, outcome) from one completed game."""
        self._buffer.extend(positions)

    def sample(self, batch_size: int) -> tuple:
        """
        Sample a random batch for training.
        Returns three stacked tensors: (states, policies, outcomes).
        """
        batch               = random.sample(self._buffer, batch_size)
        states, policies, outcomes = zip(*batch)
        return (
            torch.stack(states),                                   # (B, 55, 8, 8)
            torch.stack(policies),                                 # (B, 4096)
            torch.tensor(outcomes, dtype=torch.float32).unsqueeze(1),  # (B, 1)
        )

    def ready(self, batch_size: int) -> bool:
        """True once the buffer has enough entries to sample a full batch."""
        return len(self._buffer) >= batch_size

    def __len__(self) -> int:
        return len(self._buffer)


class GameBuffer:
    """
    Temporary store for one game in progress.
    Positions are held here until the game ends, then committed to ReplayBuffer
    with the correct perspective-relative outcome filled in.
    """

    def __init__(self):
        self._positions: list = []   # (state, policy, turn)

    def push(self, state: torch.Tensor, policy: torch.Tensor,
             turn: chess.Color) -> None:
        """
        Record one position.
        turn: chess.WHITE or chess.BLACK — whose move it was.
        """
        self._positions.append((state, policy, turn))

    def commit(self, replay: ReplayBuffer,
               winner: Optional[chess.Color]) -> None:
        """
        Game over. Fill in outcomes and move everything to the replay buffer.

        winner: chess.WHITE, chess.BLACK, or None for draw.
        Each position gets +1 if the player to move won, -1 if they lost, 0 for draw.
        """
        completed = []
        for state, policy, turn in self._positions:
            if winner is None:
                outcome = 0.0
            else:
                outcome = 1.0 if turn == winner else -1.0
            completed.append((state, policy, outcome))

        replay.extend(completed)
        self._positions.clear()

    def __len__(self) -> int:
        return len(self._positions)
