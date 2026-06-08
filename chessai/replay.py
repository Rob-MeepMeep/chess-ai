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
import os
from typing import Optional
import chess
import torch
from collections import deque


class ReplayBuffer:
    """
    Circular buffer of (encoded_state, mcts_policy, outcome) triples.
    Oldest entries are dropped automatically when capacity is reached.

    A second partition — _permanent — holds ground-truth positions (canonical
    endgames, curated seed data) that are never evicted. A fixed fraction of
    each training batch is drawn from the permanent set, keeping the value head
    calibrated on positions that rarely appear in self-play.
    """

    def __init__(self, capacity: int = 50_000):
        self._buffer: deque = deque(maxlen=capacity)
        self._permanent: list = []   # never evicted; sampled alongside rolling buffer

    def extend(self, positions: list) -> None:
        """Add a list of (state, policy, outcome) from one completed game."""
        self._buffer.extend(positions)

    def add_permanent(self, positions: list) -> None:
        """Add positions to the permanent set — never evicted, always in the mix."""
        self._permanent.extend(positions)

    def sample(self, batch_size: int) -> tuple:
        """
        Sample a random batch for training.
        Up to 25% of each batch is drawn from the permanent partition (if populated).
        Returns three stacked tensors: (states, policies, outcomes).
        """
        perm_ratio = 0.25
        perm_size = int(batch_size * perm_ratio)

        actual_perm_size = min(perm_size, len(self._permanent))
        roll_size = batch_size - actual_perm_size

        batch = []
        if actual_perm_size > 0:
            batch.extend(random.sample(self._permanent, actual_perm_size))

        if roll_size > 0 and len(self._buffer) > 0:
            actual_roll_size = min(roll_size, len(self._buffer))
            batch.extend(random.sample(self._buffer, actual_roll_size))

        random.shuffle(batch)

        states, policies, outcomes = zip(*batch)
        return (
            torch.stack(states),                                        # (B, 54, 8, 8)
            torch.stack(policies),                                      # (B, 4096)
            torch.tensor(outcomes, dtype=torch.float32).unsqueeze(1),  # (B, 1)
        )

    def ready(self, batch_size: int) -> bool:
        """True once the rolling buffer has enough entries to sample a full batch."""
        return len(self._buffer) >= batch_size

    def save(self, path: str) -> None:
        """Persist both partitions to disk."""
        torch.save({'rolling': list(self._buffer), 'permanent': self._permanent}, path)

    def load(self, path: str) -> None:
        """Reload a previously saved buffer. Handles legacy format (plain list)."""
        data = torch.load(path, weights_only=False)
        if isinstance(data, dict):
            self._buffer   = deque(data['rolling'],            maxlen=self._buffer.maxlen)
            self._permanent = data.get('permanent', [])
        else:
            # Legacy format saved before tiered buffer — no permanent partition
            self._buffer = deque(data, maxlen=self._buffer.maxlen)

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
               winner: Optional[chess.Color],
               scale: float = 1.0) -> None:
        """
        Game over. Fill in outcomes and move everything to the replay buffer.

        winner: chess.WHITE, chess.BLACK, or None for draw.
        scale:  multiply decisive outcomes by this factor. Use 0.8 for soft wins
                (e.g. cap draws where one side is materially ahead but the game
                wasn't finished — a strong signal without claiming certainty).
        Each position gets ±scale if decisive, 0 for draw.
        """
        completed = []
        for state, policy, turn in self._positions:
            if winner is None:
                outcome = 0.0
            else:
                outcome = scale * (1.0 if turn == winner else -1.0)
            completed.append((state, policy, outcome))

        replay.extend(completed)
        self._positions.clear()

    def __len__(self) -> int:
        return len(self._positions)
