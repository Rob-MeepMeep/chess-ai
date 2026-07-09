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

STORAGE FORMAT:
  Policies are stored SPARSELY as (indices, values) pairs. A chess position
  has ~30 legal moves, so a dense 4096-float policy is ~99% zeros — sparse
  storage cuts the buffer from ~6GB to ~2GB at 200k capacity, which matters
  both for RAM and for how long each buffer save blocks the training loop.
  sample() densifies on the way out, so the training code sees the same
  (B, 4096) tensors as before.

  The rolling store is a plain list used as a ring (write cursor wraps at
  capacity) rather than a deque: deque indexing is O(n), which made
  random.sample over 200k entries do tens of millions of pointer hops per
  batch. List indexing is O(1).
"""

import random
from typing import Optional
import chess
import torch

N_MOVES = 4096

# Bumped when the on-disk layout changes. load() still reads both older
# layouts (dict-of-dense and bare-list) and converts them on the fly.
_SAVE_FORMAT = 2


def _sparsify(policy: torch.Tensor):
    """Dense (4096,) tensor → (indices int16, values float32). No-op if
    already sparse. int16 is safe: move indices are 0–4095."""
    if isinstance(policy, tuple):
        return policy
    nz = torch.nonzero(policy, as_tuple=True)[0]
    return (nz.to(torch.int16), policy[nz].to(torch.float32))


def _densify(policy) -> torch.Tensor:
    """(indices, values) → dense (4096,) tensor. No-op if already dense."""
    if not isinstance(policy, tuple):
        return policy
    idx, val = policy
    dense = torch.zeros(N_MOVES)
    if idx.numel():
        dense[idx.long()] = val
    return dense


def _sparsify_all(positions: list) -> list:
    return [(state, _sparsify(policy), outcome)
            for state, policy, outcome in positions]


class ReplayBuffer:
    """
    Circular buffer of (encoded_state, mcts_policy, outcome) triples.
    Oldest entries are overwritten when capacity is reached.

    A second partition — _permanent — holds ground-truth positions (canonical
    endgames, curated seed data) that are never evicted. A fixed fraction of
    each training batch is drawn from the permanent set, keeping the value head
    calibrated on positions that rarely appear in self-play.
    """

    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self._buffer: list = []
        self._pos = 0                # ring write cursor, used once full
        self._permanent: list = []   # never evicted; sampled alongside rolling buffer

    def _append(self, item) -> None:
        if len(self._buffer) < self.capacity:
            self._buffer.append(item)
        else:
            self._buffer[self._pos] = item
            self._pos = (self._pos + 1) % self.capacity

    def extend(self, positions: list) -> None:
        """Add a list of (state, policy, outcome) from one completed game."""
        for item in _sparsify_all(positions):
            self._append(item)

    def add_permanent(self, positions: list) -> None:
        """Add positions to the permanent set — never evicted, always in the mix."""
        self._permanent.extend(_sparsify_all(positions))

    def sample(self, batch_size: int) -> tuple:
        """
        Sample a random batch for training.
        Up to 33% of each batch is drawn from the permanent partition (if populated).
        Returns three stacked tensors: (states, policies, outcomes).
        """
        perm_ratio = 0.33
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
            torch.stack([_densify(p) for p in policies]),               # (B, 4096)
            torch.tensor(outcomes, dtype=torch.float32).unsqueeze(1),  # (B, 1)
        )

    def ready(self, batch_size: int) -> bool:
        """True once the rolling buffer has enough entries to sample a full batch."""
        return len(self._buffer) >= batch_size

    def save(self, path: str) -> None:
        """Persist both partitions to disk (sparse policies — format 2)."""
        torch.save({
            'format':    _SAVE_FORMAT,
            'rolling':   self._buffer,
            'permanent': self._permanent,
        }, path)

    def load(self, path: str) -> None:
        """Reload a saved buffer. Reads all three historical layouts:
        format-2 (sparse), format-1 dict (dense), and the original bare list."""
        data = torch.load(path, weights_only=False)
        if isinstance(data, dict):
            rolling   = data['rolling']
            permanent = data.get('permanent', [])
            if data.get('format', 1) < 2:
                rolling   = _sparsify_all(rolling)
                permanent = _sparsify_all(permanent)
            self._buffer    = list(rolling)[-self.capacity:]
            self._permanent = list(permanent)
        else:
            # Legacy format saved before the tiered buffer — dense, no permanent
            self._buffer = _sparsify_all(list(data)[-self.capacity:])
        self._pos = 0   # entries are oldest→newest, so overwriting starts at 0

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
