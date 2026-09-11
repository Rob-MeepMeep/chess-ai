"""
extend_buffer_with_hanging_pieces.py — Fold the new hanging-piece
Stockfish-relabelled continuation data (Option 2) into run21's existing
accumulated buffer, in place, rather than rebuilding a buffer from
scratch.

Why this, not curate_buffer.py: curate_buffer.py rebuilds the ROLLING
partition from scratch by replaying a single games.csv, discarding
whatever else was accumulated — exactly right for bootstrapping a
fresh, randomly-initialised network (run19's original purpose), wrong
here: run22 warm-starts from run21's already-trained weights, and the
200,000-entry rolling buffer accumulated across run19+20+21 shouldn't
be thrown away just to add ~4,000 new permanent positions.

Confirmed before writing this (12 Sept 2026): run21's own startup log
already reports "200,000 rolling + 9,750 permanent" — and 9,750 exactly
matches 45 static canonical + 512 generated endgames + 193 reviewed
midgame + 9,000 from the ORIGINAL relabel_with_stockfish.py Option C
pass (checked directly against paper/buffer_candidates_reviewed.json's
accepted count and curate_buffer.py's own generation constants, not
assumed). That data was baked in at run19's bootstrap and has carried
forward unchanged ever since — nothing since has touched the permanent
partition. So this script does NOT need to regenerate or re-fold that
original set; it's already there. It only adds the NEW hanging-piece
positions on top.

Blend: same alpha-blend-at-fold-time design as curate_buffer.py, reusing
its HANGING_PIECE_ALPHA constant so both scripts share one tunable value
rather than duplicating it.

Usage:
  venv/bin/python3 extend_buffer_with_hanging_pieces.py

Run this on the machine holding checkpoints/run21_replay_buffer.pt
(checkpoints/ isn't synced via git). Needs
checkpoints/stockfish_relabeled_hanging_pieces_raw.pt to already be in
place (see relabel_hanging_piece_continuations.py).
"""

import os

import torch

from chessai.replay import ReplayBuffer
from curate_buffer import BUFFER_CAPACITY, HANGING_PIECE_RAW_PATH, HANGING_PIECE_ALPHA

INPUT_BUFFER  = "checkpoints/run21_replay_buffer.pt"
OUTPUT_BUFFER = "checkpoints/run22_seed_buffer.pt"


def main():
    if not os.path.exists(INPUT_BUFFER):
        print(f"{INPUT_BUFFER} not found — run this on the machine holding it.")
        return
    if not os.path.exists(HANGING_PIECE_RAW_PATH):
        print(f"{HANGING_PIECE_RAW_PATH} not found — run "
              f"relabel_hanging_piece_continuations.py first, or transfer "
              f"its output here.")
        return

    print(f"Loading {INPUT_BUFFER}...")
    buf = ReplayBuffer(capacity=BUFFER_CAPACITY)
    buf.load(INPUT_BUFFER)
    print(f"  {len(buf):,} rolling / {len(buf._permanent):,} permanent (before)")

    print(f"Loading {HANGING_PIECE_RAW_PATH}...")
    raw = torch.load(HANGING_PIECE_RAW_PATH, weights_only=False)
    print(f"  {len(raw):,} raw hanging-piece positions")

    new_permanent = []
    for state, policy, z, sf_value in raw:
        blended = HANGING_PIECE_ALPHA * z + (1 - HANGING_PIECE_ALPHA) * sf_value
        new_permanent.append((state, policy, float(blended)))

    buf.add_permanent(new_permanent)
    print(f"  {len(buf):,} rolling / {len(buf._permanent):,} permanent (after)")

    buf.save(OUTPUT_BUFFER)
    print(f"\nSaved to {OUTPUT_BUFFER}")
    print(f"For run22, set in train_chess.py:")
    print(f'  BUFFER_LOAD = "{OUTPUT_BUFFER}"')


if __name__ == "__main__":
    main()
