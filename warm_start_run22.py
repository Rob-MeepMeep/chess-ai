"""
warm_start_run22.py — Build run22's starting checkpoint from run21's
trained weights.

No architecture change. Unlike every prior warm start, the DATA pipeline
does change this time: run22 is meant to train against a buffer that's
been extended with the Option 2 hanging-piece continuation data (see
extend_buffer_with_hanging_pieces.py and paper/value_head_small_
material_options.md Sec 6-11) -- but the network weights themselves are
a straight copy of run21's, same as every prior warm start. Optimizer
state not carried over (fresh Adam); `steps` resets to 0 -- run22's own
step count, not run21's.

Usage:
  venv/bin/python3 warm_start_run22.py

Run this on the machine holding checkpoints/run21_hal_chess.pt
(checkpoints/ isn't synced via git). Run extend_buffer_with_hanging_
pieces.py first (or make sure checkpoints/run22_seed_buffer.pt already
exists) -- train_chess.py's BUFFER_LOAD points at that file for run22,
not at run21's own accumulated buffer directly. Then launch
train_chess.py -- its existing resume logic picks up this checkpoint
automatically since it's saved directly to CKPT_PATH.
"""

import os
# WSL2 ROCm GPU detection requirement
os.environ["HSA_ENABLE_DXG_DETECTION"] = "1"

import torch

from chessai.agent import ChessAgent

OLD_CKPT = "checkpoints/run21_hal_chess.pt"
NEW_CKPT = "checkpoints/run22_hal_chess.pt"


def main():
    if os.path.exists(NEW_CKPT):
        print(f"{NEW_CKPT} already exists — refusing to overwrite. "
              f"Delete it first if you want to rebuild it.")
        return

    device = torch.device("cpu")   # pure weight copy, no need for the GPU

    print(f"Loading {OLD_CKPT}...")
    old_ckpt = torch.load(OLD_CKPT, map_location=device, weights_only=False)

    print("Building fresh run22 agent (same architecture as run21)...")
    new = ChessAgent(device)
    new.network.load_state_dict(old_ckpt["network"])
    new.steps = 0   # run22's own step count starts fresh

    os.makedirs("checkpoints", exist_ok=True)
    new.save(NEW_CKPT)
    print(f"Saved warm-started checkpoint to {NEW_CKPT}")
    print(f"  Trained steps carried over: 0 (fresh optimizer + step count for run22)")
    print(f"  Network weights: identical to run21's checkpoint")


if __name__ == "__main__":
    main()
