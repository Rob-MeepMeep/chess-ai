"""
warm_start_run17.py — Build run17's starting checkpoint from run16's trained
weights.

Unlike warm_start_run16.py, no architecture change is involved this time —
intervention ladder rung 1b just adds a second, lower-confidence material
adjudication tier to train_chess.py's game-ending logic (see run_config.py
and the MATERIAL_ADJUDICATE_MODERATE_* constants in train_chess.py). That's
purely a training-signal change, not a network change, so this is a
straight copy of run16's network weights into a fresh ChessAgent — no
weight surgery needed. The optimizer state is not carried over (fresh
Adam, same reasoning as run16's warm start) and `steps` resets to 0 —
run17's own step count, not run16's.

Usage:
  venv/bin/python3 warm_start_run17.py

Run this on the machine holding checkpoints/run16_hal_chess.pt
(checkpoints/ isn't synced via git). Run curate_buffer.py first so
checkpoints/run17_seed_buffer.pt exists too, then launch train_chess.py —
its existing resume logic will pick up this checkpoint automatically
since it's saved directly to CKPT_PATH.
"""

import os
# WSL2 ROCm GPU detection requirement
os.environ["HSA_ENABLE_DXG_DETECTION"] = "1"

import torch

from chessai.agent import ChessAgent

OLD_CKPT = "checkpoints/run16_hal_chess.pt"
NEW_CKPT = "checkpoints/run17_hal_chess.pt"


def main():
    if os.path.exists(NEW_CKPT):
        print(f"{NEW_CKPT} already exists — refusing to overwrite. "
              f"Delete it first if you want to rebuild it.")
        return

    device = torch.device("cpu")   # pure weight copy, no need for the GPU

    print(f"Loading {OLD_CKPT}...")
    old_ckpt = torch.load(OLD_CKPT, map_location=device, weights_only=False)

    print("Building fresh run17 agent (same architecture as run16)...")
    new = ChessAgent(device)
    new.network.load_state_dict(old_ckpt["network"])
    new.steps = 0   # run17's own step count starts fresh

    os.makedirs("checkpoints", exist_ok=True)
    new.save(NEW_CKPT)
    print(f"Saved warm-started checkpoint to {NEW_CKPT}")
    print(f"  Trained steps carried over: 0 (fresh optimizer + step count for run17)")
    print(f"  Network weights: identical to run16's checkpoint")


if __name__ == "__main__":
    main()
