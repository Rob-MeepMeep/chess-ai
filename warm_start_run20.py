"""
warm_start_run20.py — Build run20's starting checkpoint from run19's
trained weights.

No architecture change, no data-pipeline change — run19's material_probe
result that looked like a training gap turned out to be a measurement
bug (see paper/material_probe_correction.md). run20 exists only to give
the corrected material_probe a clean CSV from game 0; it is otherwise a
straight continuation of run19. Straight copy of run19's network weights
into a fresh ChessAgent, no weight surgery needed. Optimizer state not
carried over (fresh Adam, same reasoning as every prior warm start);
`steps` resets to 0 -- run20's own step count, not run19's.

Usage:
  venv/bin/python3 warm_start_run20.py

Run this on the machine holding checkpoints/run19_hal_chess.pt
(checkpoints/ isn't synced via git). No curate_buffer.py run needed --
train_chess.py's BUFFER_LOAD already points at run19's accumulated
buffer (checkpoints/run19_replay_buffer.pt), carrying forward all of
run19's self-play data rather than reverting to the original curated
seed. Then launch train_chess.py -- its existing resume logic picks up
this checkpoint automatically since it's saved directly to CKPT_PATH.
"""

import os
# WSL2 ROCm GPU detection requirement
os.environ["HSA_ENABLE_DXG_DETECTION"] = "1"

import torch

from chessai.agent import ChessAgent

OLD_CKPT = "checkpoints/run19_hal_chess.pt"
NEW_CKPT = "checkpoints/run20_hal_chess.pt"


def main():
    if os.path.exists(NEW_CKPT):
        print(f"{NEW_CKPT} already exists — refusing to overwrite. "
              f"Delete it first if you want to rebuild it.")
        return

    device = torch.device("cpu")   # pure weight copy, no need for the GPU

    print(f"Loading {OLD_CKPT}...")
    old_ckpt = torch.load(OLD_CKPT, map_location=device, weights_only=False)

    print("Building fresh run20 agent (same architecture as run19)...")
    new = ChessAgent(device)
    new.network.load_state_dict(old_ckpt["network"])
    new.steps = 0   # run20's own step count starts fresh

    os.makedirs("checkpoints", exist_ok=True)
    new.save(NEW_CKPT)
    print(f"Saved warm-started checkpoint to {NEW_CKPT}")
    print(f"  Trained steps carried over: 0 (fresh optimizer + step count for run20)")
    print(f"  Network weights: identical to run19's checkpoint")


if __name__ == "__main__":
    main()
