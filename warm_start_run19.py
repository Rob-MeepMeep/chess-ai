"""
warm_start_run19.py — Build run19's starting checkpoint from run18's trained
weights.

No architecture change — run19 escalates Option C's alpha (self-play vs
Stockfish blend weight, see curate_buffer.py's STOCKFISH_ALPHA) rather
than changing the network. Straight copy of run18's network weights into
a fresh ChessAgent, no weight surgery needed. Optimizer state not carried
over (fresh Adam, same reasoning as every prior warm start); `steps`
resets to 0 — run19's own step count, not run18's.

Usage:
  venv/bin/python3 warm_start_run19.py

Run this on the machine holding checkpoints/run18_hal_chess.pt
(checkpoints/ isn't synced via git). Run relabel_with_stockfish.py (if not
already done) and then curate_buffer.py first so
checkpoints/run19_seed_buffer.pt exists too, then launch train_chess.py —
its existing resume logic will pick up this checkpoint automatically
since it's saved directly to CKPT_PATH.
"""

import os
# WSL2 ROCm GPU detection requirement
os.environ["HSA_ENABLE_DXG_DETECTION"] = "1"

import torch

from chessai.agent import ChessAgent

OLD_CKPT = "checkpoints/run18_hal_chess.pt"
NEW_CKPT = "checkpoints/run19_hal_chess.pt"


def main():
    if os.path.exists(NEW_CKPT):
        print(f"{NEW_CKPT} already exists — refusing to overwrite. "
              f"Delete it first if you want to rebuild it.")
        return

    device = torch.device("cpu")   # pure weight copy, no need for the GPU

    print(f"Loading {OLD_CKPT}...")
    old_ckpt = torch.load(OLD_CKPT, map_location=device, weights_only=False)

    print("Building fresh run19 agent (same architecture as run18)...")
    new = ChessAgent(device)
    new.network.load_state_dict(old_ckpt["network"])
    new.steps = 0   # run19's own step count starts fresh

    os.makedirs("checkpoints", exist_ok=True)
    new.save(NEW_CKPT)
    print(f"Saved warm-started checkpoint to {NEW_CKPT}")
    print(f"  Trained steps carried over: 0 (fresh optimizer + step count for run19)")
    print(f"  Network weights: identical to run18's checkpoint")


if __name__ == "__main__":
    main()
