"""
warm_start_run16.py — Build run16's starting checkpoint from run15's trained
weights, rather than training the whole network from scratch a fourth time.

Why this is needed: run16 adds a material-count input plane to the encoder
(generalisation-gap Option A — see paper/generalization_gap_options.md),
changing N_PLANES from 54 to 55. That resizes the network's first conv
layer (input_conv.0.weight: (256, 54, 3, 3) -> (256, 55, 3, 3)), so
run15's checkpoint can't be loaded into a fresh ChessNet as-is.

Every other layer — the entire residual tower, both heads — is untouched
by this change and gets copied across directly. For the one resized layer,
the 54 existing input-channel weights are copied into the matching slice
of the new layer, and the 55th (new material-plane) channel is
zero-initialised — so the new network starts out functionally identical
to run15, and only has to learn to *use* the new plane, not relearn
everything else.

The optimizer state is NOT transferred (Adam's per-parameter momentum
buffers don't transfer cleanly across a resized layer, and it's not worth
the complexity) — run16 starts with a fresh Adam optimizer. `steps` is
reset to 0: this is run16's own step count, not run15's.

Usage:
  venv/bin/python3 warm_start_run16.py

Run this on the machine that actually holds checkpoints/run15_hal_chess.pt
(checkpoints/ isn't synced via git). Run curate_buffer.py first so
checkpoints/run16_seed_buffer.pt exists too, then launch train_chess.py —
its existing resume logic will pick up this checkpoint automatically
since it's saved directly to CKPT_PATH.
"""

import os
# WSL2 ROCm GPU detection requirement
os.environ["HSA_ENABLE_DXG_DETECTION"] = "1"

import torch

from chessai.agent import ChessAgent

OLD_CKPT = "checkpoints/run15_hal_chess.pt"
NEW_CKPT = "checkpoints/run16_hal_chess.pt"
RESIZED_LAYER = "input_conv.0.weight"
OLD_N_PLANES = 54


def main():
    if os.path.exists(NEW_CKPT):
        print(f"{NEW_CKPT} already exists — refusing to overwrite. "
              f"Delete it first if you want to rebuild it.")
        return

    device = torch.device("cpu")   # pure weight surgery, no need for the GPU

    print(f"Loading {OLD_CKPT}...")
    # Load the raw checkpoint dict rather than going through ChessAgent.load() —
    # ChessAgent builds a ChessNet() internally, which now imports N_PLANES=55
    # (encoder.py already has the new plane), so there is no way to instantiate
    # the *old* 54-plane architecture anymore to load run15's weights into.
    # The state dict itself doesn't care what shape the current code expects;
    # only load_state_dict()'s shape-checking does, so we bypass that entirely
    # for the old side and work with its raw tensors directly.
    old_ckpt = torch.load(OLD_CKPT, map_location=device, weights_only=False)
    old_sd   = old_ckpt["network"]

    print("Building fresh run16 network (55-plane input)...")
    new = ChessAgent(device)
    new_sd = new.network.state_dict()

    if RESIZED_LAYER not in old_sd or RESIZED_LAYER not in new_sd:
        raise RuntimeError(f"Expected layer {RESIZED_LAYER} not found — "
                            f"has the model architecture changed?")

    for key in new_sd:
        if key == RESIZED_LAYER:
            old_w = old_sd[key]            # (256, 54, 3, 3)
            new_w = new_sd[key].clone()    # (256, 55, 3, 3), freshly initialised
            new_w[:, :OLD_N_PLANES] = old_w
            new_w[:, OLD_N_PLANES:] = 0.0  # new material-plane channel starts blank
            new_sd[key] = new_w
            print(f"  {key}: copied {old_w.shape} into {new_w.shape}, "
                  f"channel {OLD_N_PLANES} zero-initialised")
        else:
            new_sd[key] = old_sd[key]

    new.network.load_state_dict(new_sd)
    new.steps = 0   # run16's own step count starts fresh

    os.makedirs("checkpoints", exist_ok=True)
    new.save(NEW_CKPT)
    print(f"Saved warm-started checkpoint to {NEW_CKPT}")
    print(f"  Trained steps carried over: 0 (fresh optimizer + step count for run16)")
    print(f"  All other weights: identical to run15's checkpoint")


if __name__ == "__main__":
    main()
