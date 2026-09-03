"""
diagnose_network_output.py — Does the trained network actually track the
Stockfish target at each material magnitude, or ignore it?

diagnose_material_labels.py established that a real, correctly-signed
target exists even at |mat|=2-3 (mean -0.17 to -0.37) -- just noisier than
|mat|=5-7 (mean -0.67 to -0.83, much tighter std). That alone doesn't say
whether material_probe's near-zero reading for missing_bishop/two_pawns
means the network ignores the signal entirely, or is weakly tracking it
but shrunk toward zero (which is what plain MSE regression does to a
noisy target, and wouldn't be a training failure so much as expected
behaviour).

This runs the current checkpoint's network forward on every cached
Stockfish-relabelled position, grouped by the same signed material
buckets, and reports:
  - pred mean/std next to the true sf_value mean/std (shrinkage, if any)
  - Pearson correlation between the network's prediction and sf_value
    within each bucket (near 0 = not differentiating these positions at
    all; positive and comparable across buckets = differentiating fine,
    just regressing toward the mean more where the target is noisier)

Usage:
  venv/bin/python3 diagnose_network_output.py
"""

import os
os.environ["HSA_ENABLE_DXG_DETECTION"] = "1"   # WSL2 ROCm GPU detection requirement

import statistics
from collections import defaultdict

import torch

from chessai.model import ChessNet

RAW_PATH  = "checkpoints/stockfish_relabeled_raw.pt"
CKPT_PATH = "checkpoints/run19_hal_chess.pt"
BATCH_SIZE = 256


def _pearson(xs: list, ys: list) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print(f"Loading {CKPT_PATH}...")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    net = ChessNet().to(device)
    net.load_state_dict(ckpt["network"])
    net.eval()
    print(f"  steps: {ckpt.get('steps', '?')}")

    print(f"Loading {RAW_PATH}...")
    raw = torch.load(RAW_PATH, weights_only=False)
    print(f"{len(raw):,} positions\n")

    by_mag = defaultdict(lambda: {"state": [], "sf": []})
    for state, _policy, _z, sf_value in raw:
        mat_signed = round(state[54, 0, 0].item() * 20)
        by_mag[mat_signed]["state"].append(state)
        by_mag[mat_signed]["sf"].append(sf_value)

    header = (f"{'mat':>5} {'n':>6}  {'sf mean':>8} {'sf std':>7}  "
              f"{'pred mean':>9} {'pred std':>8}  {'corr':>6}")
    print(header)
    print("-" * len(header))

    for mat_signed in sorted(by_mag):
        states  = by_mag[mat_signed]["state"]
        sf_vals = by_mag[mat_signed]["sf"]

        preds = []
        with torch.inference_mode():
            for i in range(0, len(states), BATCH_SIZE):
                chunk = torch.stack(states[i:i + BATCH_SIZE]).to(device)
                _, values = net(chunk)
                preds.extend(values.squeeze(1).cpu().tolist())

        sf_mean = statistics.mean(sf_vals)
        sf_std  = statistics.pstdev(sf_vals) if len(sf_vals) > 1 else 0.0
        pred_mean = statistics.mean(preds)
        pred_std  = statistics.pstdev(preds) if len(preds) > 1 else 0.0
        corr = _pearson(preds, sf_vals)

        print(f"{mat_signed:>5} {len(sf_vals):>6}  {sf_mean:>+8.3f} {sf_std:>7.3f}  "
              f"{pred_mean:>+9.3f} {pred_std:>8.3f}  {corr:>+6.3f}")

    print(
        "\nRead this as: correlation near 0 at a given magnitude means the "
        "network isn't differentiating those positions at all -- a real "
        "training/architecture gap. Correlation comparable to the strong "
        "magnitudes (5-7) but pred mean noticeably smaller than sf mean "
        "means it IS tracking the signal, just regressing toward zero more "
        "-- expected behaviour under a noisier target, not obviously a bug."
    )


if __name__ == "__main__":
    main()
