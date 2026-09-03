"""
diagnose_material_labels.py — Check whether the Stockfish-relabelled
positions actually carry a clean, learnable signal at each material
magnitude, before spending another run escalating STOCKFISH_ALPHA again.

Why: run19's material_probe shows missing_queen/missing_rook trained well
but missing_bishop/knight/two_pawns stayed flat near zero at two different
alpha settings (0.2, 0.05). The working theory going in was "Stockfish's
centipawn-scale signal is softer at small magnitudes and gets outcompeted."
But tanh(cp/400) says otherwise: ~300cp (a bishop) -> 0.64, only moderately
smaller than ~500cp (a rook) -> 0.85. That gap doesn't obviously explain
"rook partially works, bishop doesn't work at all."

This script checks the raw cached labels directly, no retraining needed:
for each material magnitude, is sf_value (a) reliably signed and (b) low
variance (a real target the network is failing to hit), or (c) small and
noisy (there may not be a clean material-only answer to learn at that
magnitude, and a near-zero output is the statistically sane prediction)?
It also prints the raw bucket sizes, since real overweighting of one
magnitude over another would independently explain part of the gap.

mat_abs isn't saved in the raw file (relabel_with_stockfish.py drops it
before writing) but doesn't need to be recomputed from board state: plane
54 of the encoded tensor already IS material balance, current player's
perspective, clipped to +-20 and scaled to [-1, 1] (encoder.py, Option A,
run16+) -- reading it back off is just undoing that scaling.

Usage:
  venv/bin/python3 diagnose_material_labels.py
"""

import statistics
from collections import defaultdict

import torch

RAW_PATH = "checkpoints/stockfish_relabeled_raw.pt"
STOCKFISH_ALPHA = 0.05   # mirrors curate_buffer.py's current setting


def main():
    print(f"Loading {RAW_PATH}...")
    raw = torch.load(RAW_PATH, weights_only=False)
    print(f"{len(raw):,} positions\n")

    by_mag = defaultdict(lambda: {"z": [], "sf": []})

    for state, _policy, z, sf_value in raw:
        # plane 54 is a broadcast scalar plane -- any cell holds the value.
        # Signed, not abs() -- "mover is down a bishop" and "mover is up a
        # bishop" are opposite targets and must not be averaged together.
        mat_scaled = state[54, 0, 0].item()
        mat_signed = round(mat_scaled * 20)
        by_mag[mat_signed]["z"].append(z)
        by_mag[mat_signed]["sf"].append(sf_value)

    header = (f"{'mat':>6} {'n':>6}  {'z mean':>8} {'z std':>7}  "
              f"{'sf mean':>8} {'sf std':>7}  {'blended mean':>13}")
    print(header)
    print("-" * len(header))

    for mat_abs in sorted(by_mag):
        z_vals = by_mag[mat_abs]["z"]
        sf_vals = by_mag[mat_abs]["sf"]
        n = len(z_vals)

        z_mean = statistics.mean(z_vals)
        z_std = statistics.pstdev(z_vals) if n > 1 else 0.0
        sf_mean = statistics.mean(sf_vals)
        sf_std = statistics.pstdev(sf_vals) if n > 1 else 0.0

        blended = [STOCKFISH_ALPHA * z + (1 - STOCKFISH_ALPHA) * sf
                   for z, sf in zip(z_vals, sf_vals)]
        blended_mean = statistics.mean(blended)

        print(f"{mat_abs:>6} {n:>6}  {z_mean:>+8.3f} {z_std:>7.3f}  "
              f"{sf_mean:>+8.3f} {sf_std:>7.3f}  {blended_mean:>+13.3f}")

    print(
        "\nRead this as: sf mean far from 0 with a small sf std at a given "
        "|mat| means there IS a clean target there for the network to hit "
        "-- if material_probe still shows ~0 for that magnitude, that "
        "points at training dynamics (batch composition, output "
        "saturation), not the label. A small sf mean or large sf std means "
        "Stockfish itself isn't confident from material alone at that "
        "magnitude -- a near-zero trained output may be the statistically "
        "correct answer, not a training failure, and no amount of alpha "
        "tuning or more games will change that."
    )


if __name__ == "__main__":
    main()
