"""
diagnose_probe_construction.py — Does the network's material-deficit signal
disappear only because material_probe's test positions are unrealistic?

chessai/logger.py's MATERIAL_PROBE_POSITIONS (run16+) test the network on
the starting position with one piece deleted, zero moves played, empty
history -- a board state that cannot occur naturally (removing a piece
requires a capture, which requires history). diagnose_network_output.py
already showed the current checkpoint tracks Stockfish's material-deficit
judgement well (corr 0.90-0.96) on REAL mid-game positions with real
history, at every magnitude including bishop/knight/two-pawns. Those two
facts together suggest material_probe's own construction, not training,
is why it reads near zero for the smaller pieces.

This runs both side by side on the SAME checkpoint:
  (a) the exact synthetic FENs material_probe.csv has been logging,
      exactly as record_material_probe() calls them (empty history)
  (b) real positions at the same material magnitudes, freshly sampled
      from run19's own games.csv with their actual game history intact
      -- independent of the Stockfish-relabelled permanent-partition set
      diagnose_network_output.py used, so this isn't just re-measuring
      the same (possibly memorised) positions again

If (b) reads strongly negative/positive where (a) reads near zero, at the
same nominal magnitude, that confirms the gap is in how material_probe
constructs its test positions, not a training failure.

Usage:
  venv/bin/python3 diagnose_probe_construction.py
"""

import os
os.environ["HSA_ENABLE_DXG_DETECTION"] = "1"   # WSL2 ROCm GPU detection requirement

import csv
import random
import statistics
from collections import defaultdict

import chess
import torch

from chessai.model   import ChessNet
from chessai.encoder import encode
from chessai.logger  import MATERIAL_PROBE_POSITIONS

CKPT_PATH   = "checkpoints/run19_hal_chess.pt"
GAMES_CSV   = "logs/run19/games.csv"
MIN_MOVE_PLY = 20    # skip pure opening theory -- same threshold relabel_with_stockfish.py uses
SAMPLES_PER_MAGNITUDE = 300
BATCH_SIZE = 128

_PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                 chess.ROOK: 5, chess.QUEEN: 9}

# which material magnitude each synthetic probe category corresponds to,
# for lining the two methods up side by side
PROBE_MAGNITUDE = {
    "missing_queen": 9, "missing_rook": 5, "missing_bishop": 3,
    "missing_knight": 3, "missing_two_pawns": 2,
}


def _material_balance(board: chess.Board) -> int:
    score = 0
    for piece, val in _PIECE_VALUES.items():
        score += val * len(board.pieces(piece, chess.WHITE))
        score -= val * len(board.pieces(piece, chess.BLACK))
    return score


def _load_network(device):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    net = ChessNet().to(device)
    net.load_state_dict(ckpt["network"])
    net.eval()
    return net, ckpt.get("steps", "?")


def _predict(net, device, states: list) -> list:
    preds = []
    with torch.inference_mode():
        for i in range(0, len(states), BATCH_SIZE):
            chunk = torch.stack(states[i:i + BATCH_SIZE]).to(device)
            _, values = net(chunk)
            preds.extend(values.squeeze(1).cpu().tolist())
    return preds


def sample_real_positions(path: str) -> dict:
    """Replay every game in path, collect (state) samples bucketed by
    signed mover-perspective material magnitude, same sampling convention
    (sample before push, ply threshold) as relabel_with_stockfish.py."""
    by_mag = defaultdict(list)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            moves = row["moves"].split()
            board = chess.Board()
            history = []

            for ply, move_uci in enumerate(moves, start=1):
                if ply >= MIN_MOVE_PLY:
                    mat_white = _material_balance(board)
                    mat_mover = mat_white if board.turn == chess.WHITE else -mat_white
                    if mat_mover != 0:
                        state = encode([board.copy()] + history)
                        by_mag[mat_mover].append(state)

                history = ([board.copy()] + history)[:3]
                try:
                    board.push_uci(move_uci)
                except Exception:
                    break

    # cap and shuffle per bucket so results aren't dominated by one game
    for mag in by_mag:
        random.shuffle(by_mag[mag])
        by_mag[mag] = by_mag[mag][:SAMPLES_PER_MAGNITUDE]

    return by_mag


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print(f"Loading {CKPT_PATH}...")
    net, steps = _load_network(device)
    print(f"  steps: {steps}")

    print(f"Replaying {GAMES_CSV} for real, in-context positions "
          f"(ply >= {MIN_MOVE_PLY})...")
    by_mag = sample_real_positions(GAMES_CSV)
    total = sum(len(v) for v in by_mag.values())
    print(f"  {total:,} real positions collected across "
          f"{len(by_mag)} magnitude buckets\n")

    print("--- (a) synthetic material_probe FENs, empty history ---")
    header_a = f"{'category':>18} {'mag':>4}  {'synthetic pred':>15}"
    print(header_a)
    print("-" * len(header_a))
    synthetic_preds = {}
    for key, fen in MATERIAL_PROBE_POSITIONS.items():
        if key not in PROBE_MAGNITUDE:
            continue
        board = chess.Board(fen)
        state = encode([board])
        pred = _predict(net, device, [state])[0]
        synthetic_preds[key] = pred
        print(f"{key:>18} {PROBE_MAGNITUDE[key]:>4}  {pred:>+15.4f}")

    print("\n--- (b) real in-context positions, real history ---")
    header_b = f"{'mag':>4} {'n':>5}  {'real pred mean':>15} {'real pred std':>14}"
    print(header_b)
    print("-" * len(header_b))
    real_stats = {}
    for mag in sorted(by_mag, key=abs):
        states = by_mag[mag]
        if not states:
            continue
        preds = _predict(net, device, states)
        mean = statistics.mean(preds)
        std = statistics.pstdev(preds) if len(preds) > 1 else 0.0
        real_stats[mag] = (mean, std, len(preds))
        print(f"{mag:>4} {len(preds):>5}  {mean:>+15.4f} {std:>14.4f}")

    print("\n--- side by side, negative-magnitude (down material) categories ---")
    header_c = f"{'category':>18} {'mag':>4}  {'synthetic':>10}  {'real mean':>10} {'n':>5}"
    print(header_c)
    print("-" * len(header_c))
    for key, mag in PROBE_MAGNITUDE.items():
        neg_mag = -mag
        synth = synthetic_preds.get(key)
        real = real_stats.get(neg_mag)
        if synth is None or real is None:
            continue
        print(f"{key:>18} {neg_mag:>4}  {synth:>+10.4f}  "
              f"{real[0]:>+10.4f} {real[2]:>5}")

    print(
        "\nIf the 'real mean' column reads clearly negative while "
        "'synthetic' reads near zero at the same magnitude, material_probe "
        "has been measuring an out-of-distribution construction artifact, "
        "not a training failure -- the network already knows what a "
        "missing bishop/knight/two-pawns means, just not in a position "
        "shape (zero history, virgin opening) it has never seen paired "
        "with a material deficit."
    )


if __name__ == "__main__":
    main()
