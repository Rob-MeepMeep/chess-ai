"""
generate_material_probe_positions.py — Build the real, in-context position
set chessai/logger.py's material_probe now uses, replacing the old
synthetic "starting position minus one piece" FENs.

Why: see the comment above MATERIAL_PROBE_POSITIONS in chessai/logger.py.
Short version — the old probe used zero-history, zero-moves-played FENs
that can't occur naturally, and the network (correctly) never learned to
read them. diagnose_probe_construction.py proved real in-context positions
at the same magnitudes read cleanly instead.

For each category, this scans a games.csv for positions (sampled the same
way relabel_with_stockfish.py does — before pushing the ply's move, ply
>= MIN_MOVE_PLY) where exactly one piece type differs between sides by
the category's exact amount and every other piece count matches, with
White to move (mirrors the original FENs' "White down material, White to
move" framing). Stores the move sequence leading to each sampled position
(not the encoded tensor) so record_material_probe can replay it and
rebuild real history at read time, identically to how training positions
are built.

This is a one-time data-generation step, not part of the training loop —
re-run it only if you want a fresh/larger sample (e.g. from a later run's
games.csv once more games exist).

Usage:
  venv/bin/python3 generate_material_probe_positions.py [games_csv]
    games_csv defaults to logs/run19/games.csv
"""

import csv
import json
import random
import sys
from typing import Optional

import chess

MIN_MOVE_PLY     = 20   # skip pure opening theory — same threshold relabel_with_stockfish.py uses
N_PER_CATEGORY   = 30
OUTPUT_PATH      = "chessai/material_probe_positions.json"
DEFAULT_GAMES_CSV = "logs/run19/games.csv"

# Fixed order, matching the original MATERIAL_PROBE_POSITIONS dict exactly.
# _init_csv only writes a header if the CSV doesn't exist yet (chessai/logger.py),
# so an in-progress run's material_probe.csv keeps its existing header on
# restart — if this dict's key order didn't match it, every value logged
# after the fix would silently land under the wrong column label instead
# of just measuring something different (which is expected and noted
# separately). Do not reorder without also handling the existing CSV.
CATEGORY_ORDER = [
    "missing_queen", "missing_rook", "missing_bishop", "missing_knight",
    "missing_two_pawns", "black_missing_queen", "black_missing_rook",
]

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


def _counts(board: chess.Board) -> dict:
    return {pt: (len(board.pieces(pt, chess.WHITE)), len(board.pieces(pt, chess.BLACK)))
            for pt in PIECE_TYPES}


def _category_for(board: chess.Board) -> Optional[str]:
    """Which MATERIAL_PROBE_POSITIONS category (if any) this position
    matches: exactly one piece type differs by the category's exact
    amount, every other piece type is equal between sides."""
    if board.turn != chess.WHITE:
        return None

    c = _counts(board)
    same_except = lambda skip: all(
        c[pt][0] == c[pt][1] for pt in PIECE_TYPES if pt != skip
    )
    wp, bp = c[chess.PAWN]
    wn, bn = c[chess.KNIGHT]
    wb, bb = c[chess.BISHOP]
    wr, br = c[chess.ROOK]
    wq, bq = c[chess.QUEEN]

    if wq == bq - 1 and same_except(chess.QUEEN):
        return "missing_queen"
    if wr == br - 1 and same_except(chess.ROOK):
        return "missing_rook"
    if wb == bb - 1 and same_except(chess.BISHOP):
        return "missing_bishop"
    if wn == bn - 1 and same_except(chess.KNIGHT):
        return "missing_knight"
    if wp == bp - 2 and same_except(chess.PAWN):
        return "missing_two_pawns"
    if bq == wq - 1 and same_except(chess.QUEEN):
        return "black_missing_queen"
    if br == wr - 1 and same_except(chess.ROOK):
        return "black_missing_rook"
    return None


def main():
    games_csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GAMES_CSV
    print(f"Scanning {games_csv} (ply >= {MIN_MOVE_PLY})...")

    by_cat = {}
    with open(games_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            moves = row["moves"].split()
            board = chess.Board()
            for ply, move_uci in enumerate(moves, start=1):
                if ply >= MIN_MOVE_PLY:
                    cat = _category_for(board)
                    if cat is not None:
                        by_cat.setdefault(cat, []).append(list(moves[:ply - 1]))
                try:
                    board.push_uci(move_uci)
                except Exception:
                    break

    print("\nYield per category (before capping):")
    result = {}
    random.seed(42)
    for cat in CATEGORY_ORDER:
        seqs = by_cat.get(cat, [])
        random.shuffle(seqs)
        kept = seqs[:N_PER_CATEGORY]
        result[cat] = kept
        print(f"  {cat:>20}: {len(seqs):>5} found, kept {len(kept)}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
