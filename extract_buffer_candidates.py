#!/usr/bin/env python3
"""
extract_buffer_candidates.py — Find self-play positions with early material imbalance.

Reads the Run 10 game log, replays decisive games, and extracts positions where
one side is up a rook or more (abs(material) >= 5) between plies 8 and 28
(roughly moves 4–14). These are candidates for the Run 11 permanent buffer.

The material-advantaged side must have gone on to win the game — so each position
carries a reliable ground-truth outcome (±1.0) for value head training.

Output: paper/buffer_candidates.json — structured JSON ready for external agent review.
"""

import chess
import csv
import json
import random

GAMES_CSV    = "logs/run10/games.csv"
OUTPUT_JSON  = "paper/buffer_candidates.json"

MIN_GAME       = 2000   # exclude pre-bug-fix games (policy mirroring fixed at ~game 1700)
MIN_PLY        = 8      # ply 8 = after each side has moved 4 times
MAX_PLY        = 28     # ply 28 = after each side has moved 14 times
MIN_IMBALANCE  = 5      # rook (5) or more
MAX_CANDIDATES = 300    # target output size, sampled if more found

PIECE_VALUES = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
}


def material_balance(board: chess.Board) -> int:
    """Material count from White's perspective. Positive = White ahead."""
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score


def extract_candidate(row: dict):
    """
    Return the first qualifying position from a game, or None.

    Qualifying: abs(material) >= MIN_IMBALANCE at ply MIN_PLY–MAX_PLY,
    and the material-advantaged side is the game's eventual winner.
    """
    if row["end_reason"] not in ("material_resign", "value_resign", "checkmate"):
        return None

    outcome_str = row["outcome"]
    if outcome_str == "W":
        outcome = +1.0
    elif outcome_str == "B":
        outcome = -1.0
    else:
        return None

    board = chess.Board()
    for ply, move_uci in enumerate(row["moves"].split(), start=1):
        try:
            board.push_uci(move_uci)
        except Exception:
            return None

        if ply < MIN_PLY:
            continue
        if ply > MAX_PLY:
            break

        mat = material_balance(board)
        if abs(mat) < MIN_IMBALANCE:
            continue

        # Require: the side currently up in material is the eventual winner.
        # This gives us a reliable ground-truth label.
        if mat > 0 and outcome != +1.0:
            continue
        if mat < 0 and outcome != -1.0:
            continue

        return {
            "game_num":         int(row["game"]),
            "ply":              ply,
            "fen":              board.fen(),
            "material_balance": mat,
            "winner":           "white" if outcome > 0 else "black",
            "outcome":          outcome,
            "end_reason":       row["end_reason"],
        }

    return None


def main():
    candidates = []

    with open(GAMES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["game"]) < MIN_GAME:
                continue
            candidate = extract_candidate(row)
            if candidate:
                candidates.append(candidate)

    print(f"Found {len(candidates)} qualifying positions from {MIN_GAME}+ games")

    if len(candidates) > MAX_CANDIDATES:
        random.seed(42)
        candidates = random.sample(candidates, MAX_CANDIDATES)
        print(f"Sampled down to {MAX_CANDIDATES}")

    candidates.sort(key=lambda c: c["game_num"])

    for i, c in enumerate(candidates):
        c["id"] = i

    with open(OUTPUT_JSON, "w") as f:
        json.dump(candidates, f, indent=2)

    # Summary stats
    white_ahead = sum(1 for c in candidates if c["material_balance"] > 0)
    black_ahead = sum(1 for c in candidates if c["material_balance"] < 0)
    avg_imb = sum(abs(c["material_balance"]) for c in candidates) / len(candidates)
    imb_counts = {}
    for c in candidates:
        v = abs(c["material_balance"])
        imb_counts[v] = imb_counts.get(v, 0) + 1

    print(f"\nCandidates: {len(candidates)}")
    print(f"  White ahead: {white_ahead}  Black ahead: {black_ahead}")
    print(f"  Avg |imbalance|: {avg_imb:.1f}")
    print(f"  Imbalance distribution: { {k: imb_counts[k] for k in sorted(imb_counts)} }")
    print(f"\nOutput: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
