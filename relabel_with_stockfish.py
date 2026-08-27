"""
relabel_with_stockfish.py — Build a large, diverse set of positions labelled
with real Stockfish evaluations, targeting the specific material range
self-play has now failed twice to teach on its own.

Why: rung 1 (run15) and rung 1b (run17) both showed self-play can only ever
reinforce large (8+) material imbalances — the adjudication threshold
structurally filters out smaller ones, no matter how it's tuned, so no
volume of additional self-play games can generate training signal for
them. This is Option C from paper/generalization_gap_options.md: sample
positions from HAL's own self-play (nothing invented, nothing externally
sourced) specifically in the range that's been failing, and label each one
with a real Stockfish evaluation instead of relying on the noisy
whole-game outcome.

Design choices, and why (see chat, 27 August, and
paper/run17_decision_protocol.md for the diagnosis this responds to):

  - Targets |material| in [1, 7] — the full range below the strong
    adjudication tier's 8+ threshold, wider than rung 1b's 3-7 moderate
    band, since rung 1b's failure was a coverage problem, not evidence
    that this specific sub-range doesn't matter.
  - Large target size (~9,000 positions) deliberately, not a small curated
    set — run13's original bug was exactly a small, repeated permanent
    partition getting memorised rather than teaching a general rule.
    Avoid repeating that failure mode with Stockfish labels instead of
    hand labels.
  - SELF_PLAY_WEIGHT is low (0.2): rung 1b showed a weak/absent signal
    doesn't move these positions, so this test should give Stockfish's
    judgement a strong, largely undiluted voice rather than a diluted one
    that risks another ambiguous result.
  - Output feeds curate_buffer.py's permanent partition, not the rolling
    buffer — the same mechanism that already successfully teaches the
    canonical K+Q vs K endgames, for repeated (not one-time) exposure.

Usage:
  venv/bin/python3 relabel_with_stockfish.py

Requires Stockfish on PATH. One-time offline cost, not part of the
training loop — do not call this from train_chess.py; Stockfish evaluation
takes real wall-clock time per position and would wreck the lockstep
loop's throughput.
"""

import os
# WSL2 ROCm GPU detection requirement
os.environ["HSA_ENABLE_DXG_DETECTION"] = "1"

import math
import random

import chess
import chess.engine
import torch

from chessai.encoder import encode

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GAMES_CSVS       = ["logs/run16/games.csv", "logs/run17/games.csv"]
OUTPUT_PATH      = "checkpoints/run18_stockfish_positions.pt"
STOCKFISH_PATH   = "stockfish"
STOCKFISH_DEPTH  = 16   # deeper than eval_chess.py's deliberately-weak depth-1
                        # eval opponent — this is a one-time labelling cost,
                        # quality matters more than speed here.

TARGET_LOW            = 1   # inclusive
TARGET_HIGH           = 7   # inclusive — 8+ is already well-covered by the strong tier
MIN_MOVE_PLY          = 20  # skip pure opening theory
MAX_SAMPLES_PER_GAME  = 4   # cap so no single game dominates the set
BALANCED_SAMPLE_RATE  = 0.15  # also sample some near-even positions, so the
                              # network doesn't learn "nonzero plane = decisive"

SELF_PLAY_WEIGHT = 0.2   # alpha — blend weight on the position's own game
                         # outcome; (1 - alpha) on the Stockfish evaluation.

TARGET_POSITIONS = 9_000

_PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                 chess.ROOK: 5, chess.QUEEN: 9}


def _material_balance(board: chess.Board) -> int:
    """Positive = white ahead. Same definition used throughout the project."""
    score = 0
    for piece, val in _PIECE_VALUES.items():
        score += val * len(board.pieces(piece, chess.WHITE))
        score -= val * len(board.pieces(piece, chess.BLACK))
    return score


def _stockfish_value(engine: chess.engine.SimpleEngine, board: chess.Board,
                     depth: int) -> float:
    """Evaluation from the side-to-move's perspective, scaled to [-1, 1] —
    matches encode()'s current-player-perspective convention exactly."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].relative
    if score.is_mate():
        mate_in = score.mate()
        return 0.99 if mate_in > 0 else -0.99
    return math.tanh(score.score() / 400)


def _load_games(paths):
    games = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  {path} not found, skipping")
            continue
        n = 0
        with open(path, newline="") as f:
            next(f)  # header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                # Same dual-format handling as curate_buffer.py — old rows
                # (6 cols) vs new rows (8 cols, steps at index 5)
                if parts[5].isdigit():
                    moves = " ".join(parts[7:]).split()
                else:
                    moves = " ".join(parts[5:]).split()
                outcome = parts[1]
                if outcome not in ("W", "B"):
                    continue
                games.append({"outcome": outcome, "moves": moves})
                n += 1
        print(f"  {path}: {n:,} decisive games")
    return games


def main():
    print("Loading self-play games...")
    games = _load_games(GAMES_CSVS)
    random.shuffle(games)
    print(f"  {len(games):,} total games available\n")

    print(f"Starting Stockfish (depth {STOCKFISH_DEPTH})...")
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 1, "Hash": 64})

    positions = []
    evaluated = 0

    try:
        for game in games:
            if len(positions) >= TARGET_POSITIONS:
                break

            board = chess.Board()
            history = []
            winner = chess.WHITE if game["outcome"] == "W" else chess.BLACK
            candidates = []   # (board_copy, history_copy, mover, mat_abs)

            for ply, move_uci in enumerate(game["moves"], start=1):
                # Sample BEFORE pushing — this is the position the network
                # would actually need to evaluate, same convention as
                # curate_buffer.py's own game replay.
                if ply >= MIN_MOVE_PLY:
                    mat_abs = abs(_material_balance(board))
                    in_target_band = TARGET_LOW <= mat_abs <= TARGET_HIGH
                    is_balanced_sample = (mat_abs == 0
                                          and random.random() < BALANCED_SAMPLE_RATE)
                    if in_target_band or is_balanced_sample:
                        candidates.append((board.copy(), list(history), board.turn, mat_abs))

                history = ([board.copy()] + history)[:3]
                try:
                    board.push_uci(move_uci)
                except Exception:
                    break

            random.shuffle(candidates)
            for pos_board, pos_history, mover, mat_abs in candidates[:MAX_SAMPLES_PER_GAME]:
                state  = encode([pos_board] + pos_history)
                policy = torch.zeros(4096)

                z = 1.0 if mover == winner else -1.0
                sf_value = _stockfish_value(engine, pos_board, STOCKFISH_DEPTH)
                blended = SELF_PLAY_WEIGHT * z + (1 - SELF_PLAY_WEIGHT) * sf_value

                positions.append((state, policy, float(blended), mat_abs))
                evaluated += 1
                if evaluated % 200 == 0:
                    print(f"  {evaluated:,} positions evaluated "
                          f"({len(positions):,} kept)...")

                if len(positions) >= TARGET_POSITIONS:
                    break
    finally:
        engine.quit()

    print(f"\nFinal set: {len(positions):,} positions")
    from collections import Counter
    band_counts = Counter(mat_abs for _, _, _, mat_abs in positions)
    print("Material-magnitude coverage:")
    for mat_abs in sorted(band_counts):
        print(f"  |material|={mat_abs}: {band_counts[mat_abs]:,}")

    # Strip the diagnostic mat_abs column before saving — downstream
    # (curate_buffer.py, add_permanent) expects plain (state, policy, value)
    save_positions = [(s, p, v) for s, p, v, _ in positions]
    torch.save(save_positions, OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Run curate_buffer.py next — it will pick this file up automatically "
          f"if present at {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
