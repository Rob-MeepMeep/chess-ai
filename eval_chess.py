"""
eval_chess.py — Evaluate HAL-4000's chess performance.

Three tiers of evaluation:

  1. HAL vs Random (both colours) — sanity check, should win 80%+
  2. HAL vs Stockfish depth 1, 3, 5 — real benchmark against known-strength opponent
     depth 1 ≈ 1000–1200 ELO, depth 3 ≈ 1500–1800, depth 5 ≈ 2000+
  3. HAL vs previous checkpoint — measures improvement between training runs
     (skipped if only one checkpoint exists)

Usage:
  python3 eval_chess.py                        # evaluate current checkpoint
  python3 eval_chess.py --regression-only      # value head check only — ~5s, safe during training
  python3 eval_chess.py --cpu                  # force CPU — keeps MPS free for concurrent training
  python3 eval_chess.py --prev checkpoints/hal_chess_v1.pt   # also run improvement test

Note on game counts: N_GAMES_RANDOM and N_GAMES_STOCKFISH are set conservatively
so the full suite can run alongside an active training loop without dominating the
MPS backend. Bump them to 100/50 for the final paper benchmark after training ends.
"""

import argparse
import csv
import os
import random
import time
import chess
import chess.engine
import torch

from chessai.agent   import ChessAgent
from chessai.encoder import encode

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_GAMES_RANDOM     = 25    # bump to 100 for final paper benchmark
N_GAMES_STOCKFISH  = 25    # 25 per matchup — enough for meaningful numbers in a proper benchmark
N_GAMES_PREV       = 25    # games vs previous checkpoint
N_SIMS_RANDOM      = 50    # sims vs random — lower is fine, random doesn't punish weak play
# Per-depth sim counts — higher depths warrant more search to have any chance
SIMS_BY_DEPTH      = {1: 200, 3: 500, 5: 500}
MAX_GAME_MOVES     = 200   # hard cap
CKPT_PATH          = "checkpoints/run9_hal_chess.pt"
STOCKFISH_PATH     = "stockfish"   # assumes stockfish is on PATH

# ---------------------------------------------------------------------------
# Arguments — parsed early so --cpu affects device selection
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--prev", default=None,
                    help="Path to previous checkpoint for improvement test")
parser.add_argument("--regression-only", action="store_true",
                    help="Run value head regression test only — ~5s, safe during active training")
parser.add_argument("--cpu", action="store_true",
                    help="Force CPU device — keeps MPS free when running alongside a training loop")
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

if args.cpu:
    device = torch.device("cpu")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if args.cpu:
    print("Device: cpu (forced — MPS reserved for training)")

# ---------------------------------------------------------------------------
# Load HAL
# ---------------------------------------------------------------------------

hal = ChessAgent(device, n_simulations=N_SIMS_RANDOM)
hal.load(CKPT_PATH)
print(f"Loaded HAL-4000")
print(f"  Trained steps: {hal.steps:,}")
print(f"  Checkpoint:    {CKPT_PATH}\n")

# ---------------------------------------------------------------------------
# Value head regression test
# Checks whether the value head has learned to distinguish positions.
# A draw-collapsed network outputs ~0 everywhere.
# A trained network should approach +1 / -1 on decisive endgames.
# ---------------------------------------------------------------------------

REGRESSION_POSITIONS = {
    # FEN                                                    description          expect
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": ("start",              "~0.0"),
    "8/8/8/8/8/6K1/6Q1/7k w - - 0 1":                            ("K+Q vs K (w wins)", "near +1"),
    "8/8/8/8/8/6K1/6Q1/7k b - - 0 1":                            ("K+Q vs K (b move)", "near -1"),
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1": ("white missing queen", "< 0"),
}

print("── Value Head Regression ───────────────────────────────────\n")
for fen, (label, expected) in REGRESSION_POSITIONS.items():
    board = chess.Board(fen)
    v = hal.get_value(board, [])
    print(f"  {label:<24}  value={v:+.4f}  (expect {expected})")
print()

# ---------------------------------------------------------------------------
# Move functions
# ---------------------------------------------------------------------------

def hal_move(board: chess.Board, history: list) -> str:
    """HAL plays greedily at N_SIMS_RANDOM — fast, sufficient against random."""
    move_uci, _ = hal.choose_move(board, history, greedy=True)
    return move_uci

def hal_move_at(n_sims: int):
    """Return a move function that runs HAL at the given simulation count."""
    def _move(board: chess.Board, history: list) -> str:
        orig = hal.n_simulations
        hal.n_simulations = n_sims
        move_uci, _ = hal.choose_move(board, history, greedy=True)
        hal.n_simulations = orig
        return move_uci
    return _move

def random_move(board: chess.Board, history: list) -> str:
    return random.choice([m.uci() for m in board.legal_moves])

def stockfish_move(engine: chess.engine.SimpleEngine, depth: int):
    """Returns a move function that uses Stockfish at the given depth."""
    def _move(board: chess.Board, history: list) -> str:
        result = engine.play(board, chess.engine.Limit(depth=depth))
        return result.move.uci()
    return _move

# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def play_game(white_fn, black_fn):
    """
    Play one game. Returns (result, move_list).
    result is '1-0', '0-1', '1/2-1/2', or '*' (move cap hit).
    """
    board     = chess.Board()
    history   = []
    move_list = []

    while not board.is_game_over() and len(move_list) < MAX_GAME_MOVES:
        if board.turn == chess.WHITE:
            move_uci = white_fn(board, history)
        else:
            move_uci = black_fn(board, history)

        history = ([board.copy()] + history)[:3]
        board.push_uci(move_uci)
        move_list.append(move_uci)

    result = board.result() if board.is_game_over() else "*"
    return result, move_list

# ---------------------------------------------------------------------------
# Eval game log — one row per game, written to logs/run8/eval_games.csv
# ---------------------------------------------------------------------------

_EVAL_LOG_PATH = os.path.join("logs", "run9", "eval_games.csv")
_eval_game_num = 0   # sequential across the whole eval run

def _init_eval_log() -> None:
    os.makedirs(os.path.dirname(_EVAL_LOG_PATH), exist_ok=True)
    if not os.path.exists(_EVAL_LOG_PATH):
        with open(_EVAL_LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow([
                "eval_game", "matchup", "result", "n_moves",
                "hal_steps", "timestamp", "moves",
            ])

def _log_eval_game(matchup: str, result: str, move_list: list) -> None:
    global _eval_game_num
    _eval_game_num += 1
    with open(_EVAL_LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            _eval_game_num, matchup, result, len(move_list),
            hal.steps, time.strftime("%Y-%m-%d %H:%M:%S"),
            " ".join(move_list),
        ])

# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate(label: str, white_fn, black_fn, n: int) -> dict:
    """Run n games, print results, return stats dict."""
    white_wins = black_wins = draws = 0

    for i in range(n):
        result, move_list = play_game(white_fn, black_fn)
        _log_eval_game(label, result, move_list)

        if result == "1-0":
            white_wins += 1
        elif result == "0-1":
            black_wins += 1
        else:
            draws += 1

        # Progress dot every 10 games
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n}...", end="\r")

    print(f"{label}")
    print(f"  White wins: {white_wins:>4} ({white_wins/n*100:5.1f}%)")
    print(f"  Black wins: {black_wins:>4} ({black_wins/n*100:5.1f}%)")
    print(f"  Draws:      {draws:>4} ({draws/n*100:5.1f}%)")
    print()

    return {"white_wins": white_wins, "black_wins": black_wins, "draws": draws, "n": n}

# ---------------------------------------------------------------------------
# Run all matchups
# ---------------------------------------------------------------------------

if args.regression_only:
    print("=" * 60)
    print("Done.")
    exit(0)

_init_eval_log()
print("=" * 60)
print(f"Eval games logged to: {_EVAL_LOG_PATH}\n")

# --- Tier 1: vs Random ---
print("── Tier 1: HAL vs Random ──────────────────────────────────\n")
r1 = evaluate("1. HAL (White) vs Random (Black)",
              hal_move, random_move, N_GAMES_RANDOM)
r2 = evaluate("2. Random (White) vs HAL (Black)",
              random_move, hal_move, N_GAMES_RANDOM)

hal_vs_random = (r1["white_wins"] + r2["black_wins"]) / (N_GAMES_RANDOM * 2) * 100
print(f"Overall HAL win rate vs random: {hal_vs_random:.1f}%\n")

# --- Tier 2: vs Stockfish ---
print("── Tier 2: HAL vs Stockfish ───────────────────────────────\n")

try:
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 1, "Hash": 16})   # 1 thread, 16MB hash — prevent unified memory bloat

    for depth in [1, 3, 5]:
        n_sims = SIMS_BY_DEPTH.get(depth, 200)
        hal_sf = hal_move_at(n_sims)
        sf_move = stockfish_move(engine, depth)
        print(f"  (HAL using {n_sims} simulations at depth {depth})\n")
        evaluate(f"3. HAL (White) vs Stockfish depth {depth}",
                 hal_sf, sf_move, N_GAMES_STOCKFISH)
        evaluate(f"4. Stockfish depth {depth} (White) vs HAL (Black)",
                 sf_move, hal_sf, N_GAMES_STOCKFISH)

    engine.quit()

except FileNotFoundError:
    print("Stockfish not found — skipping Tier 2.")
    print("Install with: brew install stockfish\n")

# --- Tier 3: vs previous checkpoint (optional) ---

if args.prev:
    print("── Tier 3: HAL vs Previous Checkpoint ────────────────────\n")
    try:
        hal_prev = ChessAgent(device, n_simulations=N_SIMULATIONS)
        hal_prev.load(args.prev)
        print(f"Previous checkpoint steps: {hal_prev.steps:,}\n")

        def hal_prev_move(board, history):
            move_uci, _ = hal_prev.choose_move(board, history, greedy=True)
            return move_uci

        evaluate("5. HAL current (White) vs HAL previous (Black)",
                 hal_move, hal_prev_move, N_GAMES_PREV)
        evaluate("6. HAL previous (White) vs HAL current (Black)",
                 hal_prev_move, hal_move, N_GAMES_PREV)

    except FileNotFoundError:
        print(f"Previous checkpoint not found: {args.prev}\n")

print("=" * 60)
print("Done.")
