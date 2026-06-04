"""
curate_buffer.py — Build a high-quality seed buffer for Run 8.

Reads the run7 game log, filters out low-quality games, replays the
surviving games move-by-move to extract positions, adds canonical
endgame positions with correct outcomes, and saves the result as a
seed buffer file that Run 8 can load on startup.

Why: the bootstrapping problem. Early self-play generates noise — both
sides random, outcomes meaningless. Starting Run 8 with this curated
buffer means the value head gets real signal from gradient step 1
instead of from game ~500.

Quality filters applied:
  - Only games from game MIN_GAME onwards (network was coherent by then)
  - Only decisive games (material_resign or checkmate) — cap draws are noise
  - Minimum move count to exclude overconfident wrong value calls
  - Maximum move count to exclude very long potentially random shuffles

Canonical endgame positions are added with ground-truth outcomes so
the value head has direct signal for positions it rarely sees in self-play.

Usage:
  venv/bin/python3 curate_buffer.py

Output:
  checkpoints/run8_seed_buffer.pt

Then in train_chess.py for Run 8, set:
  BUFFER_LOAD = "checkpoints/run8_seed_buffer.pt"
"""

import csv
import chess
import torch

from chessai.encoder  import encode
from chessai.replay   import ReplayBuffer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GAMES_CSV      = "logs/run7/games.csv"
OUTPUT_PATH    = "checkpoints/run8_seed_buffer.pt"

# Game quality filters
MIN_GAME       = 800     # skip early training — value head developing from ~800
MIN_MOVES      = 20      # skip overconfident short games
MAX_MOVES      = 100     # skip very long games that may be random shuffling
GOOD_REASONS   = {"material_resign", "checkmate"}  # decisive, trustworthy outcomes

# Canonical positions: (FEN, outcome from current player's perspective)
# Repeated CANONICAL_REPEATS times each so they have weight in the buffer
CANONICAL_REPEATS = 200
CANONICAL_POSITIONS = [
    # K+Q vs K — trivially decisive endgames
    ("8/8/8/3K4/8/8/8/3kQ3 w - - 0 1",  1.0),   # white to move, white wins
    ("8/8/8/3K4/8/8/8/3kQ3 b - - 0 1", -1.0),   # black to move, black loses
    ("8/8/8/3k4/8/8/8/3Kq3 b - - 0 1",  1.0),   # black to move, black wins
    ("8/8/8/3k4/8/8/8/3Kq3 w - - 0 1", -1.0),   # white to move, white loses
    # K+R vs K
    ("8/8/8/3K4/8/8/8/3kR3 w - - 0 1",  1.0),
    ("8/8/8/3K4/8/8/8/3kR3 b - - 0 1", -1.0),
    # White clearly ahead — rook up
    ("r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  0.5),  # balanced
    ("r1bqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  0.7),  # black missing knight
]

BUFFER_CAPACITY = 50_000

# ---------------------------------------------------------------------------
# Load and filter games
# ---------------------------------------------------------------------------

print(f"Reading {GAMES_CSV}...")
selected = []

with open(GAMES_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        game_num   = int(row["game"])
        n_moves    = int(row["n_moves"])
        end_reason = row["end_reason"]
        outcome    = row["outcome"]
        moves      = row["moves"].split()

        if game_num < MIN_GAME:
            continue
        if n_moves < MIN_MOVES:
            continue
        if n_moves > MAX_MOVES:
            continue
        if end_reason not in GOOD_REASONS:
            continue
        if outcome not in ("W", "B"):
            continue

        selected.append({
            "game":    game_num,
            "outcome": outcome,
            "moves":   moves,
        })

print(f"  {len(selected)} games passed filters (from MIN_GAME={MIN_GAME})")

# ---------------------------------------------------------------------------
# Replay games and extract positions
# ---------------------------------------------------------------------------

buf = ReplayBuffer(capacity=BUFFER_CAPACITY)
positions_from_games = 0

for game_data in selected:
    winner = chess.WHITE if game_data["outcome"] == "W" else chess.BLACK

    board   = chess.Board()
    history = []
    game_positions = []

    for move_uci in game_data["moves"]:
        state = encode([board] + history)
        turn  = board.turn
        # Zero policy — we don't have MCTS visit counts from the game records.
        # The policy head will learn from self-play; value signal is what matters here.
        policy = torch.zeros(4096)

        game_positions.append((state, policy, turn))

        history = ([board.copy()] + history)[:3]
        try:
            board.push_uci(move_uci)
        except Exception:
            break

    # Commit positions with perspective-relative outcomes
    completed = []
    for state, policy, turn in game_positions:
        outcome = 1.0 if turn == winner else -1.0
        completed.append((state, policy, outcome))

    buf.extend(completed)
    positions_from_games += len(completed)

print(f"  {positions_from_games:,} positions extracted from game replays")

# ---------------------------------------------------------------------------
# Add canonical endgame positions
# ---------------------------------------------------------------------------

canonical_batch = []
canonical_count = 0
for fen, outcome in CANONICAL_POSITIONS:
    board = chess.Board(fen)
    state = encode([board])
    policy = torch.zeros(4096)

    for _ in range(CANONICAL_REPEATS):
        canonical_batch.append((state, policy, float(outcome)))
        canonical_count += 1

buf.add_permanent(canonical_batch)
print(f"  {canonical_count:,} canonical positions added to permanent partition ({len(CANONICAL_POSITIONS)} × {CANONICAL_REPEATS})")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

print(f"\nFinal buffer: {len(buf):,} rolling / {len(buf._permanent):,} permanent ({BUFFER_CAPACITY:,} rolling capacity)")
buf.save(OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
print(f"\nFor next run, set in train_chess.py:")
print(f"  BUFFER_LOAD = \"{OUTPUT_PATH}\"")
