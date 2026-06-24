"""
curate_buffer.py — Build a high-quality seed buffer for the next run.

Reads the previous run's game log, filters out low-quality games, replays the
surviving games move-by-move to extract positions, adds canonical
endgame positions with correct outcomes, and saves the result as a
seed buffer file that the next run can load on startup.

Why: the bootstrapping problem. Early self-play generates noise — both
sides random, outcomes meaningless. Starting with this curated buffer
means the value head gets real signal from gradient step 1
instead of from game ~500.

Quality filters applied:
  - Only games from game MIN_GAME onwards (network was coherent by then)
  - Only decisive games (material_resign or checkmate) — cap draws are noise
  - Minimum move count to exclude overconfident wrong value calls
  - Maximum move count to exclude very long potentially random shuffles

Canonical endgame positions are added with ground-truth outcomes so
the value head has direct signal for positions it rarely sees in self-play.

Run 11 adds 193 mid-game material-imbalanced positions (rook or more advantage,
plies 8-28) extracted from Run 10 self-play and reviewed by an external agent.
These anchor the value head to material imbalance reasoning in real mid-game
positions, not just K+Q vs K endgames.

Usage:
  venv/bin/python3 curate_buffer.py

Output:
  checkpoints/run11_seed_buffer.pt

Then in train_chess.py for Run 11, set:
  BUFFER_LOAD = "checkpoints/run11_seed_buffer.pt"
"""

import csv
import json
import random
import chess
import torch

from chessai.encoder  import encode
from chessai.replay   import ReplayBuffer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GAMES_CSV      = "logs/run11/games.csv"
OUTPUT_PATH    = "checkpoints/run12_seed_buffer.pt"

# Mid-game material positions reviewed by external agent (Run 11 addition)
REVIEWED_JSON  = "paper/buffer_candidates_reviewed.json"
CANDIDATES_JSON = "paper/buffer_candidates.json"

# Game quality filters
MIN_GAME       = 2000    # post-policy-mirroring-bug-fix (fixed at game ~1700)
MIN_MOVES      = 20      # skip overconfident short games
MAX_MOVES      = 100     # skip very long games that may be random shuffling
GOOD_REASONS   = {"material_resign", "checkmate", "value_resign"}  # decisive outcomes

# Canonical positions: (FEN, outcome from current player's perspective)
# Repeated CANONICAL_REPEATS times — reduced from 200 now that diverse K+Q vs K
# positions supplement these. Diversity provides the signal; repetition is less important.
CANONICAL_REPEATS = 5
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

N_DIVERSE_PAIRS    = 128  # K+Q vs K: generates 256 positions total (128 W-to-move + 128 B-to-move)
N_KR_VS_K_PAIRS    = 64   # K+R vs K: 128 positions — rook endgame conversion
N_KQ_VS_KP_PAIRS   = 64   # K+Q vs K+P: 128 positions — queen vs passer conversion

BUFFER_CAPACITY = 50_000


def load_reviewed_midgame_positions(reviewed_path, candidates_path):
    """
    Load mid-game material-imbalanced positions accepted by external agent review.
    Returns list of (chess.Board, outcome_float) tuples where outcome is from
    the current player's perspective (consistent with canonical encoding).
    """
    with open(reviewed_path) as f:
        reviewed = json.load(f)
    with open(candidates_path) as f:
        candidates = json.load(f)

    accepted_ids = {r["id"] for r in reviewed if r["accept"]}
    candidates_by_id = {c["id"]: c for c in candidates}

    positions = []
    skipped = 0
    for cid in sorted(accepted_ids):
        c = candidates_by_id.get(cid)
        if c is None:
            skipped += 1
            continue
        try:
            board = chess.Board(c["fen"])
        except Exception:
            skipped += 1
            continue

        # outcome in candidates is from White's perspective (+1.0 White won, -1.0 Black won)
        outcome_white = float(c["outcome"])
        # convert to current player's perspective
        if board.turn == chess.WHITE:
            outcome_mover = outcome_white
        else:
            outcome_mover = -outcome_white

        positions.append((board, outcome_mover))

    if skipped:
        print(f"  Skipped {skipped} positions (missing or invalid FEN)")
    return positions


def generate_diverse_kq_vs_k(num_pairs=128):
    """
    Generates a spatially diverse set of valid K+Q vs K positions.
    Returns a list of (chess.Board, outcome_float) tuples.
    num_pairs White-to-move positions (+1.0) and num_pairs Black-to-move positions (-1.0).
    Outcomes are from the current player's perspective — consistent with canonical encoding.
    """
    positions = []
    seen_fens = set()

    while len(positions) < num_pairs * 2:
        squares = random.sample(list(chess.SQUARES), 3)
        wk_sq, wq_sq, bk_sq = squares[0], squares[1], squares[2]

        board = chess.Board(None)
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(wq_sq, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))

        # White to move — current player (White) has the queen and is winning
        board_w = board.copy()
        board_w.turn = chess.WHITE
        if (board_w.is_valid()
                and not board_w.is_checkmate()
                and not board_w.is_stalemate()):
            fen_w = board_w.fen()
            if fen_w not in seen_fens:
                positions.append((board_w, 1.0))
                seen_fens.add(fen_w)

        # Black to move — current player (Black) has no queen and is losing
        board_b = board.copy()
        board_b.turn = chess.BLACK
        if (board_b.is_valid()
                and not board_b.is_checkmate()
                and not board_b.is_stalemate()):
            fen_b = board_b.fen()
            if fen_b not in seen_fens:
                positions.append((board_b, -1.0))
                seen_fens.add(fen_b)

    return positions[:num_pairs * 2]


def generate_diverse_kr_vs_k(num_pairs=64):
    """
    Generates spatially diverse K+R vs K positions.
    Rook side always wins — outcomes ±1.0 from the current player's perspective.
    Addresses the cap draw conversion gap: HAL needs to see rook endgame technique,
    not just queen endgames.
    """
    positions = []
    seen_fens = set()

    while len(positions) < num_pairs * 2:
        squares = random.sample(list(chess.SQUARES), 3)
        wk_sq, wr_sq, bk_sq = squares[0], squares[1], squares[2]

        board = chess.Board(None)
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(wr_sq, chess.Piece(chess.ROOK, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))

        board_w = board.copy()
        board_w.turn = chess.WHITE
        if (board_w.is_valid()
                and not board_w.is_checkmate()
                and not board_w.is_stalemate()):
            fen_w = board_w.fen()
            if fen_w not in seen_fens:
                positions.append((board_w, 1.0))
                seen_fens.add(fen_w)

        board_b = board.copy()
        board_b.turn = chess.BLACK
        if (board_b.is_valid()
                and not board_b.is_checkmate()
                and not board_b.is_stalemate()):
            fen_b = board_b.fen()
            if fen_b not in seen_fens:
                positions.append((board_b, -1.0))
                seen_fens.add(fen_b)

    return positions[:num_pairs * 2]


def generate_diverse_kq_vs_kp(num_pairs=64):
    """
    Generates spatially diverse K+Q vs K+P positions (White has queen, Black has pawn).
    Queen side nearly always wins — outcomes ±0.9 (soft, not ±1.0, because a
    7th-rank pawn can force a draw in some configurations).
    Teaches HAL to convert a queen advantage against a passed pawn, directly
    targeting the cap draw conversion problem.
    """
    positions = []
    seen_fens = set()

    while len(positions) < num_pairs * 2:
        wk_sq = random.choice(list(chess.SQUARES))
        wq_sq = random.choice(list(chess.SQUARES))
        bk_sq = random.choice(list(chess.SQUARES))
        # Black pawn: ranks 2–7 (chess notation) = 0-indexed ranks 1–6
        bp_sq = chess.square(random.randint(0, 7), random.randint(1, 6))

        if len({wk_sq, wq_sq, bk_sq, bp_sq}) < 4:
            continue

        board = chess.Board(None)
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(wq_sq, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(bp_sq, chess.Piece(chess.PAWN, chess.BLACK))

        board_w = board.copy()
        board_w.turn = chess.WHITE
        if (board_w.is_valid()
                and not board_w.is_checkmate()
                and not board_w.is_stalemate()):
            fen_w = board_w.fen()
            if fen_w not in seen_fens:
                positions.append((board_w, 0.9))
                seen_fens.add(fen_w)

        board_b = board.copy()
        board_b.turn = chess.BLACK
        if (board_b.is_valid()
                and not board_b.is_checkmate()
                and not board_b.is_stalemate()):
            fen_b = board_b.fen()
            if fen_b not in seen_fens:
                positions.append((board_b, -0.9))
                seen_fens.add(fen_b)

    return positions[:num_pairs * 2]


# ---------------------------------------------------------------------------
# Load and filter games
# ---------------------------------------------------------------------------

print(f"Reading {GAMES_CSV}...")
selected = []

with open(GAMES_CSV, newline="") as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        game_num   = int(parts[0])
        outcome    = parts[1]
        end_reason = parts[2]
        n_moves    = int(parts[3])
        # Run 8 games.csv has mixed formats: steps/timestamp columns were added
        # mid-run (~game 1000). Old rows: 6 cols (moves at index 5).
        # New rows: 8 cols (steps at 5, timestamp at 6, moves at 7+).
        if len(parts) >= 8 and parts[5].isdigit():
            moves = " ".join(parts[7:]).split()
        else:
            moves = " ".join(parts[5:]).split()

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

# Static canonical positions — K+R vs K, opening reference, material imbalance
static_count = 0
for fen, outcome in CANONICAL_POSITIONS:
    board = chess.Board(fen)
    state = encode([board])
    policy = torch.zeros(4096)
    for _ in range(CANONICAL_REPEATS):
        canonical_batch.append((state, policy, float(outcome)))
        static_count += 1

# Diverse K+Q vs K positions — 256 unique spatial configurations
# Each position encoded once; diversity replaces repetition as the learning signal.
diverse_positions = generate_diverse_kq_vs_k(N_DIVERSE_PAIRS)
diverse_count = 0
for board, outcome in diverse_positions:
    state  = encode([board])
    policy = torch.zeros(4096)
    canonical_batch.append((state, policy, float(outcome)))
    diverse_count += 1

# Diverse K+R vs K positions — 128 rook endgame configurations (Run 12 addition).
# Targets the cap draw conversion gap: HAL wins material but can't convert within
# 200 moves. Rook endgame technique requires active box-and-push play.
kr_positions = generate_diverse_kr_vs_k(N_KR_VS_K_PAIRS)
kr_count = 0
for board, outcome in kr_positions:
    state  = encode([board])
    policy = torch.zeros(4096)
    canonical_batch.append((state, policy, float(outcome)))
    kr_count += 1

# Diverse K+Q vs K+P positions — 128 queen-vs-passer configurations (Run 12 addition).
# Teaches HAL to convert a queen advantage against a Black passed pawn — the scenario
# where the 80% cap draw rate most often applies. Outcomes ±0.9 (soft win, not ±1.0,
# because a 7th-rank pawn can force a draw in some configurations).
kq_kp_positions = generate_diverse_kq_vs_kp(N_KQ_VS_KP_PAIRS)
kq_kp_count = 0
for board, outcome in kq_kp_positions:
    state  = encode([board])
    policy = torch.zeros(4096)
    canonical_batch.append((state, policy, float(outcome)))
    kq_kp_count += 1

# Mid-game material-imbalanced positions — carried forward from Run 11.
# 193 positions from Run 10 self-play (game 2000+), filtered to plies 8-28,
# abs(material) >= 5, material-advantaged side won. Reviewed by external agent.
midgame_positions = load_reviewed_midgame_positions(REVIEWED_JSON, CANDIDATES_JSON)
midgame_count = 0
for board, outcome in midgame_positions:
    state  = encode([board])
    policy = torch.zeros(4096)
    canonical_batch.append((state, policy, float(outcome)))
    midgame_count += 1

buf.add_permanent(canonical_batch)
print(f"  {static_count:,} static canonical positions ({len(CANONICAL_POSITIONS)} × {CANONICAL_REPEATS})")
print(f"  {diverse_count:,} diverse K+Q vs K positions ({N_DIVERSE_PAIRS} pairs, 1 each)")
print(f"  {kr_count:,} diverse K+R vs K positions ({N_KR_VS_K_PAIRS} pairs, 1 each) [Run 12]")
print(f"  {kq_kp_count:,} diverse K+Q vs K+P positions ({N_KQ_VS_KP_PAIRS} pairs, 1 each) [Run 12]")
print(f"  {midgame_count:,} mid-game material-imbalanced positions (Run 10, agent-reviewed)")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

print(f"\nFinal buffer: {len(buf):,} rolling / {len(buf._permanent):,} permanent ({BUFFER_CAPACITY:,} rolling capacity)")
buf.save(OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
print(f"\nFor Run 11, set in train_chess.py:")
print(f"  BUFFER_LOAD = \"{OUTPUT_PATH}\"")
