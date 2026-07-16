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
  - Minimum move count to exclude overconfident short games
  - Maximum move count to exclude very long potentially random shuffles

Canonical endgame positions are added with ground-truth outcomes so
the value head has direct signal for positions it rarely sees in self-play.

LABEL SAFETY (2026-07 code review fix): every canonical and generated
position now passes _label_is_safe() before entering the buffer. The
previous static FENs placed the queen/rook adjacent to the enemy king,
undefended — so the "losing" side could simply capture it (a draw, stored
as -1.0), and the winner-to-move variants were outright illegal positions
(the side not to move was in check). The generators had the same hole:
~10% of "losing side to move" samples could take the hanging piece.
Wrong labels in the PERMANENT partition are the most expensive kind —
it feeds a third of every training batch, forever.

Usage:
  venv/bin/python3 curate_buffer.py

Output:
  checkpoints/run14_seed_buffer.pt
"""

import csv
import json
import random
import chess
import os
# WSL2 ROCm GPU detection requirement
os.environ["HSA_ENABLE_DXG_DETECTION"] = "1"

import torch

from chessai.encoder  import encode
from chessai.replay   import ReplayBuffer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GAMES_CSV      = "logs/run13_retune/games.csv"
OUTPUT_PATH    = "checkpoints/run14_seed_buffer.pt"

# Mid-game material positions reviewed by external agent (Run 11 addition)
REVIEWED_JSON  = "paper/buffer_candidates_reviewed.json"
CANDIDATES_JSON = "paper/buffer_candidates.json"

# Game quality filters
MIN_GAME       = 200     # run13_retune started from a pre-trained checkpoint so
                         # the network was coherent from game 1; skip only the
                         # very first warmup games (run12's threshold of 2000 was
                         # a bug-fix boundary that doesn't apply here)
MIN_MOVES      = 20      # skip overconfident short games
MAX_MOVES      = 100     # skip very long games that may be random shuffling
GOOD_REASONS   = {"material_resign", "checkmate", "value_resign"}  # decisive outcomes

# Canonical positions: (FEN, outcome from current player's perspective).
# The winning piece is kept far from the enemy king so no label can be
# falsified by an immediate capture — enforced again by _label_is_safe().
CANONICAL_REPEATS = 5
CANONICAL_POSITIONS = [
    # K+Q vs K — trivially decisive endgames (Qd2 nowhere near the a8 king)
    ("k7/8/8/8/4K3/8/3Q4/8 w - - 0 1",  1.0),   # white to move, white wins
    ("k7/8/8/8/4K3/8/3Q4/8 b - - 0 1", -1.0),   # black to move, black loses
    ("8/3q4/8/4k3/8/8/8/K7 b - - 0 1",  1.0),   # black to move, black wins
    ("8/3q4/8/4k3/8/8/8/K7 w - - 0 1", -1.0),   # white to move, white loses
    # K+R vs K
    ("k7/8/8/8/4K3/8/3R4/8 w - - 0 1",  1.0),
    ("k7/8/8/8/4K3/8/3R4/8 b - - 0 1", -1.0),
    # Opening references: balanced start, knight odds, double knight odds
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  0.0),
    ("r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  0.5),  # black missing one knight
    ("r1bqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  0.7),  # black missing both knights
]

N_DIVERSE_PAIRS    = 128  # K+Q vs K: generates 256 positions total (128 W-to-move + 128 B-to-move)
N_KR_VS_K_PAIRS    = 64   # K+R vs K: 128 positions — rook endgame conversion
N_KQ_VS_KP_PAIRS   = 64   # K+Q vs K+P: 128 positions — queen vs passer conversion

BUFFER_CAPACITY = 200_000


def _label_is_safe(board: chess.Board) -> bool:
    """
    Reject positions whose ground-truth label the side to move can
    immediately falsify. Applied to every canonical and generated position.

      - illegal or already-finished positions carry no usable label
      - capturing the winning side's queen/rook turns a "loss" into a draw
      - a promotion rewrites the material story the label was based on
    """
    if not board.is_valid() or board.is_game_over():
        return False
    for move in board.legal_moves:
        if move.promotion:
            return False
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured and captured.piece_type in (chess.QUEEN, chess.ROOK):
                return False
    return True


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


def _generate_piece_endgames(num_pairs: int, white_piece: int,
                             outcome_magnitude: float = 1.0,
                             black_pawn: bool = False):
    """
    Shared generator for spatially diverse won endgames: White king + one
    winning piece vs Black king (optionally + a Black pawn). Returns
    (chess.Board, outcome) tuples from the current player's perspective:
    +magnitude when the winning side is to move, -magnitude otherwise.
    Every candidate must pass _label_is_safe() for BOTH sides.
    """
    positions = []
    seen_fens = set()

    while len(positions) < num_pairs * 2:
        squares = random.sample(list(chess.SQUARES), 3)
        wk_sq, wp_sq, bk_sq = squares

        board = chess.Board(None)
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(wp_sq, chess.Piece(white_piece, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))

        if black_pawn:
            # Ranks 3–7 only: a pawn on rank 2 is one move from promoting,
            # which flips the outcome the label claims
            bp_sq = chess.square(random.randint(0, 7), random.randint(2, 6))
            if bp_sq in (wk_sq, wp_sq, bk_sq):
                continue
            board.set_piece_at(bp_sq, chess.Piece(chess.PAWN, chess.BLACK))

        for turn, outcome in ((chess.WHITE,  outcome_magnitude),
                              (chess.BLACK, -outcome_magnitude)):
            b = board.copy()
            b.turn = turn
            if not _label_is_safe(b):
                continue
            fen = b.fen()
            if fen not in seen_fens:
                positions.append((b, outcome))
                seen_fens.add(fen)

    return positions[:num_pairs * 2]


def generate_diverse_kq_vs_k(num_pairs=128):
    """K+Q vs K — queen side always wins. ±1.0."""
    return _generate_piece_endgames(num_pairs, chess.QUEEN, 1.0)


def generate_diverse_kr_vs_k(num_pairs=64):
    """K+R vs K — rook side always wins. ±1.0.
    Addresses the cap draw conversion gap: HAL needs rook endgame technique,
    not just queen endgames."""
    return _generate_piece_endgames(num_pairs, chess.ROOK, 1.0)


def generate_diverse_kq_vs_kp(num_pairs=64):
    """K+Q vs K+P — queen side nearly always wins. ±0.9 (soft, not ±1.0:
    some pawn configurations hold a draw). Teaches conversion against a
    passed pawn — the scenario where cap draws most often applied."""
    return _generate_piece_endgames(num_pairs, chess.QUEEN, 0.9, black_pawn=True)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main():
    # --- Load and filter games ---
    print(f"Reading {GAMES_CSV}...")
    selected = []

    try:
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
    except FileNotFoundError:
        print(f"  Warning: {GAMES_CSV} not found. Skipping self-play game extraction. "
              f"Seed buffer will only contain canonical and reviewed positions.")

    # --- Replay games and extract positions ---
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

    # --- Add canonical endgame positions ---
    canonical_batch = []

    # Static canonical positions — verified at build time; a bad FEN should
    # stop the pipeline, not silently poison the permanent partition
    static_count = 0
    for fen, outcome in CANONICAL_POSITIONS:
        board = chess.Board(fen)
        if abs(outcome) > 0.001 and not _label_is_safe(board):
            raise ValueError(f"Canonical position failed label safety: {fen}")
        state = encode([board])
        policy = torch.zeros(4096)
        for _ in range(CANONICAL_REPEATS):
            canonical_batch.append((state, policy, float(outcome)))
            static_count += 1

    generated = [
        ("diverse K+Q vs K",   generate_diverse_kq_vs_k(N_DIVERSE_PAIRS)),
        ("diverse K+R vs K",   generate_diverse_kr_vs_k(N_KR_VS_K_PAIRS)),
        ("diverse K+Q vs K+P", generate_diverse_kq_vs_kp(N_KQ_VS_KP_PAIRS)),
    ]
    for label, positions in generated:
        for board, outcome in positions:
            state  = encode([board])
            policy = torch.zeros(4096)
            canonical_batch.append((state, policy, float(outcome)))
        print(f"  {len(positions):,} {label} positions")

    # Mid-game material-imbalanced positions — carried forward from Run 11.
    # 193 positions from Run 10 self-play (game 2000+), filtered to plies 8-28,
    # abs(material) >= 5, material-advantaged side won. Reviewed by external agent.
    midgame_positions = load_reviewed_midgame_positions(REVIEWED_JSON, CANDIDATES_JSON)
    for board, outcome in midgame_positions:
        state  = encode([board])
        policy = torch.zeros(4096)
        canonical_batch.append((state, policy, float(outcome)))

    buf.add_permanent(canonical_batch)
    print(f"  {static_count:,} static canonical positions "
          f"({len(CANONICAL_POSITIONS)} × {CANONICAL_REPEATS})")
    print(f"  {len(midgame_positions):,} mid-game material-imbalanced positions "
          f"(Run 10, agent-reviewed)")

    # --- Save ---
    print(f"\nFinal buffer: {len(buf):,} rolling / {len(buf._permanent):,} permanent "
          f"({BUFFER_CAPACITY:,} rolling capacity)")
    buf.save(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")
    print(f"\nFor the next run, set in train_chess.py:")
    print(f"  BUFFER_LOAD = \"{OUTPUT_PATH}\"")


if __name__ == "__main__":
    main()
