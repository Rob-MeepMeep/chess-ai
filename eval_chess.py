"""
eval_chess.py — Evaluate HAL-4000's chess performance.

Three tiers of evaluation:

  1. HAL vs Random (both colours) — sanity check, should win 80%+
  2. HAL vs Stockfish depth 1, 3, 5 — a fixed, repeatable opponent at a fixed
     search depth. NOT a calibrated ELO benchmark -- Stockfish's own strength
     at low fixed depths varies with hardware and version and doesn't map
     cleanly to a rating figure (an earlier depth->ELO guess here was removed
     10 Sept 2026, per an independent assessment: "no defensible Elo estimate
     follows from these results... HAL's rating must come from an actual
     calibrated competition protocol"). Useful as a consistent, comparable
     opponent across checkpoints; not useful as a rating number. A real
     rating requires actual rated games (see Phase 4 / Lichess bot plan).
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
# WSL2 ROCm GPU detection requirement
os.environ["HSA_ENABLE_DXG_DETECTION"] = "1"

import random
import time
import chess
import chess.engine
import torch

from chessai.agent   import ChessAgent
from chessai.encoder import encode
from run_config      import CKPT_PATH, LOG_DIR   # single source for the active run

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_GAMES_RANDOM     = 25    # bump to 100 for final paper benchmark
N_GAMES_STOCKFISH  = 25    # 25 per matchup — enough for meaningful numbers in a proper benchmark
N_GAMES_PREV       = 25    # games vs previous checkpoint
N_SIMS_RANDOM      = 50    # sims vs random — lower is fine, random doesn't punish weak play
N_SIMS_PREV        = 100   # sims for the checkpoint-vs-checkpoint improvement test
# Per-depth sim counts — higher depths warrant more search to have any chance
SIMS_BY_DEPTH      = {1: 200, 3: 500, 5: 500}
MAX_GAME_MOVES     = 200   # hard cap — 200 plies is enough; if HAL can't convert by then it's a policy problem
# Used only with --no-adjudication: real games without the early-stop
# streaks need more headroom to reach an actual conclusion. Still a
# cap, not infinite -- reported as its own distinct "unresolved" bucket
# (10 Sept 2026 assessment: "training and evaluation do not need
# identical stopping rules"). 400 PLIES (200 full moves) -- move_list
# counts one entry per ply, like MAX_GAME_MOVES above; naming this
# "moves" when it counts plies is this project's existing convention
# (train_chess.py's MAX_GAME_MOVES does the same), not something to
# perpetuate in prose describing it.
MAX_GAME_MOVES_NO_ADJUDICATION = 400
STOCKFISH_PATH     = "stockfish"   # assumes stockfish is on PATH

# Mirrors train_chess.py's intervention-ladder rung 1 (run15+): without this,
# a game where HAL builds a winning material lead against Random but can't
# force mate within MAX_GAME_MOVES was scored as a draw ('*') — silently
# undercounting HAL's real win rate. Keep these in sync with
# train_chess.py's MATERIAL_ADJUDICATE_* constants.
MATERIAL_ADJUDICATE_THRESHOLD = 8    # |material| this decisive triggers adjudication
MATERIAL_ADJUDICATE_STREAK    = 6    # consecutive plies it must hold (3 full moves)
MATERIAL_ADJUDICATE_MIN_MOVE  = 60   # earliest move either tier applies from

# Rung 1b (run17+) — second, lower-confidence tier. Kept in sync with
# train_chess.py's MATERIAL_ADJUDICATE_MODERATE_* constants for the same
# reason as above: eval must adjudicate identically to training, or its
# win-rate numbers silently undercount again.
MATERIAL_ADJUDICATE_MODERATE_LOW    = 3
MATERIAL_ADJUDICATE_MODERATE_HIGH   = 8
MATERIAL_ADJUDICATE_MODERATE_STREAK = 16

# ---------------------------------------------------------------------------
# Arguments — parsed early so --cpu affects device selection
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=None,
                    help="Checkpoint to evaluate (default: the active run's, from run_config)")
parser.add_argument("--prev", default=None,
                    help="Path to previous checkpoint for improvement test")
parser.add_argument("--regression-only", action="store_true",
                    help="Run value head regression test only — ~5s, safe during active training")
parser.add_argument("--cpu", action="store_true",
                    help="Force CPU device — keeps MPS free when running alongside a training loop")
parser.add_argument("--no-adjudication", action="store_true",
                    help="Disable material-adjudication early stopping — play to real "
                         f"checkmate/draw or a {MAX_GAME_MOVES_NO_ADJUDICATION}-ply "
                         f"({MAX_GAME_MOVES_NO_ADJUDICATION // 2} full moves) cap instead. "
                         "Slower, but gives a genuine (not adjudicated) conversion rate. "
                         "Intended for the paper benchmark, not routine evals alongside training.")
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

if args.cpu:
    device = torch.device("cpu")
elif torch.cuda.is_available():     # NVIDIA CUDA or AMD ROCm (appears as "cuda" under ROCm)
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if args.cpu:
    print("Device: cpu (forced — MPS reserved for training)")

# ---------------------------------------------------------------------------
# Load HAL
# ---------------------------------------------------------------------------

ckpt_path = args.ckpt or CKPT_PATH
hal = ChessAgent(device, n_simulations=N_SIMS_RANDOM)
try:
    hal.load(ckpt_path)
    print(f"Loaded HAL-4000")
    print(f"  Trained steps: {hal.steps:,}")
    print(f"  Checkpoint:    {ckpt_path}\n")
except FileNotFoundError:
    if args.regression_only:
        # A fresh network is a legitimate regression baseline — should read ~0
        # everywhere. Useful before game 1 of a new run.
        print(f"No checkpoint at {ckpt_path} — regression on fresh random weights "
              f"(expect ~0.0 for every position).\n")
    else:
        print(f"No checkpoint at {ckpt_path}.")
        print(f"Train first, or point at one with --ckpt <path>.")
        exit(1)

# ---------------------------------------------------------------------------
# Value head regression test
# Checks whether the value head has learned to distinguish positions.
# A draw-collapsed network outputs ~0 everywhere.
# A trained network should approach +1 / -1 on decisive endgames.
# ---------------------------------------------------------------------------

# K+Q FENs replaced 2026-07-09: the old w_wins position was ILLEGAL (black in
# check with white to move) and the old b_move position was already checkmate —
# a terminal state the value head never trains on. regression.csv values before
# this date are not comparable. Keep in sync with logger.record_regression().
REGRESSION_POSITIONS = {
    # FEN                                                    description          expect
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": ("start",              "~0.0"),
    "k7/8/8/8/4K3/8/3Q4/8 w - - 0 1":                            ("K+Q vs K (w wins)", "near +1"),
    "k7/8/8/8/4K3/8/3Q4/8 b - - 0 1":                            ("K+Q vs K (b move)", "near -1"),
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
    move_uci, _, _ = hal.choose_move(board, history, greedy=True)
    return move_uci

def hal_move_at(n_sims: int):
    """Return a move function that runs HAL at the given simulation count."""
    def _move(board: chess.Board, history: list) -> str:
        orig = hal.n_simulations
        hal.n_simulations = n_sims
        move_uci, _, _ = hal.choose_move(board, history, greedy=True)
        hal.n_simulations = orig
        return move_uci
    return _move

def hal_move_noisy_at(n_sims: int):
    """HAL at n_sims with Dirichlet noise at root — breaks determinism for diverse Stockfish evals.
    Still uses greedy (argmax) move selection so quality isn't degraded."""
    def _move(board: chess.Board, history: list) -> str:
        orig = hal.n_simulations
        hal.n_simulations = n_sims
        move_uci, _, _ = hal.choose_move(board, history, greedy=True, add_noise=True)
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

_PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                 chess.ROOK: 5, chess.QUEEN: 9}

def _material_balance(board: chess.Board) -> int:
    """Positive = white ahead. Same definition as train_chess.py's helper."""
    score = 0
    for piece, val in _PIECE_VALUES.items():
        score += val * len(board.pieces(piece, chess.WHITE))
        score -= val * len(board.pieces(piece, chess.BLACK))
    return score


def play_game(white_fn, black_fn, no_adjudication: bool = False):
    """
    Play one game. Returns (result, end_reason, move_list).

    result is '1-0', '0-1', '1/2-1/2', or '*' (ply cap hit with no
    resolution — a genuinely undecided position, not just an unforced mate).

    end_reason is one of "checkmate", "rule_draw" (real legal termination —
    stalemate, repetition, 50-move, insufficient material), "material_
    adjudication"/"material_adjudication_moderate" (stopped early on a
    sustained lead, mirroring train_chess.py's own rules), or "move_cap"
    (hit the ply limit with nothing resolved). Distinguishing these matters:
    an "adjudicated win" and an actual checkmate are not the same claim
    (10 Sept 2026 assessment — "keep the existing metric as an explicitly
    named material-adjudication score").

    no_adjudication=True disables the material-adjudication early stop
    entirely and raises the ply cap to MAX_GAME_MOVES_NO_ADJUDICATION —
    training and evaluation don't need identical stopping rules, and eval
    can afford to play real games out instead of using training's
    throughput-driven shortcut.
    """
    board                          = chess.Board()
    history                        = []
    move_list                      = []
    material_streak                = 0
    material_streak_side           = None
    material_streak_moderate       = 0
    material_streak_moderate_side  = None
    max_moves = MAX_GAME_MOVES_NO_ADJUDICATION if no_adjudication else MAX_GAME_MOVES

    while not board.is_game_over() and len(move_list) < max_moves:
        if board.turn == chess.WHITE:
            move_uci = white_fn(board, history)
        else:
            move_uci = black_fn(board, history)

        history = ([board.copy()] + history)[:3]
        board.push_uci(move_uci)
        move_list.append(move_uci)

        if no_adjudication:
            continue   # real games only — never break early on a material streak

        # Mirrors train_chess.py's SelfPlayGame streak logic exactly,
        # including the same-side check -- magnitude alone doesn't mean
        # one side has held the lead the whole streak; it could be two
        # different sides each holding it for a few plies either side of
        # a swing (10 Sept 2026 assessment, same bug duplicated here).
        past_min_move = len(move_list) > MATERIAL_ADJUDICATE_MIN_MOVE
        mat           = _material_balance(board)
        mat_abs       = abs(mat)
        mat_favoured  = chess.WHITE if mat > 0 else chess.BLACK

        if (past_min_move and mat_abs >= MATERIAL_ADJUDICATE_THRESHOLD
                and mat_favoured == material_streak_side):
            material_streak += 1
        elif past_min_move and mat_abs >= MATERIAL_ADJUDICATE_THRESHOLD:
            material_streak = 1
            material_streak_side = mat_favoured
        else:
            material_streak = 0
            material_streak_side = None

        in_moderate_band = (past_min_move
                            and MATERIAL_ADJUDICATE_MODERATE_LOW <= mat_abs < MATERIAL_ADJUDICATE_MODERATE_HIGH)
        if in_moderate_band and mat_favoured == material_streak_moderate_side:
            material_streak_moderate += 1
        elif in_moderate_band:
            material_streak_moderate = 1
            material_streak_moderate_side = mat_favoured
        else:
            material_streak_moderate = 0
            material_streak_moderate_side = None

        if (material_streak >= MATERIAL_ADJUDICATE_STREAK
                or material_streak_moderate >= MATERIAL_ADJUDICATE_MODERATE_STREAK):
            break

    if board.is_game_over():
        result = board.result()
        end_reason = "checkmate" if result in ("1-0", "0-1") else "rule_draw"
    elif material_streak >= MATERIAL_ADJUDICATE_STREAK:
        result     = "1-0" if _material_balance(board) > 0 else "0-1"
        end_reason = "material_adjudication"
    elif material_streak_moderate >= MATERIAL_ADJUDICATE_MODERATE_STREAK:
        result     = "1-0" if _material_balance(board) > 0 else "0-1"
        end_reason = "material_adjudication_moderate"
    else:
        result     = "*"
        end_reason = "move_cap"

    return result, end_reason, move_list

# ---------------------------------------------------------------------------
# Eval game log — one row per game, written to <LOG_DIR>/eval_games.csv
# ---------------------------------------------------------------------------

_EVAL_LOG_PATH = os.path.join(LOG_DIR, "eval_games.csv")
_eval_game_num = 0   # sequential across the whole eval run
_EVAL_LOG_HEADER = [
    "eval_game", "matchup", "result", "end_reason", "n_moves",
    "hal_steps", "timestamp", "moves",
]

def _init_eval_log() -> None:
    os.makedirs(os.path.dirname(_EVAL_LOG_PATH), exist_ok=True)
    if not os.path.exists(_EVAL_LOG_PATH):
        with open(_EVAL_LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(_EVAL_LOG_HEADER)
        return

    # Migrate an older-format file (no end_reason column) in place, rather
    # than either breaking the column count for new rows or silently
    # inventing history for old ones. Added 10 Sept 2026 so per-game end
    # reason (real checkmate vs material adjudication vs an unresolved
    # ply-cap) is queryable directly instead of needing to replay every
    # game's moves after the fact -- exactly what the independent Codex
    # assessment had to do to produce this same breakdown.
    with open(_EVAL_LOG_PATH, newline="") as f:
        rows = list(csv.reader(f))
    if rows and rows[0] == _EVAL_LOG_HEADER:
        return   # already current format

    print(f"  Migrating {_EVAL_LOG_PATH} to add end_reason "
          f"(existing rows backfilled as 'unknown', not replayed)...")
    old_header, old_rows = rows[0], rows[1:]
    result_pos = old_header.index("result")
    migrated = [row[:result_pos + 1] + ["unknown"] + row[result_pos + 1:]
                for row in old_rows]

    # Write to a temp file and replace atomically -- opening the original
    # log directly for writing would truncate it immediately, so an
    # interruption partway through (kill, crash, disk full) could destroy
    # historical eval data instead of just failing the migration. Same
    # pattern chessai/replay.py's save() already uses (10 Sept 2026
    # assessment).
    tmp_path = _EVAL_LOG_PATH + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_EVAL_LOG_HEADER)
        w.writerows(migrated)
    os.replace(tmp_path, _EVAL_LOG_PATH)

def _log_eval_game(matchup: str, result: str, end_reason: str, move_list: list) -> None:
    global _eval_game_num
    _eval_game_num += 1
    with open(_EVAL_LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            _eval_game_num, matchup, result, end_reason, len(move_list),
            hal.steps, time.strftime("%Y-%m-%d %H:%M:%S"),
            " ".join(move_list),
        ])

# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate(label: str, white_fn, black_fn, n: int, no_adjudication: bool = False) -> dict:
    """
    Run n games, print results, return stats dict.

    Breaks wins and draws down by how the game actually ended — a real
    checkmate and a material-adjudicated "win" are not the same claim,
    and an unresolved ply-cap is not the same as a genuine draw (10 Sept
    2026 assessment: "keep the existing metric as an explicitly named
    material-adjudication score"). See play_game()'s end_reason values.
    """
    white_wins = black_wins = draws = 0
    white_checkmates = white_adjudicated = 0
    black_checkmates = black_adjudicated = 0
    real_draws = unresolved = 0

    for i in range(n):
        result, end_reason, move_list = play_game(white_fn, black_fn,
                                                   no_adjudication=no_adjudication)
        _log_eval_game(label, result, end_reason, move_list)

        adjudicated = end_reason in ("material_adjudication", "material_adjudication_moderate")

        if result == "1-0":
            white_wins += 1
            white_checkmates  += end_reason == "checkmate"
            white_adjudicated += adjudicated
        elif result == "0-1":
            black_wins += 1
            black_checkmates  += end_reason == "checkmate"
            black_adjudicated += adjudicated
        else:
            draws += 1
            real_draws += end_reason == "rule_draw"
            unresolved += end_reason == "move_cap"

        # Progress dot every 10 games
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n}...", end="\r")

    print(f"{label}")
    print(f"  White wins: {white_wins:>4} ({white_wins/n*100:5.1f}%)"
          f"  [checkmate: {white_checkmates}, adjudicated: {white_adjudicated}]")
    print(f"  Black wins: {black_wins:>4} ({black_wins/n*100:5.1f}%)"
          f"  [checkmate: {black_checkmates}, adjudicated: {black_adjudicated}]")
    print(f"  Draws:      {draws:>4} ({draws/n*100:5.1f}%)"
          f"  [real: {real_draws}, unresolved ply-cap: {unresolved}]")
    print()

    return {
        "white_wins": white_wins, "black_wins": black_wins, "draws": draws, "n": n,
        "white_checkmates": white_checkmates, "white_adjudicated": white_adjudicated,
        "black_checkmates": black_checkmates, "black_adjudicated": black_adjudicated,
        "real_draws": real_draws, "unresolved": unresolved,
    }

# ---------------------------------------------------------------------------
# Run all matchups
# ---------------------------------------------------------------------------

def run_evaluation_suite() -> None:
    """
    Runs Tiers 1-3 (or just the regression printout if --regression-only).
    Guarded behind __main__ (10 Sept 2026 test-suite work) so `import
    eval_chess` defines everything -- agent, play_game, evaluate,
    hal_move, the CSV helpers -- without kicking off the full (slow,
    Stockfish-dependent) suite. Previously this whole section ran at
    plain-import time regardless of --regression-only, which is why
    testing anything here needed a sys.argv + importlib workaround.
    """
    if args.regression_only:
        print("=" * 60)
        print("Done.")
        return

    _init_eval_log()
    print("=" * 60)
    print(f"Eval games logged to: {_EVAL_LOG_PATH}\n")
    if args.no_adjudication:
        print(f"--no-adjudication: playing to real checkmate/draw or a "
              f"{MAX_GAME_MOVES_NO_ADJUDICATION}-ply "
              f"({MAX_GAME_MOVES_NO_ADJUDICATION // 2} full moves) cap, not the "
              f"material-adjudication shortcut. Slower; expect this to take a while.\n")

    # --- Tier 1: vs Random ---
    print("── Tier 1: HAL vs Random ──────────────────────────────────\n")
    r1 = evaluate("1. HAL (White) vs Random (Black)",
                  hal_move, random_move, N_GAMES_RANDOM, no_adjudication=args.no_adjudication)
    r2 = evaluate("2. Random (White) vs HAL (Black)",
                  random_move, hal_move, N_GAMES_RANDOM, no_adjudication=args.no_adjudication)

    hal_vs_random = (r1["white_wins"] + r2["black_wins"]) / (N_GAMES_RANDOM * 2) * 100
    hal_checkmates_vs_random = r1["white_checkmates"] + r2["black_checkmates"]
    print(f"Overall HAL win rate vs random (material-adjudication inclusive): "
          f"{hal_vs_random:.1f}%")
    print(f"  Of which actual checkmates: {hal_checkmates_vs_random}/{N_GAMES_RANDOM * 2}\n")

    # --- Tier 2: vs Stockfish ---
    print("── Tier 2: HAL vs Stockfish ───────────────────────────────\n")

    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine.configure({"Threads": 1, "Hash": 16})   # 1 thread, 16MB hash — prevent unified memory bloat

        for depth in [1]:   # depth 3/5 re-enable when HAL reaches 20% W/D vs depth 1
            n_sims = SIMS_BY_DEPTH.get(depth, 200)
            hal_sf = hal_move_noisy_at(n_sims)   # noise breaks determinism; greedy selection preserved
            sf_move = stockfish_move(engine, depth)
            print(f"  (HAL using {n_sims} simulations at depth {depth})\n")
            evaluate(f"3. HAL (White) vs Stockfish depth {depth}",
                     hal_sf, sf_move, N_GAMES_STOCKFISH, no_adjudication=args.no_adjudication)
            evaluate(f"4. Stockfish depth {depth} (White) vs HAL (Black)",
                     sf_move, hal_sf, N_GAMES_STOCKFISH, no_adjudication=args.no_adjudication)

        engine.quit()

    except FileNotFoundError:
        print("Stockfish not found — skipping Tier 2.")
        print("Install with: brew install stockfish\n")

    # --- Tier 3: vs previous checkpoint (optional) ---

    if args.prev:
        print("── Tier 3: HAL vs Previous Checkpoint ────────────────────\n")
        try:
            # (was N_SIMULATIONS — a name this file never defined; every --prev
            # run died with a NameError the except below didn't catch)
            hal_prev = ChessAgent(device, n_simulations=N_SIMS_PREV)
            hal_prev.load(args.prev)
            print(f"Previous checkpoint steps: {hal_prev.steps:,}\n")

            def hal_prev_move(board, history):
                move_uci, _, _ = hal_prev.choose_move(board, history, greedy=True)
                return move_uci

            # hal_move (used elsewhere for the cheap vs-random tier) runs at
            # whatever hal.n_simulations currently is -- N_SIMS_RANDOM (50),
            # not N_SIMS_PREV (100). Using it here gave the current checkpoint
            # half the search budget of the previous one in every --prev run
            # (10 Sept 2026 assessment). hal_move_at() explicitly sets the
            # budget per call, same pattern Tier 2 already uses correctly.
            hal_move_prev_tier = hal_move_at(N_SIMS_PREV)

            evaluate("5. HAL current (White) vs HAL previous (Black)",
                     hal_move_prev_tier, hal_prev_move, N_GAMES_PREV,
                     no_adjudication=args.no_adjudication)
            evaluate("6. HAL previous (White) vs HAL current (Black)",
                     hal_prev_move, hal_move_prev_tier, N_GAMES_PREV,
                     no_adjudication=args.no_adjudication)

        except FileNotFoundError:
            print(f"Previous checkpoint not found: {args.prev}\n")

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    run_evaluation_suite()
