"""
train_chess.py — AlphaZero-style self-play training loop for HAL-4000.

Each game: use MCTS to select every move, store (position, policy, turn) in
a GameBuffer. At game end fill in outcomes and commit to the ReplayBuffer.
Once the buffer is large enough, run a batch of training steps.

Speed guide (MacBook Air M3, MPS, untrained network):
  N_SIMULATIONS = 200  — ~64s/game, ~178 hours for 10k games  (desktop training)
  N_SIMULATIONS = 100  — ~32s/game, ~89 hours for 10k games
  N_SIMULATIONS =  50  — ~16s/game, ~45 hours for 10k games   (recommended for M3)
  N_SIMULATIONS =  25  — ~8s/game,  ~22 hours for 10k games   (quick test run)

To run overnight (prevents sleep, sleeps Mac on completion):
  caffeinate -dims python3 train_chess.py; osascript -e 'tell application "System Events" to sleep'
"""

import os
import csv
import time
import chess
import torch

from chessai.agent   import ChessAgent
from chessai.encoder import encode
from chessai.logger  import Logger
from chessai.replay  import ReplayBuffer, GameBuffer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_GAMES          = 10_000
N_SIMULATIONS    = 100      # reduce for faster iteration; increase on desktop GPU
BATCH_SIZE       = 64
TRAIN_STEPS      = 5        # gradient updates per game (once buffer is ready)
MIN_BUFFER       = 500      # don't train until buffer holds this many positions
MAX_GAME_MOVES   = 150      # hard cap — bumped from 100; resign logic should terminate most games first
CHECKPOINT_EVERY = 10       # save checkpoint every N games
SNAPSHOT_EVERY   = 50      # log MCTS strategy snapshots every N games
PRINT_EVERY      = 10       # print progress line every N games
RESIGN_THRESHOLD   = -0.95  # value head score below which a position is hopeless
                             # -0.95 is conservative — loosen to -0.85 if decisive games remain rare after 500 games
RESIGN_CONSECUTIVE = 3      # consecutive moves below threshold before resigning
RESIGN_MATERIAL    = 5      # resign if down by more than a rook in material (bootstraps early training)

# Run identity — change RUN_NAME to start a new named run with its own logs and buffer.
# CKPT_LOAD: None = load RUN_NAME's own checkpoint; set to a path to seed weights from another run.
# BUFFER_LOAD: None = load RUN_NAME's own buffer; set to a path to load from another run.
RUN_NAME    = "run4"
CKPT_LOAD   = None   # None = load run4's own checkpoint (run4_hal_chess.pt)
BUFFER_LOAD = None                          # no buffer — start clean so draw-poisoned data is discarded

CKPT_PATH   = f"checkpoints/{RUN_NAME}_hal_chess.pt"
BUFFER_PATH = f"checkpoints/{RUN_NAME}_replay_buffer.pt"
LOG_DIR     = f"logs/{RUN_NAME}"

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Initialise or resume
# ---------------------------------------------------------------------------

os.makedirs("checkpoints", exist_ok=True)

agent  = ChessAgent(device, n_simulations=N_SIMULATIONS)
replay = ReplayBuffer()
logger = Logger(log_dir=LOG_DIR, snapshot_interval=SNAPSHOT_EVERY)

start_game = 0
_ckpt_to_load = CKPT_LOAD or CKPT_PATH
if os.path.exists(_ckpt_to_load):
    agent.load(_ckpt_to_load)
    # Only resume game number if continuing the same run's own checkpoint
    if _ckpt_to_load == CKPT_PATH:
        openings_path = os.path.join(LOG_DIR, "openings.csv")
        if os.path.exists(openings_path):
            with open(openings_path) as f:
                rows = list(csv.reader(f))
            if len(rows) > 1:
                start_game = int(rows[-1][0])
    print(f"Loaded weights from {_ckpt_to_load} — starting at game {start_game + 1}")
    print(f"  Trained steps so far: {agent.steps:,}")
    _buf_to_load = BUFFER_LOAD or BUFFER_PATH
    if os.path.exists(_buf_to_load):
        replay.load(_buf_to_load)
        print(f"  Replay buffer loaded: {len(replay):,} positions ({_buf_to_load})")
    else:
        print("  Replay buffer: empty (starting clean)")
else:
    print("Starting fresh training run.")

print(f"N_SIMULATIONS = {N_SIMULATIONS} | N_GAMES = {N_GAMES:,}\n")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

_PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                 chess.ROOK: 5, chess.QUEEN: 9}

def _material_balance(board: chess.Board) -> int:
    """Positive = white ahead. Used for material-based resignation."""
    score = 0
    for piece, val in _PIECE_VALUES.items():
        score += val * len(board.pieces(piece, chess.WHITE))
        score -= val * len(board.pieces(piece, chess.BLACK))
    return score


_game_times: list = []   # rolling window for seconds-per-game estimate
_t_run_start = time.time()

for game_num in range(start_game + 1, N_GAMES + 1):
    _t_game_start = time.time()

    board   = chess.Board()
    history = []
    game_buf = GameBuffer()
    moves   = []
    loss    = 0.0

    # --- Self-play: one complete game ---
    resign_streak = 0     # consecutive moves where the side to move is hopeless
    resign_cause  = None  # "material" or "value" — which condition fired

    while not board.is_game_over() and len(moves) < MAX_GAME_MOVES:

        # Encode current position BEFORE making the move
        state = encode([board] + history)

        # MCTS selects a move — stochastic during training
        move_uci, policy = agent.choose_move(
            board, history, greedy=False
        )

        # Store this position in the game buffer
        game_buf.push(state, policy, board.turn)
        moves.append(move_uci)

        # Keep only the last 3 boards — encoder uses current + 3 history = 4 total
        history = ([board.copy()] + history)[:3]
        board.push_uci(move_uci)

        # Check whether either side is in a hopeless position.
        # Material check fires independently of the network — bootstraps early training.
        # Value check kicks in once the value head has learned to distinguish positions.
        # abs() used for both: we want to resign regardless of which side is losing.
        v            = agent.get_value(board, history)
        mat          = _material_balance(board)
        mat_hopeless = abs(mat) > RESIGN_MATERIAL
        val_hopeless = abs(v) > abs(RESIGN_THRESHOLD)
        if mat_hopeless or val_hopeless:
            resign_streak += 1
            # Record which condition fired first (material takes priority if both true)
            if resign_cause is None:
                resign_cause = "material" if mat_hopeless else "value"
        else:
            resign_streak = 0
            resign_cause  = None
        if resign_streak >= RESIGN_CONSECUTIVE:
            break   # losing side resigns

    # --- Determine winner and end reason ---
    result = board.result()
    if resign_streak >= RESIGN_CONSECUTIVE:
        # Side with more material wins; imbalance was already confirmed by resign_streak
        winner     = chess.WHITE if _material_balance(board) > 0 else chess.BLACK
        end_reason = f"{resign_cause}_resign"
    elif result == "1-0":
        winner     = chess.WHITE
        end_reason = "checkmate"
    elif result == "0-1":
        winner     = chess.BLACK
        end_reason = "checkmate"
    elif board.is_stalemate() or board.is_insufficient_material():
        winner     = None
        end_reason = "rule_draw"
    elif board.is_fifty_moves() or board.is_repetition():
        winner     = None
        end_reason = "rule_draw"
    elif len(moves) >= MAX_GAME_MOVES:
        winner     = None
        end_reason = "cap_draw"
    else:
        winner     = None
        end_reason = "rule_draw"

    # Commit game positions with outcomes to replay buffer
    game_buf.commit(replay, winner)

    # --- Training steps ---
    if replay.ready(MIN_BUFFER):
        for _ in range(TRAIN_STEPS):
            batch = replay.sample(BATCH_SIZE)
            loss  = agent.train(batch)

    # --- Logging ---
    logger.record_game(game_num, winner, moves, loss, end_reason)

    if game_num % SNAPSHOT_EVERY == 0:
        logger.record_snapshot(game_num, agent)

    # --- Checkpoint ---
    if game_num % CHECKPOINT_EVERY == 0:
        agent.save(CKPT_PATH)
        replay.save(BUFFER_PATH)

    # --- Terminal progress ---
    _game_times.append(time.time() - _t_game_start)
    if len(_game_times) > 20:
        _game_times.pop(0)

    if game_num % PRINT_EVERY == 0 or game_num <= 5:
        w_str      = "W" if winner == chess.WHITE else "B" if winner == chess.BLACK else "D"
        secs       = sum(_game_times) / len(_game_times)
        games_left = N_GAMES - game_num
        eta_h      = (secs * games_left) / 3600
        elapsed_h  = (time.time() - _t_run_start) / 3600
        print(
            f"Game {game_num:>5} | {w_str} | "
            f"moves: {len(moves):>3} | "
            f"loss: {loss:.4f} | "
            f"buffer: {len(replay):>6} | "
            f"steps: {agent.steps:>6} | "
            f"{secs:.0f}s/game | "
            f"elapsed: {elapsed_h:.1f}h | "
            f"ETA: {eta_h:.1f}h"
        )

# ---------------------------------------------------------------------------
# Final checkpoint and summary
# ---------------------------------------------------------------------------

agent.save(CKPT_PATH)
print(f"\nTraining complete — {N_GAMES:,} games, {agent.steps:,} training steps.")
print(f"Checkpoint: {CKPT_PATH}")
print(f"Logs:       {LOG_DIR}/")
