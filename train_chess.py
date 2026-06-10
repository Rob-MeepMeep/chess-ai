"""
train_chess.py — AlphaZero-style self-play training loop for HAL-4000.

Each game: use MCTS to select every move, store (position, policy, turn) in
a GameBuffer. At game end fill in outcomes and commit to the ReplayBuffer.
Once the buffer is large enough, run a batch of training steps.

Speed guide (MacBook Pro M5 Pro, MPS, untrained network):
  N_SIMULATIONS = 200  — ~90-115s/game, ~250-320 hours for 10k games
  N_SIMULATIONS = 100  — ~45-60s/game,  ~125-165 hours for 10k games
  N_SIMULATIONS =  50  — ~20-30s/game,  ~55-85 hours for 10k games
  N_SIMULATIONS =  25  — ~10-15s/game,  ~28-42 hours for 10k games

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
from chessai.moves   import mirror_policy
from chessai.logger  import Logger
from chessai.replay  import ReplayBuffer, GameBuffer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_GAMES          = 10_000
N_SIMULATIONS    = 100      # halved from 200 — ~2x speedup; quality tradeoff acceptable at current training stage
BATCH_SIZE       = 64
TRAIN_STEPS      = 5        # gradient updates per game (once buffer is ready)
MIN_BUFFER       = 500      # don't train until buffer holds this many positions
MAX_GAME_MOVES   = 150      # hard cap — bumped from 100; resign logic should terminate most games first
CHECKPOINT_EVERY = 10       # save checkpoint every N games
SNAPSHOT_EVERY   = 50      # log MCTS strategy snapshots every N games
PRINT_EVERY      = 10       # print progress line every N games
RESIGN_THRESHOLD   = -0.95  # value head score below which a position is hopeless
RESIGN_CONSECUTIVE = 5      # raised from 3 — let positions breathe, force more closing technique
RESIGN_MATERIAL    = 3      # lowered from 7 for Run 9 — forces HAL to grind out major advantages
                            # (rook/queen up) rather than resigning early; floods rolling buffer
                            # with winning-side positions to balance the b-move training signal

# Run identity — change RUN_NAME to start a new named run with its own logs and buffer.
# CKPT_LOAD: None = load RUN_NAME's own checkpoint; set to a path to seed weights from another run.
# BUFFER_LOAD: None = load RUN_NAME's own buffer; set to a path to load from another run.
RUN_NAME    = "run10"
CKPT_LOAD   = "checkpoints/run9_hal_chess.pt"           # continue from run9 — value head healthy, w-wins 0.948, b-move 0.995
BUFFER_LOAD = "checkpoints/run10_seed_buffer.pt"        # curated seed buffer from run9 games 800-1000

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
replay = ReplayBuffer(capacity=75_000)
logger = Logger(log_dir=LOG_DIR, snapshot_interval=SNAPSHOT_EVERY)

start_game = 0
# Prefer the run's own checkpoint on resume; fall back to CKPT_LOAD for a fresh start.
# Mirrors the buffer logic — CKPT_LOAD is only used when no own checkpoint exists yet.
if os.path.exists(CKPT_PATH):
    _ckpt_to_load = CKPT_PATH
    if CKPT_LOAD and CKPT_LOAD != CKPT_PATH:
        print(f"  Note: CKPT_LOAD ignored — own checkpoint found at {CKPT_PATH}")
else:
    _ckpt_to_load = CKPT_LOAD or CKPT_PATH
if os.path.exists(_ckpt_to_load):
    agent.load(_ckpt_to_load)
    if _ckpt_to_load == CKPT_PATH:
        openings_path = os.path.join(LOG_DIR, "openings.csv")
        if os.path.exists(openings_path):
            with open(openings_path) as f:
                rows = list(csv.reader(f))
            if len(rows) > 1:
                start_game = int(rows[-1][0])
    print(f"Loaded weights from {_ckpt_to_load} — starting at game {start_game + 1}")
    print(f"  Trained steps so far: {agent.steps:,}")
else:
    print("Starting fresh training run.")

# Prefer the run's own accumulated buffer on resume; fall back to seed buffer
# for a fresh start. BUFFER_LOAD is only used when no accumulated buffer exists yet.
if os.path.exists(BUFFER_PATH):
    _buf_to_load = BUFFER_PATH
    if BUFFER_LOAD and BUFFER_LOAD != BUFFER_PATH:
        print(f"  Note: BUFFER_LOAD ignored — accumulated buffer found at {BUFFER_PATH}")
else:
    _buf_to_load = BUFFER_LOAD or BUFFER_PATH
if os.path.exists(_buf_to_load):
    replay.load(_buf_to_load)
    perm_n = len(replay._permanent)
    perm_str = f" + {perm_n:,} permanent" if perm_n else ""
    print(f"  Replay buffer loaded: {len(replay):,} rolling{perm_str} ({_buf_to_load})")

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
_tally_w, _tally_b, _tally_d = 0, 0, 0  # W/B/D counts since last tally reset

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
        move_uci, policy = agent.choose_move(board, history, greedy=False)
        if board.turn == chess.BLACK:
            policy = mirror_policy(policy)

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
        val_hopeless = abs(v) > 0.95   # RESIGN_THRESHOLD magnitude
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
    result        = board.result()
    outcome_scale = 1.0   # overridden to 0.8 for material-imbalanced cap draws
    if resign_streak >= RESIGN_CONSECUTIVE:
        if resign_cause == "material":
            winner = chess.WHITE if _material_balance(board) > 0 else chess.BLACK
        else:
            # value resignation: winner is whoever the value head says is winning
            winner = board.turn if v > 0 else (chess.WHITE if board.turn == chess.BLACK else chess.BLACK)
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
        # Cap draw: if one side is clearly ahead on material, treat as a soft win
        # rather than a draw. Assigning outcome 0.0 to lopsided positions contaminates
        # the value head — it learns "endgame-looking positions = draw" which directly
        # contradicts the canonical endgame signal.
        _cap_mat = _material_balance(board)
        if abs(_cap_mat) > 3:
            winner        = chess.WHITE if _cap_mat > 0 else chess.BLACK
            outcome_scale = 0.8   # strong signal without claiming a win was forced
        else:
            winner        = None
            outcome_scale = 1.0
        end_reason = "cap_draw"
    else:
        winner     = None
        end_reason = "rule_draw"

    # Commit game positions with outcomes to replay buffer
    game_buf.commit(replay, winner, scale=outcome_scale)

    # --- Training steps ---
    if replay.ready(MIN_BUFFER):
        for _ in range(TRAIN_STEPS):
            batch = replay.sample(BATCH_SIZE)
            loss  = agent.train(batch)

    # --- Logging ---
    logger.record_game(game_num, winner, moves, loss, end_reason, steps=agent.steps)

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

    # Running W/B/D tally — resets every 50 games so bias is visible early
    if winner == chess.WHITE:
        _tally_w += 1
    elif winner == chess.BLACK:
        _tally_b += 1
    else:
        _tally_d += 1

    if game_num % PRINT_EVERY == 0 or game_num <= 5:
        w_str      = "W" if winner == chess.WHITE else "B" if winner == chess.BLACK else "D"
        secs       = sum(_game_times) / len(_game_times)
        games_left = N_GAMES - game_num
        eta_h      = (secs * games_left) / 3600
        elapsed_h  = (time.time() - _t_run_start) / 3600
        tally_str  = f"W{_tally_w}/B{_tally_b}/D{_tally_d}"
        print(
            f"Game {game_num:>5} | {w_str} | "
            f"moves: {len(moves):>3} | "
            f"loss: {loss:.4f} | "
            f"[{tally_str}] | "
            f"buffer: {len(replay):>6} | "
            f"steps: {agent.steps:>6} | "
            f"{secs:.0f}s/game | "
            f"elapsed: {elapsed_h:.1f}h | "
            f"ETA: {eta_h:.1f}h"
        )

    # Reset tally every 50 games so it stays readable
    if game_num % 50 == 0:
        _tally_w, _tally_b, _tally_d = 0, 0, 0

# ---------------------------------------------------------------------------
# Final checkpoint and summary
# ---------------------------------------------------------------------------

agent.save(CKPT_PATH)
print(f"\nTraining complete — {N_GAMES:,} games, {agent.steps:,} training steps.")
print(f"Checkpoint: {CKPT_PATH}")
print(f"Logs:       {LOG_DIR}/")
