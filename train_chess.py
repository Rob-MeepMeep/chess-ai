"""
train_chess.py — AlphaZero-style self-play training loop for HAL-4000.

LOCKSTEP SELF-PLAY (Run 13 perf work):
  N_PARALLEL_GAMES games run at once in this single process. Each step, every
  game's MCTS contributes its wave of leaf positions to ONE pooled network
  call — 16 games × 8-leaf waves ≈ a 128-position batch — instead of each
  game making its own tiny batch-8 calls. The GPU was idling at ~70%
  utilisation waiting for sequential Python between batch-8 calls; pooling
  across games is how it gets fed properly without any multiprocessing.

  Games are still completely independent: each has its own board, history,
  GameBuffer and resign state. They just share inference. A finished game is
  committed, logged and replaced with a fresh one, so game numbers are
  assigned in completion order.

Each game: use MCTS to select every move, store (position, policy, turn) in
a GameBuffer. At game end fill in outcomes and commit to the ReplayBuffer.
Once the buffer is large enough, run a batch of training steps.

The resign check now reuses the search's own value estimate (the chosen
child's Q) instead of a separate network call per move — a whole search is a
strictly better estimate than one raw forward pass, and it's free.
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
from run_config      import RUN_NAME, CKPT_PATH, BUFFER_PATH, LOG_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_GAMES          = 10_000
N_SIMULATIONS    = 600
N_PARALLEL_GAMES = 16       # games sharing each pooled network call (lockstep)
BATCH_SIZE       = 512
TRAIN_STEPS      = 5        # gradient updates per game (once buffer is ready)
MIN_BUFFER       = 500      # don't train until buffer holds this many positions
MAX_GAME_MOVES   = 150      # hard cap — bumped from 100; resign logic should terminate most games first
CHECKPOINT_EVERY  = 10      # save weights every N games
BUFFER_SAVE_EVERY = 50      # save the replay buffer every N games — was 200 but that
                            # meant hours of self-play data at risk on interrupt
SNAPSHOT_EVERY    = 50      # log MCTS strategy snapshots every N games
PRINT_EVERY       = 10      # print progress line every N games
REGRESSION_EVERY  = 200     # log value head regression to regression.csv
RESIGN_THRESHOLD   = -0.95  # value score below which a position is hopeless
RESIGN_CONSECUTIVE = 5      # raised from 3 — let positions breathe, force more closing technique
# RESIGN_MATERIAL removed for Run 12 — Stage 2 resign now active.
# The resign signal is the search value; note that a fresh network resigns
# nothing (value ≈ 0 everywhere), so early games run to the move cap.

# CKPT_LOAD: None = load RUN_NAME's own checkpoint; set to a path to seed weights from another run.
# BUFFER_LOAD: None = load RUN_NAME's own buffer; set to a path to load from another run.
# RUN_NAME itself lives in run_config.py — shared with eval/watcher/API.
CKPT_LOAD   = None
BUFFER_LOAD = "checkpoints/run13_seed_buffer.pt"

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

if torch.cuda.is_available():       # NVIDIA CUDA or AMD ROCm (appears as "cuda" under ROCm)
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Initialise or resume
# ---------------------------------------------------------------------------

os.makedirs("checkpoints", exist_ok=True)

agent  = ChessAgent(device, n_simulations=N_SIMULATIONS)
replay = ReplayBuffer(capacity=200_000)
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

print(f"N_SIMULATIONS = {N_SIMULATIONS} | N_PARALLEL_GAMES = {N_PARALLEL_GAMES} "
      f"| N_GAMES = {N_GAMES:,}\n")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                 chess.ROOK: 5, chess.QUEEN: 9}

def _material_balance(board: chess.Board) -> int:
    """Positive = white ahead. Used for scoring games that hit the move cap."""
    score = 0
    for piece, val in _PIECE_VALUES.items():
        score += val * len(board.pieces(piece, chess.WHITE))
        score -= val * len(board.pieces(piece, chess.BLACK))
    return score


class SelfPlayGame:
    """One game's state in the lockstep pool: its board, its in-progress
    search, and its resign bookkeeping. Everything the old single-game loop
    kept in local variables, one object per concurrent game."""

    def __init__(self):
        self.board         = chess.Board()
        self.history       = []          # last 3 boards for the 4-frame encoder
        self.buf           = GameBuffer()
        self.moves         = []
        self.search        = None        # in-progress SearchState, or None
        self.resign_streak = 0
        self.v             = 0.0         # search value after the latest move
        self.t_start       = time.time()

    @property
    def over(self) -> bool:
        return (self.board.is_game_over()
                or self.resign_streak >= RESIGN_CONSECUTIVE
                or len(self.moves) >= MAX_GAME_MOVES)


def _finish_game(g: SelfPlayGame) -> tuple:
    """
    Determine winner and end reason, commit the game to the replay buffer.
    A real board result always outranks resignation: if checkmate landed on
    the same ply the resign streak filled, the board is the truth — the old
    order let the value head's opinion overwrite an actual mate.
    """
    outcome_scale = 1.0

    if g.board.is_game_over():
        result = g.board.result()
        if result == "1-0":
            winner, end_reason = chess.WHITE, "checkmate"
        elif result == "0-1":
            winner, end_reason = chess.BLACK, "checkmate"
        else:
            winner, end_reason = None, "rule_draw"   # stalemate, repetition, 50-move, bare kings
    elif g.resign_streak >= RESIGN_CONSECUTIVE:
        # Value resignation: winner is whoever the search says is winning.
        # g.v is from the perspective of the player to move after the last push.
        if g.v > 0:
            winner = g.board.turn
        else:
            winner = chess.WHITE if g.board.turn == chess.BLACK else chess.BLACK
        end_reason = "value_resign"
    else:
        # Move cap: if one side is clearly ahead on material, treat as a soft win
        # rather than a draw. Assigning outcome 0.0 to lopsided positions contaminates
        # the value head — it learns "endgame-looking positions = draw" which directly
        # contradicts the canonical endgame signal.
        cap_mat = _material_balance(g.board)
        if abs(cap_mat) > 3:
            winner        = chess.WHITE if cap_mat > 0 else chess.BLACK
            outcome_scale = 0.8   # strong signal without claiming a win was forced
        else:
            winner = None
        end_reason = "cap_draw"

    g.buf.commit(replay, winner, scale=outcome_scale)
    return winner, end_reason


# ---------------------------------------------------------------------------
# Lockstep training loop
# ---------------------------------------------------------------------------

_t_run_start  = time.time()
_game_times: list = []                   # per-game durations (overlapping wall time)
_tally_w, _tally_b, _tally_d = 0, 0, 0   # W/B/D counts since last tally reset

game_num      = start_game               # completed-game counter (log numbering)
games_started = start_game
active: list  = []
loss          = 0.0
mcts          = agent.mcts

try:
    while game_num < N_GAMES:

        # Keep the pool full while there are games left to schedule
        while len(active) < N_PARALLEL_GAMES and games_started < N_GAMES:
            active.append(SelfPlayGame())
            games_started += 1
        if not active:
            break

        # --- 1. Every game needs a search in progress ---
        for g in active:
            if g.search is None:
                g.search = mcts.begin_search(g.board, g.history,
                                             N_SIMULATIONS, add_noise=True)

        # --- 2. Pool every game's leaf wave into ONE network call ---
        batches = [mcts.gather_leaves(g.search) for g in active]
        tensors = [b for b in batches if b is not None]
        if tensors:
            logits, values = mcts.evaluate(torch.cat(tensors))
        off = 0
        for g, b in zip(active, batches):
            if b is None:
                mcts.apply_results(g.search, None, None)   # all-terminal wave: backup only
            else:
                n = b.shape[0]
                mcts.apply_results(g.search, logits[off:off + n], values[off:off + n])
                off += n

        # --- 3. Finished searches become moves; finished games get replaced ---
        finished = []
        for g in active:
            if not g.search.done:
                continue

            # Encode current position BEFORE making the move
            state = encode([g.board] + g.history)

            policy = mcts.extract_policy(g.search)
            move_uci, policy, v = agent.pick_move(g.board, policy, g.search.root,
                                                  greedy=False)
            g.search = None
            g.v      = v

            # Policy targets are stored in the network's frame of reference —
            # mirrored when it was black to move, matching the encoded state
            stored_policy = mirror_policy(policy) if g.board.turn == chess.BLACK else policy
            g.buf.push(state, stored_policy, g.board.turn)
            g.moves.append(move_uci)

            # Keep only the last 3 boards — encoder uses current + 3 history = 4 total
            g.history = ([g.board.copy()] + g.history)[:3]
            g.board.push_uci(move_uci)

            # Stage 2 resign: the search value is the sole resign signal.
            # abs() — resign regardless of which side is hopeless.
            if abs(v) > abs(RESIGN_THRESHOLD):
                g.resign_streak += 1
            else:
                g.resign_streak = 0

            if g.over:
                finished.append(g)

        # --- 4. Commit finished games: outcomes, training, logging, checkpoints ---
        for g in finished:
            active.remove(g)
            game_num += 1
            winner, end_reason = _finish_game(g)

            if replay.ready(MIN_BUFFER):
                for _ in range(TRAIN_STEPS):
                    loss = agent.train(replay.sample(BATCH_SIZE))

            logger.record_game(game_num, winner, g.moves, loss, end_reason,
                               steps=agent.steps)

            if game_num % SNAPSHOT_EVERY == 0:
                logger.record_snapshot(game_num, agent)
            if game_num % REGRESSION_EVERY == 0:
                logger.record_regression(game_num, agent)
            if game_num % CHECKPOINT_EVERY == 0:
                agent.save(CKPT_PATH)
            if game_num % BUFFER_SAVE_EVERY == 0:
                replay.save(BUFFER_PATH)
                agent.save(CKPT_PATH)   # keep weights in sync with buffer

            # --- Terminal progress ---
            _game_times.append(time.time() - g.t_start)
            if len(_game_times) > 20:
                _game_times.pop(0)

            if winner == chess.WHITE:
                _tally_w += 1
            elif winner == chess.BLACK:
                _tally_b += 1
            else:
                _tally_d += 1

            if game_num % PRINT_EVERY == 0 or game_num <= 5:
                w_str     = "W" if winner == chess.WHITE else "B" if winner == chess.BLACK else "D"
                elapsed_h = (time.time() - _t_run_start) / 3600
                done_n    = game_num - start_game
                # Throughput, not per-game time — games overlap in the lockstep pool
                rate      = done_n / elapsed_h if elapsed_h > 0 else 0.0
                eta_h     = (N_GAMES - game_num) / rate if rate > 0 else float("inf")
                tally_str = f"W{_tally_w}/B{_tally_b}/D{_tally_d}"
                print(
                    f"Game {game_num:>5} | {w_str} | "
                    f"moves: {len(g.moves):>3} | "
                    f"loss: {loss:.4f} | "
                    f"[{tally_str}] | "
                    f"buffer: {len(replay):>6} | "
                    f"steps: {agent.steps:>6} | "
                    f"{rate:.1f} games/h | "
                    f"elapsed: {elapsed_h:.1f}h | "
                    f"ETA: {eta_h:.1f}h"
                )

            # Reset tally every 50 games so it stays readable
            if game_num % 50 == 0:
                _tally_w, _tally_b, _tally_d = 0, 0, 0


except KeyboardInterrupt:
    print(f"\n{'='*60}")
    print(f"Interrupted at game {game_num} — saving checkpoint and buffer...")
    agent.save(CKPT_PATH)
    replay.save(BUFFER_PATH)
    print(f"  Checkpoint: {CKPT_PATH}")
    print(f"  Buffer:     {BUFFER_PATH}  ({len(replay):,} rolling + {len(replay._permanent):,} permanent)")
    print(f"  Resume will start at game {game_num + 1}")
    print(f"{'='*60}")

else:
    # Natural completion — no interrupt
    agent.save(CKPT_PATH)
    replay.save(BUFFER_PATH)
    print(f"\nTraining complete — {game_num:,} games, {agent.steps:,} training steps.")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Logs:       {LOG_DIR}/")
