"""
run_config.py — Single source of truth for the active training run.

train_chess.py, eval_chess.py, eval_watcher.py and main.py all derive their
checkpoint and log paths from this. Previously each file carried its own
copy of the run name, and they drifted (eval loading run10 while training
wrote run13, the watcher polling a finished run's log, the API server
loading a checkpoint name no run ever produced).

Change RUN_NAME here when starting a new run — nowhere else.
"""

# run19: Option C, escalated. run18's material_probe result (64 readings,
# three windows) showed missing_queen solved decisively (100% correct
# every window, cross-verified by regression.csv) and missing_rook strong
# (76-95%), but missing_bishop/knight/two_pawns declined the same way
# rung 1b's self-play-only attempt did -- the alpha=0.2 blend used in
# run18 was likely too weak at smaller material magnitudes, where
# Stockfish's evaluation is a softer signal (rarely forcing/mate-adjacent)
# competing against the +-1.0 character of self-play outcomes elsewhere
# in the batch. run19 escalates to alpha=0.05 (95% Stockfish trust) via
# curate_buffer.py's STOCKFISH_ALPHA -- cheap this time, since
# relabel_with_stockfish.py now saves raw (self-play outcome, Stockfish
# eval) pairs instead of a pre-blended value, so no Stockfish re-run was
# needed. See paper/run18_decision_protocol.md and
# paper/phase3_rl_arc_closing_report.md. run19 warm-starts from run18's
# checkpoint (warm_start_run19.py) -- no architecture change. run18's
# logs/weights/eval results stay untouched for comparison.
RUN_NAME = "run19"

CKPT_PATH   = f"checkpoints/{RUN_NAME}_hal_chess.pt"
BUFFER_PATH = f"checkpoints/{RUN_NAME}_replay_buffer.pt"
LOG_DIR     = f"logs/{RUN_NAME}"
