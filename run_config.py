"""
run_config.py — Single source of truth for the active training run.

train_chess.py, eval_chess.py, eval_watcher.py and main.py all derive their
checkpoint and log paths from this. Previously each file carried its own
copy of the run name, and they drifted (eval loading run10 while training
wrote run13, the watcher polling a finished run's log, the API server
loading a checkpoint name no run ever produced).

Change RUN_NAME here when starting a new run — nowhere else.
"""

# run16: generalisation-gap Option A (material-count input plane) on top of
# run15's rung-1 intervention. run15's Gate 3 was GREEN (90.0% wins vs
# random, later reaching a perfect 100.0%) but missing_queen never moved
# and Stockfish depth 1 stayed 0/300 across the whole project — rung 1
# fixed conversion, not generalisation. See paper/generalization_gap_options.md.
# run16 warm-starts from run15's checkpoint (warm_start_run16.py) rather
# than training from scratch a fourth time; its logs/weights/eval results
# stay untouched for comparison.
RUN_NAME = "run16"

CKPT_PATH   = f"checkpoints/{RUN_NAME}_hal_chess.pt"
BUFFER_PATH = f"checkpoints/{RUN_NAME}_replay_buffer.pt"
LOG_DIR     = f"logs/{RUN_NAME}"
