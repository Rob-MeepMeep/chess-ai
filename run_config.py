"""
run_config.py — Single source of truth for the active training run.

train_chess.py, eval_chess.py, eval_watcher.py and main.py all derive their
checkpoint and log paths from this. Previously each file carried its own
copy of the run name, and they drifted (eval loading run10 while training
wrote run13, the watcher polling a finished run's log, the API server
loading a checkpoint name no run ever produced).

Change RUN_NAME here when starting a new run — nowhere else.
"""

RUN_NAME = "run13_retune"

CKPT_PATH   = f"checkpoints/{RUN_NAME}_hal_chess.pt"
BUFFER_PATH = f"checkpoints/{RUN_NAME}_replay_buffer.pt"
LOG_DIR     = f"logs/{RUN_NAME}"
