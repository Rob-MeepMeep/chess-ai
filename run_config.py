"""
run_config.py — Single source of truth for the active training run.

train_chess.py, eval_chess.py, eval_watcher.py and main.py all derive their
checkpoint and log paths from this. Previously each file carried its own
copy of the run name, and they drifted (eval loading run10 while training
wrote run13, the watcher polling a finished run's log, the API server
loading a checkpoint name no run ever produced).

Change RUN_NAME here when starting a new run — nowhere else.
"""

# run17: intervention ladder rung 1b -- a second, lower-confidence material
# adjudication tier (3-7 imbalance, 16-ply streak, 0.7 scale) alongside
# run16's existing 8+ tier. run16's material_probe diagnostic showed
# queen/rook-scale positions genuinely improving but bishop/knight/two_pawns
# getting WORSE, monotonically, over 2,000+ games -- the 8-point threshold
# only ever reinforces large imbalances, so self-play had no mechanism to
# teach it smaller ones matter. See paper/run16_decision_protocol.md and
# paper/run17_decision_protocol.md. run17 warm-starts from run16's
# checkpoint (warm_start_run17.py) -- no architecture change this time,
# so it's a straight weight copy with a fresh optimizer/step count, not
# the weight surgery run16 needed. run16's logs/weights/eval results stay
# untouched for comparison.
RUN_NAME = "run17"

CKPT_PATH   = f"checkpoints/{RUN_NAME}_hal_chess.pt"
BUFFER_PATH = f"checkpoints/{RUN_NAME}_replay_buffer.pt"
LOG_DIR     = f"logs/{RUN_NAME}"
