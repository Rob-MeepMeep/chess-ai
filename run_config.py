"""
run_config.py — Single source of truth for the active training run.

train_chess.py, eval_chess.py, eval_watcher.py and main.py all derive their
checkpoint and log paths from this. Previously each file carried its own
copy of the run name, and they drifted (eval loading run10 while training
wrote run13, the watcher polling a finished run's log, the API server
loading a checkpoint name no run ever produced).

Change RUN_NAME here when starting a new run — nowhere else.
"""

# run18: generalisation-gap Option C -- self-play positions in the 1-7
# material band relabelled with real Stockfish evaluations
# (relabel_with_stockfish.py), folded into the permanent partition
# alongside the canonical positions. Both self-play adjudication attempts
# (rung 1 in run15, rung 1b in run17) proved unable to generate training
# signal for material imbalances below 8 -- run17's material_probe
# diagnostic showed missing_bishop/knight declining to essentially never
# correct (3%) across three large windows, worse in relative terms than
# run16's terminal state. See paper/run17_decision_protocol.md and
# paper/generalization_gap_options.md. run18 warm-starts from run17's
# checkpoint (warm_start_run18.py) -- no architecture change, straight
# weight copy. run17's logs/weights/eval results stay untouched for
# comparison.
RUN_NAME = "run18"

CKPT_PATH   = f"checkpoints/{RUN_NAME}_hal_chess.pt"
BUFFER_PATH = f"checkpoints/{RUN_NAME}_replay_buffer.pt"
LOG_DIR     = f"logs/{RUN_NAME}"
