"""
run_config.py — Single source of truth for the active training run.

train_chess.py, eval_chess.py, eval_watcher.py and main.py all derive their
checkpoint and log paths from this. Previously each file carried its own
copy of the run name, and they drifted (eval loading run10 while training
wrote run13, the watcher polling a finished run's log, the API server
loading a checkpoint name no run ever produced).

Change RUN_NAME here when starting a new run — nowhere else.
"""

# run20: not an experimental change -- a measurement fix. run19's
# material_probe (game ~1190, step 5950) was showing missing_bishop/
# knight/two_pawns flat/declining, same as run17 and run18 before it.
# Turned out to be a probe-construction bug, not a training gap: the
# synthetic test FENs (starting position minus one piece, zero moves
# played, empty history) can't occur naturally, and the network had
# never seen anything shaped like them. On real in-context positions at
# the same magnitudes, the same run19 checkpoint reads a clean, correctly
# -0.30 to -0.37 -- see paper/material_probe_correction.md for the full
# diagnosis and evidence (diagnose_material_labels.py,
# diagnose_network_output.py, diagnose_probe_construction.py) and
# chessai/logger.py's MATERIAL_PROBE_POSITIONS comment for the fix itself.
#
# run20 warm-starts from run19's checkpoint (warm_start_run20.py) with
# nothing else changed -- same architecture, same STOCKFISH_ALPHA=0.05,
# same accumulated replay buffer carried forward (BUFFER_LOAD in
# train_chess.py points at run19's buffer, not a fresh curate_buffer.py
# run, since nothing about the data pipeline changed). The only reason
# for a new run number at all is a clean material_probe.csv from game 0
# -- run19's own file has the old synthetic-FEN readings before the fix
# and the new real-position ones after, not directly comparable within
# one file.
RUN_NAME = "run20"

CKPT_PATH   = f"checkpoints/{RUN_NAME}_hal_chess.pt"
BUFFER_PATH = f"checkpoints/{RUN_NAME}_replay_buffer.pt"
LOG_DIR     = f"logs/{RUN_NAME}"
