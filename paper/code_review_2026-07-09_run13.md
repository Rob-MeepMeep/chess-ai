# chess-ai Codebase Review — Run 13 (main @ c913bb5)

**Date:** 9 July 2026
**Authors:** Rob Kirkland, Ellis Ward
**Scope:** Full review of every Python file (~4,500 lines): `chessai/` package,
`train_chess.py`, `eval_chess.py`, `curate_buffer.py`, `main.py`, run13 tooling
(`dashboard.py`, `eval_watcher.py`, `extract_buffer_candidates.py`), and the completed
`connect4/` and `tictactoe/` phases. FEN claims below were verified with python-chess.

---

## Summary

The AlphaZero machinery — the hard part — is correct. The MCTS backup sign convention,
the board mirroring for Black (encoder mirroring, policy un-mirroring at expansion,
re-mirroring of training targets in the train loop), virtual loss bookkeeping, batched
leaf evaluation, and terminal value handling were traced end-to-end and all hold
together. This is the part most implementations get wrong.

The significant problems are in the **training-signal plumbing**: hand-written positions
feeding the value head carry wrong ground-truth labels, and the regression metric used
to make run decisions cannot measure what it is trusted to measure.

---

## Findings

### 1. CRITICAL — Canonical endgame FENs are illegal or mislabeled
`curate_buffer.py:68-80`

All six static K+Q and K+R positions place the piece on e1 directly adjacent to the
enemy king on d1, with the white king on d5 — too far away to defend it. Verified:

- The three **"+1.0, winner to move"** versions are illegal positions — the side *not*
  to move is in check (`board.is_valid()` returns `False`).
- The three **"-1.0, loser to move"** versions are legal, but the losing king simply
  plays **Kd1xe1**, capturing the undefended queen/rook. The true outcome is a dead
  draw (0.0), stored as a certain loss (-1.0).

These enter the permanent buffer partition at ×5 repeats. With `perm_ratio` raised to
0.33 (`chessai/replay.py:55`), permanent positions are oversampled roughly 100× versus
rolling positions — wrong labels here appear in every batch, indefinitely.

### 2. CRITICAL — Diverse generators mislabel capturable-piece positions
`curate_buffer.py:131-264`

All three generators label every "loser to move" position as a certain loss, but the
`is_valid` / `is_checkmate` / `is_stalemate` filters do not catch the case where the
losing king can immediately capture the hanging piece:

- `generate_diverse_kq_vs_k` / `generate_diverse_kr_vs_k`: roughly one in ten
  black-to-move samples places the black king adjacent to an undefended queen/rook —
  black captures, draw — labeled -1.0.
- `generate_diverse_kq_vs_kp`: the black pawn may spawn on rank 2, one move from
  promotion. Black to move promotes into K+Q vs K+Q (drawish), labeled -0.9. The
  queen-capture adjacency case applies here too.

### 3. HIGH — Regression metric rests on an illegal and a terminal position
`eval_chess.py:93-99`, `chessai/logger.py` (`record_regression`)

- The `w_wins` FEN (`8/8/8/8/8/6K1/6Q1/7k w`) is **illegal** — Black is in check while
  it is White's move.
- The `b_move` FEN is **checkmate** — a terminal position. `GameBuffer` only stores
  pre-move states, so the value head is never trained on terminal positions at all.

This matters because run decisions cite this metric: material-resign was removed for
Run 12+ on the grounds that "w_wins and b_move have been ≥ ±0.99 for thousands of
games" (`train_chess.py:44-46`). The metric can read +0.99 while conversion ability is
untested — which is exactly the cap-draw symptom described in the run notes.

### 4. HIGH — `--prev` eval crashes with NameError
`eval_chess.py:284`

`ChessAgent(device, n_simulations=N_SIMULATIONS)` — `N_SIMULATIONS` is not defined in
this file (the defined constants are `N_SIMS_RANDOM` and `SIMS_BY_DEPTH`). The
surrounding `try` only catches `FileNotFoundError`, so the Tier 3
previous-checkpoint comparison kills the eval run.

### 5. MEDIUM — Resign branch outranks a real checkmate
`train_chess.py:189-194`

If checkmate lands on the same ply the resign streak reaches 5, the
`resign_streak >= RESIGN_CONSECUTIVE` branch takes precedence and the winner is taken
from the value head's opinion rather than the actual board result. A mate delivered by
the "losing" side is recorded backwards, poisoning that entire game's outcome labels.
Fix is cheap: check `board.is_game_over()` before the resign branch.

### 6. MEDIUM — Value-only resign assumes a trained value head; Run 13 starts fresh
`train_chess.py:52-54`

`run13_retune` sets `CKPT_LOAD = None` — fresh 20×256 weights — but the material-resign
fallback was removed on the premise that the value head is already trained. A fresh
value head outputs ~0, so nothing resigns; every early game runs to the 150-move cap at
600 simulations, and the cap-draw material rule becomes the dominant label source.
Possibly intentional (the seed buffer supplies early signal), but the Stage-2 rationale
does not hold for a from-scratch run. Decision needed rather than code.

### 7. LOW — Stale run-name constants scattered across files

- `main.py:25` loads `checkpoints/hal_chess.pt` — no run produces that filename; the
  API prints one startup warning and then silently serves an untrained network.
- `eval_watcher.py:19` watches `run12` while training writes `run13_retune` — the
  watcher never fires.
- `curate_buffer.py` docstring says Run 11 output but writes run13; `eval_chess.py`
  header says Stockfish depths 1/3/5 but the loop is `[1]`; `model.py` docstring still
  describes 10 blocks / 160 filters and the old hardware.

A single shared `RUN_NAME` source (small config module or environment variable) used by
train/eval/watcher/curate/main would eliminate this class of drift.

### 8. LOW — Miscellaneous

- `curate_buffer.py` has no `if __name__ == "__main__"` guard — importing it executes
  the full pipeline. (`extract_buffer_candidates.py` and `eval_watcher.py` have guards.)
- At Run 13 scale the replay buffer is ~6 GB in memory (dense 4,096-float policy per
  position plus 54×8×8 states) and is `torch.save`d every 10 games — slow saves and
  heavy SSD writes on long runs. Sparse policy storage (legal-move entries only) or
  float16 would cut this substantially.
- Zero-policy permanent positions mean roughly a third of each batch contributes no
  policy gradient — acceptable if intentional, but it silently deflates loss numbers
  when comparing across runs.
- `random.sample` over a 200k-entry deque is O(n·k) per batch; a list or array index
  is a cheap improvement.
- Minor MCTS note: on the first selection of each wave all root children have UCB 0
  (root visit count is 0), so the first simulation always picks the first legal move
  rather than the highest-prior move. Self-corrects within the wave; negligible.

### Verified sound — no action needed

MCTS backup and perspective mathematics; encoder mirroring including history frames,
castling and en-passant planes; policy re-mirroring in the training loop; virtual loss
add/remove symmetry; batched leaf deduplication; terminal value handling; the
underpromotion-to-queen simplification (documented and consistently applied); the
Connect Four and Tic-Tac-Toe phase code (complete phases, only minor known two-agent
DQN wrinkles that no longer matter).

---

## Recommendations

1. **`curate_buffer.py`** — replace the six static FENs with legal positions where the
   piece is defended or distant; add a "loser cannot immediately capture the piece"
   check (and a promotion-distance check for K+Q vs K+P) to all three generators.
   Consider validating every generated label before it enters the permanent set.
2. **Regression positions** — replace both broken FENs with legal, non-terminal,
   unambiguous positions verified once against Stockfish or a tablebase. Keep the
   `regression.csv` column names, and note the discontinuity in the run notes.
3. **`eval_chess.py:284`** — replace `N_SIMULATIONS` with a defined constant.
4. **`train_chess.py`** — real game results before the resign branch; decide whether
   Run 13 from scratch needs a temporary bootstrap resign rule or deliberately leans on
   the seed buffer.
5. **Config hygiene** — single run-name source; fix the `main.py` checkpoint path; add
   a `__main__` guard to `curate_buffer.py`; refresh stale docstrings.

Regenerating the Run 13 seed buffer after fix 1 is required for it to take effect.

## Verification (desktop training machine)

- Re-run `curate_buffer.py` with an assertion pass: every permanent position is
  `is_valid()`, non-terminal, and its label survives a hanging-piece/promotion check.
- `python eval_chess.py --regression-only` — new positions produce sensible values.
- `python eval_chess.py --prev <old checkpoint>` — Tier 3 completes without error.
- Short smoke run (or a unit test of the winner-determination block) — a same-ply
  checkmate-plus-resign game records the board result, not the value head's opinion.
