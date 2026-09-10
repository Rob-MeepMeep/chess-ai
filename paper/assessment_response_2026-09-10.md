# Response to the 10 September 2026 Independent Assessment

**Project:** chess-ai
**Phase:** 3 — AlphaZero-style chess agent (HAL-4000)
**Authors:** Rob Kirkland, Ellis Ward
**Date:** 10 September 2026
**Status:** Changelog and forward plan, addressed to the assessment team.
Covers every fix made in direct response to `assessment/
project_assessment_2026-09-10.md`, plus how we intend to tackle the
larger design-level recommendations that aren't done yet. Revised once
after a second review pass on this document itself (five implementation
notes plus a framing correction, all folded in below rather than
appended as a separate addendum, since this hadn't gone out yet).

---

## 1. Summary

The report's implementation findings held up under direct reproduction —
every specific bug we could isolate and test (the buffer eviction order,
the MCTS prior sum, the end-reason key mismatch, the sim-budget
asymmetry) reproduced exactly as described and stayed fixed under
verification. That's a different, narrower claim than "the assessment
was right about all of it," which overstates it: the report's broader
behavioural interpretations — *why* HAL plays the way it does, what a
given number implies about capability — are plausible reads of the
evidence, not verified facts, and this document shouldn't inherit more
confidence in them than that. Where we've treated something as settled
below, it's because we independently reproduced it ourselves, not because
the report asserted it.

This also caught two mistakes in our own work done *during* this
project's remediation of an earlier finding (the `material_probe`
correction) — a pooled-bucket comparison that looked like independent
evidence for bishop and knight but wasn't, and a mischaracterisation of
`snapshots.csv` as raw network confidence when it's actually MCTS visit
shares. Both are acknowledged below rather than argued with.

As the assessment itself put it: this materially improves the project's
experimental foundation. It doesn't yet show HAL plays better — it makes
the next comparison much more capable of actually answering that
question.

Every specific, reproducible finding in the report's implementation
table is fixed, verified, and pushed. The larger design-level
recommendations (a tactical benchmark, a resignation-calibration audit,
a properly controlled LR experiment, an automated test suite, a
benchmark manifest) are not implemented yet — §3 covers the order we
intend to tackle them in and why.

## 2. What was fixed

Each item was reproduced in isolation before fixing where practical, and
swept for other consumers/duplicated logic in the codebase before being
considered done — several turned up a second, independent copy of the
same bug that the report itself didn't flag (noted below).

| Finding | Fix | Commit |
|---|---|---|
| `end_reasons.csv` silently dropped every `material_adjudication`/`material_adjudication_moderate` game (64% of run20) — window-key name mismatch, not a real match | Explicit `_END_REASON_TO_WINDOW_KEY` map covering all six real end-reason strings, with a loud warning instead of a silent drop for anything unmapped. Same root-cause bug also found and fixed in `dashboard.py`'s end-reason chart colour map. | `c2fc4ef` |
| `eval_chess.py` Tier 3 gave the current checkpoint half the search budget (50 sims) of the previous one (100 sims) in every `--prev` run | Paired both sides through `hal_move_at(N_SIMS_PREV)`, matching how Tier 2 already handled per-call sim budgets correctly | `fdd02ed` |
| Replay buffer eviction order corrupted by any restart after the ring had wrapped (`save()` persisted physical slot order, not the write cursor; `load()` reset the cursor to 0 and assumed chronological order) | Rotate to true chronological order in `save()` before writing, so `load()`'s existing (previously false) assumption becomes actually true. Reproduced first (capacity-3 buffer, confirmed wrong entry evicted after reload), confirmed fixed after. | `9112e2d` |
| MCTS child priors didn't sum to 1 at any position with a promotable pawn (all four promotion pieces collide on one policy index; softmax computed the redundant entries before the dict write silently discarded three of four) | `np.unique()` the move indices before the softmax, not after. Reproduced (0.677 instead of 1.0 for a real position), confirmed fixed (1.0 within float32 tolerance) for promotion, non-promotion, and black-to-move-mirrored cases. | `37af31f` |
| Material/resign adjudication streaks tracked magnitude only, never checked the *same side* stayed favoured — a lead that flips sides mid-streak could satisfy the threshold | Added side-tracking to all three streak counters (`train_chess.py`'s `SelfPlayGame`, plus an independent duplicate of the same bug found in `eval_chess.py`'s `play_game`). Verified the fixed state machine against a scripted flip scenario. | `13aee6d` |
| `/move` never validated its own documented "sanity check" FEN against the replayed board; no rejection of terminal positions; `n_simulations` unbounded; `/health` always reported "ok" regardless of whether the checkpoint loaded | FEN now compared via `epd()`; terminal positions and out-of-range `n_simulations` (1–1000) rejected; `/health` reports `checkpoint_loaded` and a `degraded` status when it's false. Live-tested all five cases (no `httpx` in this venv, so tested by calling the route functions directly). | `b10148c` |
| Requested audit of `train_chess.py`/`eval_watcher.py` for dead code and stale docs | `eval_watcher.py`'s docstring still named `run12` though the code has followed `run_config.LOG_DIR` dynamically for a while. `train_chess.py` imported `RUN_NAME` and never used it (removed), and tracked a rolling window of per-game durations (`_game_times`) that was written every game and never read — replaced with a deque of completion *timestamps* and a real "recent games/h" figure in the progress line, since durations don't sum to a valid rate when 16 games run concurrently in the lockstep pool. | `e13084c` |
| Uncalibrated Stockfish-depth→ELO mapping in `eval_chess.py`'s docstring; closing report's "0 draws... across the entire project" overclaimed beyond what it actually measured | ELO mapping replaced with an explicit note that fixed-depth Stockfish is a repeatable opponent, not a rating instrument. Added a dated addendum to `phase3_rl_arc_closing_report.md` citing the real, earlier draws (`run_notes.md`, Run 10, game 59) the "entire project" wording missed. | `ecf9094` |
| Headline win-rate metric conflated real chess outcomes with material-adjudicated ones; no way to play evaluation out to genuine conclusion | `play_game()` now returns an explicit `end_reason`; `evaluate()` reports checkmate vs adjudicated wins and real vs unresolved-move-cap draws separately, not one blended number. New `--no-adjudication` flag plays real games to actual conclusion or a 400-ply (200-move) cap instead of training's throughput-driven shortcut. `eval_games.csv` gets an `end_reason` column via an in-place migration (old rows backfilled `"unknown"`, not guessed) rather than breaking the column count for a file that already has rows in every prior run. `dashboard.py` updated to match. | `72c263a` |

### Where we scoped differently than the letter of a recommendation

- **Bishop/knight bucket conflation** (§ correction to our own earlier work): confirmed and acknowledged in conversation with the project lead; the *fix* for it already happened by construction once `generate_material_probe_positions.py` required "exactly one piece type differs, all else equal" — the pooled-bucket comparison script (`diagnose_probe_construction.py`) that produced the misleading identical numbers is left as-is with its original (now-corrected-in-practice) limitation, rather than rewritten, since it already carries an inline note about what it was actually comparing.
- **`main.py`'s `n_simulations` bound**: chose 1000 as the cap rather than a value tied to any specific existing constant — no strong signal for the "right" number, flagged as a judgement call rather than a derived one.
- We did not attempt to re-verify run16/17/18's checkpoints against the corrected `material_probe` (an open item `material_probe_correction.md` already flagged as unresolved before this assessment) — still open, not addressed by this round of fixes.

## 3. Plan for the remaining design-level recommendations

None of these are implemented. This is the order we intend to tackle
them in, and why.

**1. Automated test suite, first.** Cheapest and highest-leverage of what's
left: most of the verification work already happened informally this
session (isolated reproductions for the replay buffer, MCTS priors, and
the streak state machine; a live functional test for the API; a full
random-vs-random run for the new eval end-reasons). Converting those into
a checked-in `tests/` suite (plain `pytest`, no CI infrastructure planned
yet — a single-developer project doesn't need much more than "run this
before you trust a change") directly protects everything just fixed from
silently regressing, which is a better first move than building new
things on top of possibly-fragile ground. Planned coverage matches the
report's list: perspective/mirroring (`encoder.py`), mate selection and
promotion (`moves.py`, `mcts.py`), history reconstruction, adjudication
(the streak logic), replay persistence, and evaluation fairness.

**2. Tactical benchmark suite, second.** A small, fixed set of *hand-curated*
positions — not sampled from self-play, deliberately, to avoid repeating
the same kind of sample-composition issue the assessment found in
`material_probe`'s positions (drawn from run19's own games, so not a
clean unseen test set). Covering: a hanging piece (is a free capture
taken), mate-in-1 (find it), mate-in-1 defence (prevent it — the
`Qxf7#` pattern the assessment found is exactly this category), and
elementary K+Q/K+R vs K conversion.

Two corrections to that plan from a second review pass, both real gaps
in what was written above, not just style notes:

- **Hand-curated still needs realistic history, or this recreates the
  exact problem it's meant to avoid.** A hand-built FEN with no move
  history behind it is precisely `material_probe`'s original defect —
  encoder.py fills 48 of 55 input planes from history, and HAL has never
  seen a position with real tactics paired with an empty history any more
  than it saw a missing queen that way. Every benchmark position needs an
  actual move sequence reaching it (played out, e.g., from a real or
  synthetic game opening), not just an end-state FEN.
- **Reusing `curate_buffer.py`'s canonical-position generators isn't
  automatically an unseen test set.** Those exact positions (or the same
  generator's output) are already in the training buffer's permanent
  partition — evaluating on them would be measuring fit to training data,
  not generalisation. If we reuse that infrastructure at all, it has to
  generate a *disjoint* set (different squares, different generator seed,
  explicitly diffed against what's actually in the current buffer), not
  the same positions HAL trains on directly.

We'll also validate the position set's own "correct answers" against
Stockfish before trusting it as a benchmark — the `material_probe` lesson
this whole assessment is built on is that an unvalidated test can be
wrong in ways nobody notices for a long time.

**3. Paired LR experiment, third — after the benchmark exists.** The
existing `lr_schedule_design.md` proposal needs revising before it's run,
not just executing as originally drafted: it committed to a specific
backfilled step count that turned out to be inconsistent with the actual
checkpoint record (5,750 vs the real 5,950), and defaulted to a specific
decay schedule presented more confidently than the evidence supports.
Revised plan: two short branches from the *same* checkpoint and buffer,
optimizer state handled identically between them (both reset or both
preserved, not one of each), compared against each other on the frozen
tactical benchmark from step 2 — not against an assumed "correct" target
rate.

**4. Resignation-calibration audit — independent of the above, can run
anytime.** Sample a set of self-play games that hit the resign
threshold, replay them from that point with resignation disabled, and
check whether the predicted loser actually goes on to lose for real. This
stays decoupled from the training loop itself (an offline replay/re-play
script against saved checkpoints, not a change to `train_chess.py`) so it
doesn't need its own paired-branch experiment design the way the LR
question does.

**5. Benchmark manifest — an ongoing habit, not a one-off task.** Once the
tactical benchmark and the LR experiment exist, every comparison run
under either should record checkpoint identity, engine version, search
budget, and seeds as a small structured sidecar file, not just prose in
a report. Simplest to introduce alongside whichever of items 2–3 lands
first, rather than retrofitting it separately.

## 4. Open items this response doesn't close

- Re-verifying run16/17/18's checkpoints against the corrected
  `material_probe` (from `material_probe_correction.md`, still
  unresolved).
- Whether run20's small-piece material drift (documented in
  `run20_drift_investigation.md`) is a real LR-related effect or
  something else — waiting on item 3 above for a properly controlled
  answer rather than another guess.
