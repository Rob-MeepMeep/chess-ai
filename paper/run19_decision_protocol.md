# Run 19 — Pre-Registered Decision Protocol

**Date:** 28 August 2026 (registered at game 0, before any run19 data exists)
**Authors:** Rob Kirkland, Ellis Ward

## Purpose

Run19 escalates generalisation-gap Option C's blend weight. Run18's final
result (64 `material_probe.csv` readings, three ~21-reading windows,
1,280 games): `missing_queen` **100% correct in every window**,
independently cross-verified by `regression.csv` to four decimal places —
the strongest, most stable result of the entire arc. `missing_rook` held
76–95% correct throughout. But `missing_bishop`, `missing_knight`, and
`missing_two_pawns` declined across the same three windows, ending below a
coin flip — the identical trajectory rung 1b failed with.

Diagnosis: run18's blend (α = 0.2 self-play / 0.8 Stockfish) likely gave
Stockfish's judgement a loud enough voice at queen/rook scale — where its
evaluation is often forcing (a mate line the engine can see at depth 16),
producing a target similar in character to self-play's ±1.0 outcomes — but
not at smaller magnitudes, where Stockfish returns a softer centipawn-scale
evaluation that's plausibly getting outcompeted by the dominant all-or-
nothing pattern surrounding it in every batch.

## Method

No new position sampling, no new Stockfish evaluation — the fix and the
escalation are decoupled by design (see commit `a6d09b2`).
`relabel_with_stockfish.py` was corrected to save raw
(self-play outcome, Stockfish evaluation) pairs instead of a pre-blended
value, so this test only changes `curate_buffer.py`'s
`STOCKFISH_ALPHA`: **0.2 → 0.05** (95% Stockfish trust, versus 80% in
run18). A small self-play floor is kept rather than full replacement (0.0)
— a safety net against an occasional misleading Stockfish read, not a
meaningful dilution of the test.

Same ~9,000-position set as run18 (now also including run18's own games
as a third source when regenerated, for more diversity — a minor,
disclosed addition, not the variable under test). No architecture change.
`warm_start_run19.py` performs a straight weight copy from run18, fresh
optimizer and step count.

**How much this steps away from pure self-play, quantified (updated)**:
same footprint as run18 — the Stockfish-relabelled set is still roughly a
quarter to a third of a typical training batch, two-thirds to
three-quarters remains pure self-play. What changes is the strength of
the signal *within* that already-affected slice, not how much of the
batch it touches.

## Expected effects and outcomes

**Core expectation**: `missing_bishop`, `missing_knight`, and
`missing_two_pawns` should show real, sustained improvement this time,
holding across multiple windows rather than the early-promise-then-decline
pattern both prior attempts (rung 1b, run18 at α=0.2) showed.

**Should not regress**: `missing_queen` and `missing_rook` — both strong
and stable in run18 — should hold or strengthen further, since the only
change increases Stockfish's influence, not decreases it.

**What a null result would mean**: if the small-piece positions are still
flat or declining after a comparable window to run18's (~1,000–1,300
games), that would mean the problem is not primarily about blend strength
either. At that point the remaining, more expensive options are Option B
(full synthetic pretraining) or accepting the gap as a bounded limitation
and moving to Phase 4 with the current capability set — not further alpha
tuning, which would have now been tried at two meaningfully different
settings (0.2, 0.05) without success.

## Decision on outcome

- **GREEN**: sustained improvement in the small-piece positions within
  ~1,000 games, holding across multiple windows.
- **RED**: flat or declining over a comparable window to run18's. Escalate
  no further on this specific lever — move to Option B or Phase 4.

## Status at registration (game 0)

Registered before `warm_start_run19.py` has been run or any run19 game has
been played. No data yet.

## Resolution (3 September 2026, game ~1190, step 5,950)

Neither GREEN nor RED, as written. Through game 1190, `missing_bishop`/
`missing_knight`/`missing_two_pawns` were reading flat-to-declining
against `material_probe.csv` — trending toward the RED criterion above.
Investigating why turned up a construction bug in the probe itself, not a
training failure: `MATERIAL_PROBE_POSITIONS` (`chessai/logger.py`) tested
the starting position with one piece deleted, zero moves played, empty
history — a shape that cannot occur in a legal game once material has
changed, and one the network had never been trained on. On the same
checkpoint, real in-context positions at the identical magnitudes read a
clean, correctly-signed -0.30 to -0.37, consistent with a smooth,
monotonic material-sensitivity curve across the entire ±1 to ±20+ range.

Full diagnosis, evidence, and disposition: `material_probe_correction.md`.

This run is marked **resolved, not RED**. The data that would have
supported either verdict was measuring the wrong thing throughout, so the
pre-registered decision can't be made from it either way. No further
alpha tuning is planned as a result of this run specifically — not
because escalating α was vindicated, but because the flat/declining
signal that would have justified continuing to escalate it was never
real. `run20` continues from this checkpoint with the corrected probe
in place; see `material_probe_correction.md` §7 for what that changes.
