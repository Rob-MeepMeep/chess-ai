# Run 18 — Pre-Registered Decision Protocol

**Date:** 27 August 2026 (registered at game 0, before any run18 data exists)
**Authors:** Rob Kirkland, Ellis Ward

## Purpose

Run18 tests **generalisation-gap Option C**: self-play positions with a
material imbalance of 1–7 relabelled with real Stockfish evaluations,
folded into the permanent training partition. Chosen after both self-play
adjudication attempts failed:

- **Rung 1** (run15): fixed conversion decisively but never touched
  generalisation — `missing_queen` stayed flat across the whole run.
- **Rung 1b** (run17): a second, lower-confidence adjudication tier for
  the 3–7 band. Final result (game ~1,780, three ~30-reading windows):
  `missing_bishop`/`missing_knight` declined from modest early promise
  (43%/27% correct sign) to essentially never correct (3%) — worse in
  relative terms than run16's terminal state. `missing_two_pawns` and
  `black_missing_rook` showed the same steady decline.

Diagnosis: material adjudication only ever fires above a threshold —
self-play structurally cannot generate reinforced training examples below
it, no matter how the threshold or streak length is tuned. Option C
sidesteps the threshold entirely by directly constructing labelled
examples via Stockfish, rather than hoping self-play produces them.

## Method

`relabel_with_stockfish.py` sampled positions from run16 and run17's
self-play (4,197 candidate games), targeting `|material|` in [1, 7] past
move 20, capped at 4 per game, plus a smaller sample of near-balanced
positions. Each was evaluated by Stockfish at depth 16 and blended with
the position's own self-play outcome at **α = 0.2 self-play / 0.8
Stockfish** — deliberately favouring Stockfish strongly, since rung 1b's
failure was diagnosed as a *coverage* problem (not enough labelled
examples existing at all), not evidence that a weak signal is insufficient
once examples exist. Target size ~9,000 positions, chosen to be large and
diverse enough to avoid the run13-style small-permanent-partition
memorisation trap.

The relabelled set is added to the existing permanent partition via
`curate_buffer.py`, the same mechanism that already successfully teaches
the canonical K+Q vs K endgames — repeated exposure, not one-time.

**How much this steps away from pure self-play, quantified**: the
positions HAL learns from remain 100% self-play-generated — nothing
invented or externally sourced. Only the *value label* for a slice of them
changes. The rolling partition (~67%+ of every training batch) is
untouched. Within the ~33% permanent partition, the Stockfish-relabelled
set is now the dominant component by volume (~9,000 vs ~750 canonical),
so roughly a quarter to a third of a typical training batch carries
Stockfish influence — two-thirds to three-quarters remains pure self-play
outcome data, same as every prior run.

No architecture change — same 55-plane encoder and network as run16/17.
`warm_start_run18.py` performs a straight weight copy from run17, fresh
optimizer and step count.

## Expected effects and outcomes

**Core expectation**: `missing_bishop`, `missing_knight`, and
`missing_two_pawns` should show real, sustained improvement — not the
early-promise-then-reversal pattern both prior attempts showed, since this
mechanism doesn't depend on self-play's incidental frequency at that
magnitude the way rungs 1 and 1b did.

**Should not regress**: `missing_queen` and `missing_rook` — genuinely
strong and stable since run16 — should hold, since nothing about their
training signal changes.

**What a null result would mean**: if the small-piece positions are still
flat or declining after a comparable window to run17's (~1,500–2,000
games), that would suggest the problem is deeper than a labelling/coverage
issue — worth re-examining whether α=0.2 was still too diluted, whether
9,000 positions is enough diversity, or whether the value head's capacity
itself is the limit at this network size. Escalating α toward full
replacement is a cheap same-day follow-up (no need to re-run Stockfish),
so that would be the first thing tried before concluding the mechanism
itself has failed.

## Decision on outcome

- **GREEN**: sustained improvement in the small-piece positions within
  ~1,000–2,000 games, holding (not reversing) across multiple windows —
  the standard that let run16 and run17's results be called with
  confidence. Confirms Option C closes the gap self-play alone couldn't.
- **RED**: flat or declining over a comparable window. Escalate α first
  (cheap); if that doesn't help either, this is the point to reconsider
  Option B or accept the gap as a hard limit of this architecture/scale.

## Status at registration (game 0)

Registered before `warm_start_run18.py` has been run or any run18 game has
been played. No data yet.
