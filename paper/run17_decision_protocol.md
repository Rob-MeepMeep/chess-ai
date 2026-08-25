# Run 17 — Pre-Registered Decision Protocol

**Date:** 22 August 2026 (registered at game 0, before any run17 data exists)
**Authors:** Rob Kirkland, Ellis Ward

## Purpose

Run17 tests intervention ladder **rung 1b**: a second, lower-confidence
material adjudication tier alongside run16's existing one. Run16 tested
generalisation-gap Option A (a material-count input plane) and produced a
genuinely mixed, statistically clear result — not ambiguous noise, a real
split:

| Position (game 60→2120, four time windows) | Trend |
|---|---|
| `missing_rook` | improved sharply, then partially backslid (43%→77%→77%→63% correct sign) |
| `black_missing_queen` | robust, steadily growing win (83%→96% correct sign) |
| `black_missing_rook` | genuinely recovering (0%→56% correct sign) |
| `missing_bishop` | **monotonically worse** (26%→0% correct sign) |
| `missing_knight` | **monotonically worse** (26%→0% correct sign) |
| `missing_two_pawns` | **monotonically worse** (39%→7% correct sign) |

Full detail: `run16_decision_protocol.md`. The queen/rook-scale positions
genuinely learned something; the small-piece positions didn't just fail to
improve, they got *worse* across four consecutive windows — ruling out
"just needs more time" as an explanation. The diagnosed cause:
`train_chess.py`'s material adjudication only fires at `|material| ≥ 8`
(roughly queen-or-more). Self-play never produces reinforced training
signal for smaller imbalances, no matter how many games are played,
because the adjudication rule structurally filters them out before they
can be labelled decisively.

Rung 1b tests whether the fix is that specific and that cheap: add a
second tier that adjudicates the 3–7 band too, held for a longer streak as
extra proof, at lower confidence. This is deliberately the smallest-cost,
smallest-compromise fix available — still pure self-play, no external
evaluator — before reaching for Option B or C's Stockfish-based
approaches.

## Method / changes made

- `train_chess.py`: new `MATERIAL_ADJUDICATE_MODERATE_*` constants —
  `LOW=3`, `HIGH=8` (exclusive — the existing strong tier owns ≥8),
  `STREAK=16` plies (vs the strong tier's 6), `SCALE=0.7` confidence (vs
  0.9). Checked in priority *after* the strong tier, so material that
  escalates from moderate to strong correctly falls through to the
  stronger, faster-firing rule rather than getting stuck on the weaker one.
- `eval_chess.py`: identical logic mirrored in `play_game()` — learned the
  hard way from Gate 2's original bug that eval must adjudicate identically
  to training, or its win-rate numbers silently undercount again.
- `curate_buffer.py`: seed buffer now built from run16's games (not
  run15's); `MIN_GAME` dropped 300→20, since run16 was warm-started and
  played competently from game 1 — no genuine warmup period to skip this
  time. `GOOD_REASONS` gains `material_adjudication_moderate`.
- `warm_start_run17.py` (new): simpler than run16's — no architecture
  change this time, so it's a straight network-weight copy with a fresh
  optimizer and step count, not weight surgery.
- Verified in isolation before shipping: six scenarios (sustained
  moderate, sustained strong, escalating mid-streak, oscillating between
  bands, sub-threshold, all-before-move-60) all behaved exactly as
  designed — see commit `87ae5b1`.

## Expected effects and outcomes

**Core expectation**: `missing_bishop`, `missing_knight`, and
`missing_two_pawns` should stop getting worse and start showing real
positive movement (crossing back above 20–30% correct sign within a few
hundred games, trending toward run16's better-performing positions'
trajectory) — because they'll finally have a training-data source that
labels their exact material range.

**Should not regress**: the already-improving positions (`missing_rook`,
`black_missing_queen`, `black_missing_rook`) should continue their run16
trajectory largely undisturbed, since the strong tier (≥8) is completely
unchanged and still takes priority whenever material actually reaches that
range.

**What a null result would mean**: if the small-piece positions are
*still* getting worse, or still flat, after a comparable number of games
to run16's diagnostic window (~2,000), that would be a stronger and more
specific finding than run16's — it would mean the problem isn't really
about *which* imbalances get labelled at all, and self-play-only fixes
are exhausted. That result would be the clearest case yet for moving to
Option C (Stockfish-relabelled positions), which doesn't depend on any
adjudication threshold to generate signal for small material differences.

**Not the target of this test**: beating Stockfish depth 1. Same caveat
as run16 — this targets a specific, diagnosed gap in material
generalisation, not the broader positional judgement a sound opponent
requires. Movement here would be a genuine win on its own terms even if
Stockfish depth 1 stays at 0/50.

## Decision on outcome

- **GREEN**: small-piece positions show real, sustained improvement
  (rising `frac_correct_sign`, not just a brief excursion) within
  ~1,000–2,000 games, without meaningfully disturbing the already-good
  large-piece positions. Confirms the adjudication-threshold hypothesis
  and that pure self-play can close this specific gap cheaply.
- **RED**: small-piece positions remain flat or continue worsening over a
  comparable window to run16's. Confirms self-play-only fixes are
  exhausted for this problem; proceed to Option C.

## Status at registration (game 0)

Registered before `warm_start_run17.py` has been run or any run17 game has
been played. No data yet.
