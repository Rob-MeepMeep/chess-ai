# Run 14 — Pre-Registered Decision Protocol

**Date:** 17 July 2026 (registered before the game-1,500 eval fired; run at ~game 630 at time of writing)
**Authors:** Rob Kirkland, Ellis Ward

## Purpose

Run 14 tests one hypothesis: **run 13's learning stall was caused by corrupted
value labels** — mislabeled canonical/generated endgame positions in the
oversampled permanent buffer partition, and a regression metric built on an
illegal position and a terminal position that could not measure generalisation.

Run 14 is a from-scratch run with label-safe seed data, corrected regression
positions, and the lockstep self-play loop. This document fixes the pass/fail
criteria **in advance**, so results are read against pre-committed thresholds
rather than post-hoc judgement. Nothing is changed between gates; if a red line
fires, interventions are applied one at a time.

## Baseline (run13_retune, 1,330 games)

- Eval vs random: 26% wins / 65% move-cap games at first eval (steps 1,915);
  24% / 66% at steps 5,185. No improvement across 3,270 further steps.
- `missing_queen` regression: never left the ±0.1 noise band in 1,200 games.
- Cap-draw share: pinned at ~47–60% throughout.
- Throughput: 29–33 games/h (single-game loop, 600 sims).

## Gate 1 — game-800 regression row

| Signal | Call |
|---|---|
| `missing_queen` ≤ −0.15 | GREEN — label fix confirmed ahead of schedule |
| `missing_queen` inside ±0.1 | NEUTRAL — expected; lagging indicator; no action |
| Cap share < 45% in recent 200-game window | GREEN (secondary) — early conversion signal |

No fail condition exists at this gate.

## Gate 2 — game-1,500 eval (PRIMARY GATE)

Eval vs random, n = 100 games (50 each colour), fired automatically by
eval_watcher.

| Result | Call |
|---|---|
| ≥ 40% wins, or ≤ 50% caps | **PASS** — hands off, run to 3,000 |
| 30–40% wins | AMBIGUOUS — indistinguishable from baseline + noise at n=100; no intervention; wait for Gate 3 |
| ≤ 30% wins AND ≥ 60% caps AND `missing_queen` inside ±0.1 | **FAIL** — labels were not the binding constraint; begin intervention ladder |

Intervention ladder (one change at a time, in order):
1. **Cap-draw early adjudication** — end games where \|material\| ≥ 8 is
   sustained past move 60, scored as a win. Fixes the label semantics
   (currently 55%+ of games teach "be ahead at move 150 = 0.8 win", rewarding
   shuffling over converting) and refunds compute spent on dead plies.
2. **Anneal `perm_ratio`** 0.33 → 0.10 once self-play data is plentiful —
   the permanent partition is loud enough to be memorised as a shortcut.
3. (Last resort, admission of defeat on generalisation) material-count input
   plane — to be documented as such if ever used.

## Gate 3 — game-3,000 second eval (HARD STOP)

- GREEN: clear trend across the two evals — win rate up ≥ 10 points
  eval-to-eval, or cap share clearly falling.
- RED: no separation from baseline on any conversion metric (win rate, cap
  share, stalemates, `missing_queen`) → **stop the run and redesign**. No
  extensions.

## Standing red lines (any time)

- Loss climbing past ~2.5 and still rising after game 1,000
  (the ~1.8 plateau should break downward)
- Colour split worse than 65/35 sustained over 200+ games (colour collapse —
  a known historical failure mode of this project)
- Stalemate count climbing block-on-block (won positions being systematically
  butchered)
- Throughput < 20 games/h without pauses; any crash or checkpoint corruption

## Standing green lines (any time)

- `missing_queen` < −0.15 on two consecutive regression rows
- Cap share < 45% over any 200-game window
- Quick mates (≤ 30 plies) becoming rarer while total mate share holds —
  evidence the Fool's-Mate-family patterns are being learned defensively
  (they are currently being both dealt and suffered by both colours:
  games 419, 427, 585, 622)

## Interpretation notes

- `missing_queen` expectation: correct answer is deeply negative
  (roughly −0.5 to −0.9 at maturity; value ≈ expected score, and queen-odds
  loses ~always). −0.02 is not "nearly right" — it is the network claiming a
  missing queen barely matters, and it matches the network's noise on the
  balanced start position. **Magnitude, not sign, is the test.** It is also a
  held-out full-board position with no near neighbours in the seed data, so it
  is expected to move late — hence it can confirm success early but cannot
  declare failure until Gate 3.
- `value_resign` frequency was provisionally treated as an early positive
  (~4–5% of games 300–350 vs baseline ~1–2%) but went quiet in games 477–629;
  withdrawn as an indicator until it re-establishes.

## Status at registration (game ~630)

Loss plateaued at ~1.8 (on the predicted trajectory: fall → rise → crest).
Colour balanced. Throughput 26–28 games/h. Cap share ~55–57% — Gate 1's
secondary green is not currently on track; primary verdict deferred to Gate 2
as designed. Regression rows so far: +0.03 / −0.05 / −0.02 at games
200/400/600 — inside the noise band, matching baseline behaviour, no verdict.
