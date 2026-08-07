# Run 15 — Pre-Registered Decision Protocol

**Date:** 7 August 2026 (registered at game 1 — before any meaningful data exists)
**Authors:** Rob Kirkland, Ellis Ward

## Purpose

Run 15 tests intervention ladder rung 1 from `run14_decision_protocol.md`:
**early material adjudication**. Run 14's Gate 3 hard stop showed cap-draw
share rising from 62% to 77% between its two evals while win rate fell and
`missing_queen` stayed flat — the move-150 cap was teaching "hold a lead to
move 150 = 0.8 win," rewarding shuffling over converting, and burning
compute on plies after the game was already decided.

Run 15 adds a fourth game-ending condition to `train_chess.py`: if
`|material| >= 8` holds for 6 consecutive plies past move 60, the game ends
immediately as a 0.9-confidence win for the side ahead, logged as
`end_reason = "material_adjudication"`. Everything else — network
architecture, lockstep self-play, 600 simulations, label-safe seed data — is
unchanged from run14. One variable changes at a time.

This document fixes the pass/fail criteria **in advance**, exactly as
run14's protocol did, and reuses run14's exact game milestones (800 / 1,500
/ 3,000) rather than picking new ones. This makes every gate a direct,
matched-checkpoint comparison against run14's numbers — the strongest
evidence available for whether rung 1 actually works.

## Baseline — run14 (this run's comparison point throughout)

| Checkpoint | Win rate vs random | Cap-draw share | `missing_queen` |
|---|---|---|---|
| Gate 2 (game 1,500) | 30.0% | 62.0% | ~0.00 (nearest row) |
| Gate 3 (game ~3,278) | 23.1% | 76.9% | −0.009 |

`missing_queen` across all sixteen run14 regression rows never exceeded
±0.058 — the empirical noise ceiling this run measures against. Zero losses
to random at Gate 3 despite the falling win rate: HAL was not losing more,
it was converting less and running out the clock more.

## Gate 1 — game 800 (mechanical sanity check, no fail condition)

This gate is not about quality — it is about whether the mechanism fires at
all.

| Signal | Call |
|---|---|
| `material_adjudication` share is meaningfully > 0% of games | GREEN — the mechanism is engaging |
| `material_adjudication` share ≈ 0% | NEUTRAL — early network may not yet be building sustained 8+ leads; re-examine thresholds only if this persists past Gate 2 |
| Cap-draw share below run14's game-800 level (~53%) | GREEN (secondary) — early conversion signal |
| Average game length below run14's ~118-move average | GREEN (secondary) — compute-refund signal |

No fail condition exists at this gate.

## Gate 2 — game 1,500 (PRIMARY GATE)

Eval vs random, n = 100 games (50 each colour), fired automatically by
eval_watcher. Compared directly against run14's Gate 2: 30.0% wins / 62.0%
caps.

| Result | Call |
|---|---|
| Win rate clearly above 30%, OR cap share clearly below 62% | **PASS** — rung 1 is doing something; run to Gate 3 |
| Roughly matching run14 (within noise of both numbers) | AMBIGUOUS — no intervention; wait for Gate 3 |
| Worse than run14 on both metrics, OR `material_adjudication` share is negligible | **FAIL** — either the mechanism isn't engaging or it's actively hurting; proceed to intervention ladder rung 2 |

## Gate 3 — game ~3,000 (HARD STOP)

Second eval, n = 100. Compared directly against run14's Gate 3: 23.1% wins
/ 76.9% caps / `missing_queen` flat.

- **GREEN**: cap share or win rate meaningfully better than run14 at the
  matched checkpoint. Bonus green: `missing_queen` reads beyond ±0.06
  (run14's empirical ceiling) on two consecutive rows — the first
  generalisation signal this project has produced across two full runs.
- **RED**: no improvement over run14 on cap share or win rate at matched
  checkpoints → rung 1 is exonerated as *not* the fix. Proceed to rung 2
  (anneal `perm_ratio` 0.33 → 0.10). No extensions on this run.

## Standing metrics (watched continuously, not just at gates)

- **`material_adjudication` share and average game length** — the direct
  fingerprint of whether the mechanism is engaging, independent of whether
  it's helping. Visible every `PRINT_EVERY` line and in `end_reasons.csv`.
- **Losses-to-random pattern** — does the zero-losses-but-falling-win-rate
  pattern from run14's Gate 3 persist, worsen, or break? A win-rate
  improvement that comes with losses reappearing would need a different
  read than one that comes with caps simply converting to wins.

## Standing red lines (carried over from run14, unchanged)

- Loss climbing past ~2.5 and still rising after game 1,000
- Colour split worse than 65/35 sustained over 200+ games
- Stalemate count climbing block-on-block
- Throughput < 20 games/h without pauses; any crash or checkpoint corruption

## Standing green lines (carried over from run14, unchanged)

- `missing_queen` < −0.15 on two consecutive regression rows
- Cap share < 45% over any 200-game window
- Quick mates (≤ 30 plies) becoming rarer while total mate share holds

## Known confound

Run 15's seed buffer is built from run14's own self-play games (curated,
decisive, ≤100 moves) rather than run13_retune's — a larger and
label-cleaner source, but a different one. Early-game behaviour may differ
from run14's for reasons unrelated to the material-adjudication mechanism
itself. Not correctable; noted so it isn't mistaken for a mechanism effect
if early numbers look unusual.

## Status at registration (game 1)

Registered before any meaningful data exists — game 1 was a 7-ply Fool's
Mate, expected behaviour for fresh, noise-injected weights and uninformative
at this scale. No status to report yet.
