# Run20 Drift Investigation — Climbing Policy Loss and Softening Material Judgment

**Project:** chess-ai
**Phase:** 3 — AlphaZero-style chess agent (HAL-4000)
**Authors:** Rob Kirkland, Ellis Ward
**Date:** 7 September 2026 (run20, game 2,274, step 11,250)
**Status:** Investigation report. Diagnosis with a leading hypothesis —
not yet tested or acted on. See `lr_schedule_design.md` for the proposed
response.

---

## 1. Why this investigation

Routine `material_probe.csv` and `loss_breakdown.csv` check-ins around
game 1,800–2,260 turned up two coincident trends worth understanding
before just continuing to watch them:

- `avg_policy_loss` had briefly leveled off (games 1,550–1,800, holding
  at 1.37–1.40) but resumed climbing immediately after, reaching 1.55 by
  game 2,250 — a faster rise than the climb that preceded the plateau.
  `avg_value_loss` stayed flat and healthy throughout (~0.064–0.070),
  so this is specific to the policy head.
- `missing_bishop` and `missing_knight` (the two hardest-to-learn
  `material_probe` categories, see `material_probe_correction.md`) both
  accelerated their drift toward zero in the same window, while
  `missing_two_pawns` recovered slightly and `missing_queen`/`missing_rook`
  continued a slower, steadier drift they'd already been showing for
  longer.

Neither trend is dramatic on its own. Together, in the same window, they
were worth a real investigation rather than another "watch and see."

## 2. Working theory ruled out: MCTS search getting sharper

The initial hypothesis (raised while explaining the original loss split)
was that climbing policy loss might be benign — MCTS visit-count targets
getting sharper/more decisive as search quality improves, making
cross-entropy against that target harder to hit even as the network's
actual move judgement keeps improving.

This doesn't hold up. `snapshots.csv` records the network's own top-move
confidence at 5 fixed canonical positions every 50 games, with no MCTS
noise. For the starting position specifically:

| games | top move | confidence |
|---|---|---|
| 50–500 | e2e3 | 27–38% |
| 1,000–1,400 | e2e3 | 21–29% |
| 1,500–1,800 | e2e3 / h2h4 | 17–28% |
| 1,850–2,250 | e2e3 / h2h4 | **16–23%** |

Confidence has *fallen* over the run, not risen. The other four canonical
positions (`after_e4`, `after_d4`, `italian_game`, `queens_gambit`) show
no clear trend either direction. If the target were getting harder to
match *because it was sharpening*, the network's own distribution
tracking it should be sharpening too, or at least not flattening. It
isn't. This explanation is wrong.

## 3. What corroborates instead: a real, modest narrowing in actual play

`openings.csv` (real self-play games, full search + exploration noise —
a different signal than the noise-free canonical snapshots above) shows
first-move entropy dropping in the same recent window:

| games | entropy (bits) | top move | share |
|---|---|---|---|
| 20–620 | 3.229 | h2h4 | 25.5% |
| 640–1,220 | 3.299 | e2e3 | 24.3% |
| 1,240–1,820 | 3.236 | a2a3 | 23.6% |
| 1,840–2,260 | **3.000** | a2a3 | **32.8%** |

`a2a3` rose from ~10% share early in the run to nearly a third of all
games in the most recent window, and overall entropy dropped to its
lowest point of the run. Real, if modest — this is the training process
itself consolidating around fewer choices, in the same window the other
two symptoms appeared.

`training.csv`'s win/loss balance stayed even throughout (no skewed
windows, draws in the normal 0–4 range), which rules out gross
instability, a corrupted checkpoint, or a crashed/partially-resumed
training process as the explanation.

## 4. Leading hypothesis: no learning-rate schedule, ~17,000 combined steps deep

`chessai/agent.py` sets `LR = 1e-4` for Adam and never changes it —
confirmed no scheduler exists anywhere in `agent.py` or `train_chess.py`.
Every warm start (`warm_start_run19.py`, `warm_start_run20.py`) resets
`agent.steps` to 0 and builds a fresh optimizer at the same fixed LR —
so from the optimizer's point of view, run20 is training at exactly the
same step size run19 started at, with no awareness that this network
lineage is now roughly **17,000 combined steps deep** (run19's 5,750 +
run20's 11,250 so far).

A fixed learning rate held constant this far into training is a known
way to end up in a noisy, non-converging regime rather than settling
into a minimum — the step size stays large enough to keep perturbing
fine distinctions long after it should be taking smaller, more careful
steps. This is consistent with every symptom observed:

- **Policy loss not settling**: gradient noise at a constant step size
  prevents the kind of fine convergence that would let the loss curve
  flatten and stay flat, rather than plateauing briefly and resuming.
- **Bishop/knight drifting faster than queen/rook**: queen and rook
  deficits are large-margin, unambiguous signals — there's a lot of room
  for noise to perturb the network's output before it would actually
  flip sign or shrink meaningfully. Bishop and knight's real signal is
  already small (see `material_probe_correction.md` §4 — Stockfish's own
  mean at |mat|=3 is only -0.37 with std 0.575) and sits much closer to
  the noise floor, so the same amount of gradient noise erodes it faster.
- **Modest narrowing in actual play**: consistent with the policy
  settling toward whichever basin of attraction noisy updates happen to
  reinforce, rather than confidently converging on a considered best
  choice.

## 5. What this is not — limits of this investigation

This is a diagnosis by elimination and mechanism, not a proven cause.
Specifically not done:

- **No controlled test.** The LR-schedule hypothesis hasn't been tried
  against a held-out comparison — there's no run20-with-schedule to
  compare against run20-without-schedule at the same step count.
- **No alternative causes fully excluded.** Data pipeline drift (has the
  self-play population's own average strength/style shifted enough to
  change what `curate_buffer.py`'s rolling buffer contains?), or a subtler
  interaction between the Stockfish-relabelled permanent partition and
  an increasingly large rolling buffer, haven't been checked directly.
- **Magnitude is still small.** None of the material_probe categories
  have moved outside a range that would itself be alarming in isolation
  (queen is still -0.60, nowhere near the near-zero readings the old
  broken probe used to produce). This is a trend worth acting on before
  it compounds further, not an emergency.

## 6. Next step

See `lr_schedule_design.md` for the proposed response (an LR decay
schedule) and the design questions that need deciding before
implementing it — in particular, how to track cumulative steps across
warm-started runs, since `agent.steps` resetting to 0 at each warm start
is exactly the gap that let this go unaddressed for two runs running.

## References

- `logs/run20/material_probe.csv`, `loss_breakdown.csv`, `training.csv`,
  `snapshots.csv`, `openings.csv` — raw data underlying every table above
- `chessai/agent.py` — `LR = 1e-4`, no scheduler
- `material_probe_correction.md` §4 — Stockfish's own signal strength by
  material magnitude, cited above as the basis for "bishop/knight are
  closer to the noise floor than queen/rook"
