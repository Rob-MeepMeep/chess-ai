# Learning-Rate Schedule — Design Proposal

**Project:** chess-ai
**Phase:** 3 — AlphaZero-style chess agent (HAL-4000)
**Authors:** Rob Kirkland, Ellis Ward
**Date:** 7 September 2026
**Status:** Design proposal — not implemented. Responds to
`run20_drift_investigation.md`. Needs a decision on scope and parameters
before any code changes.

---

## 1. Problem, restated

`chessai/agent.py` trains at a fixed `LR = 1e-4` for the network's entire
life. Every warm start (`warm_start_run19.py`, `warm_start_run20.py`)
resets `agent.steps` to 0 and builds a fresh Adam optimizer at that same
fixed rate — so the optimizer has no memory of how deep into training
this network lineage actually is. It's currently ~17,000 combined steps
deep (run19: 5,750 + run20: 11,250 so far) and still taking exactly the
same size step it took at step 0.

`run20_drift_investigation.md` lays out the leading hypothesis: this is
producing a noisy, non-converging regime rather than a settled one —
policy loss climbing instead of flattening, and the network's weakest,
smallest-margin value judgements (bishop, knight) eroding faster than its
strongest ones (queen, rook), consistent with gradient noise perturbing
fine distinctions before they can fully settle.

**Goal of this document**: decide whether to introduce an LR schedule,
and if so, which kind, anchored to what, with what parameters — not to
implement it unilaterally.

## 2. The design problem specific to this project: warm starts reset step count

A standard LR schedule (step decay, cosine, etc.) is normally driven by
the optimizer's own step counter. Here that counter is deliberately reset
to 0 on every warm start (`new.steps = 0` in every `warm_start_runN.py`,
by design — see their docstrings — "run21's own step count, not run20's").
That's correct and worth keeping for `training.csv`'s per-run accounting.
But it means a schedule driven by `agent.steps` would restart from
"early training" every single time a new run begins, defeating the whole
point — it would never reflect the ~17,000 real steps of accumulated
training this network has actually had.

**This means any schedule needs its own counter, separate from `steps`,
that warm starts carry forward instead of resetting.** Proposed:
`agent.lifetime_steps` — incremented every `train()` call like `steps`
is, but copied forward (not reset) by warm-start scripts. Existing run19/
run20 checkpoints don't have this field; introducing it means backfilling
it once with the known true count (17,000 as of this writing) rather than
letting it silently start over at 0, which would just reproduce the exact
gap that let this go unnoticed for two runs.

## 3. Options for the decay function

| Option | How it works | Pros | Cons |
|---|---|---|---|
| **A. Exponential decay** | `lr = LR_BASE * rate^(lifetime_steps / decay_every)` | Smooth, no discontinuities; no need to commit to a total training horizon in advance — fits an open-ended, run-by-run project | Two arbitrary constants (`rate`, `decay_every`) to choose without much prior data to tune them against |
| **B. Step decay** | Halve (or ×0.1) LR every fixed number of lifetime steps | Simplest to reason about and log | Visible kinks in the loss curve right after each drop; same arbitrary-constant problem as A, just discretised |
| **C. Cosine annealing** | Smooth decay from `LR_BASE` to ~0 over a fixed horizon `T` | Well-established, smooth, self-documenting decay curve | Requires committing to a total step horizon `T` up front — awkward here, since we don't know how many more runs this project will have. Setting `T` too low forces LR to ~0 prematurely; too high and it barely decays at all, no better than not scheduling |
| **D. Reduce-on-plateau** | Monitor a metric (e.g. rolling `avg_value_loss`, or the material_probe categories' windowed mean) and cut LR when it stops improving | Adaptive — doesn't require guessing a horizon or arbitrary decay constants; directly responsive to the actual symptom | More moving parts: needs a "has it plateaued" rule (window size, tolerance) that itself needs tuning; risk of reacting to short-term noise materials_probe already shows checkpoint-to-checkpoint |
| **E. One-time manual drop** | Just lower `LR` to a smaller fixed constant for future runs, no scheduler code at all | Trivial to implement, zero new failure modes | Doesn't solve the actual problem — the network will eventually be just as deep past *that* fixed rate as it is now past the current one. Kicks the can down the road exactly one run |

## 4. Recommendation

**Option A (exponential decay on `lifetime_steps`)**, with a concrete
starting proposal to react to rather than a final answer:

```
LR_BASE     = 1e-4
DECAY_RATE  = 0.5
DECAY_EVERY = 5,000 lifetime steps
effective_lr = LR_BASE * DECAY_RATE ** (lifetime_steps / DECAY_EVERY)
```

At the current lifetime count (~17,000), this already gives
`1e-4 * 0.5^3.4 ≈ 9.5e-6` — roughly a 10x reduction from where training
sits today. That's a substantial jump if backfilled immediately, which is
exactly the point: the network is already 17,000 steps past where a
schedule would have started decaying it, so catching up to where it
*should* be is supposed to look like a big one-time correction, not a
gentle nudge.

Why A over the others: C needs a horizon this project doesn't have yet
(runs are added incrementally as decisions get made, not planned as a
fixed total step count from the start). D is the most principled
long-term answer but adds real complexity and a new set of tunable
thresholds before we even know if the core hypothesis is right. B is a
discretised, kinkier version of A with the same tuning problem and no
upside. E doesn't address the actual mechanism.

This is a starting proposal, not a fixed prescription — `DECAY_RATE` and
`DECAY_EVERY` are both real judgement calls with no strong data to derive
them from yet. Worth treating the first run under this schedule as a
calibration run rather than assuming these constants are right.

## 5. Implementation sketch (not yet written)

- `chessai/agent.py`: add `self.lifetime_steps = 0` in `__init__`;
  compute `effective_lr` each `train()` call from the formula above and
  set `self.optimizer.param_groups[0]['lr'] = effective_lr` before the
  optimizer step; increment `lifetime_steps` alongside `steps`.
- `save()` / `load()`: persist and restore `lifetime_steps` like `steps`
  already is; `load()` should default missing `lifetime_steps` to `0` for
  old checkpoints (with a loud print, not a silent default, so a missing
  backfill is obvious rather than quietly wrong).
- Whatever `warm_start_run21.py` looks like: explicitly carry
  `lifetime_steps` forward from the old checkpoint (`new.lifetime_steps
  = old_ckpt.get("lifetime_steps", <manually backfilled true count>)`),
  the same way it already carries network weights forward but *not*
  `steps`.
- `chessai/logger.py` / `training.csv`: log the effective LR each window,
  same reasoning as the loss-breakdown split — if this changes anything,
  we want it visible, not inferred after the fact.

## 6. Testing plan — don't modify run20 mid-flight

Same discipline as the `material_probe` fix and run20 itself: **run this
as a new run (run21), warm-started from run20's current checkpoint**,
rather than changing run20's optimizer mid-run. That keeps run20 as a
clean, unconfounded baseline and makes the comparison honest — if
`missing_bishop`/`missing_knight`'s drift slows or reverses under run21's
schedule relative to where run20's continued at the same step count, and
policy loss actually flattens, that's real evidence for the hypothesis.
If neither happens, the LR explanation is wrong and `run20_drift_
investigation.md` §5's other candidates (data pipeline drift in
particular) need checking instead.

## 7. Decision needed

Before any code gets written:

1. **Proceed with an LR schedule experiment at all**, or let run20
   continue unscheduled a while longer to see if the drift keeps
   accelerating, plateaus, or reverses on its own first?
2. **If proceeding**: accept Option A (exponential decay) and the
   starting constants in §4, or adjust them?
3. **Backfill `lifetime_steps` at 17,000** (the honest, known total) when
   introducing it, rather than starting the counter at 0 — confirm this
   is wanted, since it means the very first run under the new schedule
   starts already 3.4 decay-periods in, i.e. at a ~10x reduced LR from
   day one rather than easing into it.
