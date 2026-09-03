# Closing the Reinforcement Learning Arc — HAL-4000, Runs 13–18

**Project:** chess-ai
**Phase:** 3 — AlphaZero-style chess agent (HAL-4000)
**Authors:** Rob Kirkland, Ellis Ward
**Date:** 28 August 2026
**Status:** Closing report. Marks the end of the initial self-play
reinforcement learning arc — not the end of the project.

> **Addendum, 3 September 2026:** the "small-piece positions still
> failing" verdicts below for run16/17/18, and this document's framing of
> the whole arc as chasing that gap, were measured with a `material_probe`
> that turned out to be structurally broken (synthetic zero-history test
> positions, never seen in training). The current checkpoint (run19,
> step 5,950) already handles bishop/knight/two-pawn deficits correctly
> once measured on real positions instead. Full diagnosis:
> `material_probe_correction.md`. The rows below are left as written —
> they're an accurate record of what the probe reported and how the
> project responded at the time — but should be read as **probe
> readings, not confirmed capability**, unless independently
> re-verified against a given run's actual checkpoint.

---

## 1. Why this document exists

Six runs, 17,010 self-play games, and roughly 408 active hours (17 days)
of continuous GPU training sit behind this point. The arc that started with
a corrupted-label bug report has run its natural course: the cheap fixes
have been tried, the diagnosis has gotten sharper with every run, and the
remaining gap now requires either accepting a real philosophical cost
(leaning further on Stockfish) or a diminishing-returns self-play tweak
whose odds we can already reason are poor. That's the right moment to stop,
write down what actually happened end to end, and treat the next phase as
a deliberate choice rather than a foregone continuation.

This report is the single reference for the whole arc — the individual
run protocols (`run14_decision_protocol.md` through
`run18_decision_protocol.md`) remain the detailed record of each step;
this pulls the throughline out of them.

## 2. The arc at a glance

| Run | Games | Active hours | What changed | Verdict |
|---|---|---|---|---|
| run13_retune | 1,330 | 47.0h | (baseline — pre-overhaul) | Stalled: 26%→24% win rate, no improvement over 3,270 steps |
| run14 | 3,279 | 119.0h | Fixed corrupted labels + illegal regression FENs | Gate 3 RED — every conversion metric moved the wrong way |
| run15 | 6,840 | 153.3h | Rung 1: early material adjudication | Gate 3 GREEN — 100% wins vs random, cap share collapsed |
| run16 | 2,493 | 42.6h | Option A: material-count input plane | Partial — large material generalised, small material got worse |
| run17 | 1,788 | 26.7h | Rung 1b: second adjudication tier (3–7 band) | RED — small-piece positions declined further, self-play-only exhausted |
| run18 | 1,280 | 19.6h | Option C: Stockfish-relabelled positions | Partial — queen/rook solved decisively, bishop/knight/pawns still fail |

**Total: 17,010 games, ~408 active hours (17 days), across six runs.**

## 3. The throughline, in five findings

### 3.1 A label bug can look like a training failure for a long time

Run13_retune's stall (26% wins vs random, unmoving for 3,270 gradient
steps) looked like a fundamental limitation until a code review found
the actual cause: illegal and capturable canonical FENs in the permanent
training partition, plus a regression metric built on an illegal position
and an already-checkmated one. The fix (run14) was necessary and correct
— but it turned out not to be sufficient, which is itself a lesson: fixing
a real bug doesn't guarantee it was the binding constraint.

### 3.2 Conversion and generalisation are different capabilities, and self-play can teach one without the other

Rung 1 (run15) — ending games early once a large material lead is
sustained, rather than grinding to a soft-scored move-cap draw — was the
single biggest win of the entire arc. Win rate vs random went from
run14's 23% to a perfect 100%, and it has **held at that ceiling through
every subsequent run**, run16 through run18, untouched by anything since.

But it never moved `missing_queen` — the value head's ability to
recognise material advantage in an unfamiliar position shape. That metric
sat inside a ±0.06 noise band for nineteen consecutive readings across two
full runs. Conversion and generalisation turned out to be separable, and
self-play alone only ever taught the first one.

### 3.3 Self-play generalises what it's repeatedly shown — not an abstract concept

Option A (run16) added a material-count input plane, handing the network
the correctly-computed material balance directly rather than requiring it
to derive counting from raw piece positions. The result was the clearest
diagnostic finding of the arc: `missing_queen` and `missing_rook`
(queen/rook-scale material) genuinely, robustly generalised — while
`missing_bishop`, `missing_knight`, and `missing_two_pawns`
**got monotonically worse across four consecutive time windows**, worse
in relative terms than before the fix.

The reason: `train_chess.py`'s material adjudication only ever fires
above a threshold (originally 8+). Self-play structurally cannot generate
reinforced training examples below that threshold, no matter how many
games are played — the data those positions would need simply never gets
labelled. Self-play doesn't learn "material matters" as a general
principle; it learns whatever magnitude it's repeatedly shown a decisive
outcome for, and nothing else.

### 3.4 A second, cheaper self-play fix confirmed the diagnosis rather than solving it

Rung 1b (run17) tested the cheapest possible response: a second,
lower-confidence adjudication tier for the 3–7 band, still entirely
self-play, no external evaluator. Across three large, well-separated
windows (89 readings, ~1,780 games), `missing_bishop` and `missing_knight`
declined from modest early promise (43%/27% correct sign) to essentially
never correct (3%) — worse than run16's terminal state. This wasn't
ambiguous. It confirmed that the problem is a *coverage* problem
self-play cannot solve on its own, regardless of how the threshold is
tuned, not a tuning problem.

### 3.5 Directly supplying the missing signal works — but only where the signal is loud enough

Option C (run18) sidestepped the threshold entirely: positions from
self-play in the 1–7 material range, relabelled with real Stockfish
evaluations (depth 16), blended 80/20 toward Stockfish, folded into the
permanent training partition (~9,000 positions, deliberately large and
diverse to avoid the small-permanent-partition memorisation trap that
caused the original run13 bug).

The result, over 64 readings across three windows (1,280 games, the
run's full and final length):
`missing_queen` reached **100% correct across every single window**,
independently cross-verified by `regression.csv` to four decimal places.
`missing_rook` held 76–95% correct throughout. Both `black_missing_queen`
and `black_missing_rook` showed real, sustained improvement over their
run17 state.

But `missing_bishop`, `missing_knight`, and `missing_two_pawns` declined
across the same three windows, ending below a coin flip. The most likely
reason: Stockfish evaluations for large material differences are often
forcing (mate lines the engine can see at depth 16), producing a loud
target nearly identical in character to ordinary ±1.0 self-play outcomes.
Moderate material differences rarely force anything — Stockfish returns
softer centipawn-scale evaluations that, after scaling, are a
different-shaped signal getting outcompeted by the dominant all-or-nothing
pattern surrounding them in every batch.

## 4. What HAL can and cannot do today

| Capability | Status |
|---|---|
| Convert a winning position vs a weak opponent | **Solved.** 100% vs random, holding since run15. |
| Recognise queen/rook-scale material advantage in unfamiliar positions | **Solved.** 100%/76-95% correct, cross-verified, stable across three windows. |
| Recognise bishop/knight/pawn-scale material advantage in unfamiliar positions | **Unsolved.** Below coin-flip after every fix tried. |
| Beat Stockfish depth 1 | **Unsolved.** 0 wins, 0 draws, in 300+ games across the entire project, every checkpoint measured, no movement in either direction. |
| Avoid the Fool's-Mate-family tactical pattern (early kingside pawn pushes punished by piece infiltration) | **Unsolved, and unrelated to any of the above.** Recurs in every run, including run18's game 1. A genuinely different capability gap — tactical foresight, not material evaluation. |

## 5. Key learnings, project-wide

**L1 — Pre-registration is what makes a negative result trustworthy.**
Every gate in this arc — run14's Gate 3, run16's GREEN, run17's RED,
run18's split verdict — was scored against thresholds fixed before the
data existed. This is the only reason "self-play-only fixes are
exhausted" is a conclusion rather than a guess: the criteria for reaching
it were written down before anyone knew which way the numbers would land.

**L2 — A single-position regression test is a fragile instrument; a
seven-position battery with independent cross-checks is a much better
one.** The original `missing_queen`-only regression metric couldn't
distinguish "the network is confused" from "this one FEN happens to be
unlucky." `material_probe.csv`'s broader battery — varying piece type and
which side is ahead — is what actually let run16's diagnosis (queen/rook
works, small pieces don't) be stated with confidence rather than
suspected.

**L3 — Windowed trend analysis catches false positives that a single
reading cannot.** Run17's `black_missing_queen` looked like a clean win at
320 games (90% correct) and had reversed to 31% by 1,080. Run18's
game-20 reading for bishop/knight/pawns looked like a total solution and
had reverted to noise by game 40. Every genuine verdict in this arc came
from multiple large, separated windows agreeing, never from a single
exciting number.

**L4 — Diagnosing the mechanism matters more than fixing the symptom.**
The generalisation gap was never one problem. It was "conversion vs
generalisation" (rung 1 vs everything after), then "which magnitude gets
training signal" (Option A's split result), then "how loud is the signal
once it exists" (Option C's further split). Each fix that worked, worked
because the diagnosis from the previous failure pointed at a specific,
falsifiable mechanism — not because of a broader or more expensive
intervention in the abstract.

**L5 — Self-play's ceiling on this architecture, at this scale, is now
known rather than assumed.** Two independent self-play-only attempts
(rungs 1 and 1b) failed to teach material generalisation below a
reinforcement threshold. That's not a tuning failure — it's now a fairly
well-evidenced structural limit of pure self-play at this scale, echoing
(at a much smaller scale) the same reason AlphaZero-class systems needed
tens of millions of games: self-play generalises what it's shown, and at
this game count the *what it's shown* distribution has hard edges.

**L6 — Consulting an external evaluator is not one decision, it's a
dial, and different settings answer different questions.** Option C's
result — a clean win at one magnitude, no win at another, from the
*same* mechanism and the *same* training run — shows that "does
consulting Stockfish help" was never a single yes/no question. It helped
exactly where its signal was loud enough to compete with self-play's
existing patterns, and not where it wasn't. Any future escalation should
be read the same way: as a specific, testable claim about signal
strength, not a blanket philosophical step.

## 6. What's next

Three live options, not yet decided:

1. **Escalate α further toward Stockfish** for the already-relabelled
   position set (no rebuild needed — same 9,000 positions, different
   blend weight). Best technical odds, real additional cost to the
   self-play story, bounded to the same ~25–30%-of-batch footprint
   already in play.
2. **Anneal `perm_ratio`** — untested since it was first proposed on
   run14's original intervention ladder. Stays self-play-adjacent, weaker
   odds of success on the current diagnosis (the failure looks like a
   signal-strength problem, not an exposure-frequency one).
3. **Stop here.** `missing_queen`'s result alone is a genuine, durable,
   independently-verified capability HAL did not have six runs ago.
   Accepting the bishop/knight/pawn gap as a bounded, known limitation
   and moving to Phase 4 (UCI wrapper, Lichess rating) is a legitimate,
   not a defeated, choice.

None of these are blocked on more data or more analysis — the arc's
diagnostic work is complete. This is a values decision, not a technical
one, and belongs to Rob.

## 7. Conclusion

The initial reinforcement learning arc set out to build a chess agent
from self-play and see how far it could get. It got further than run13's
stall suggested was possible — a perfect conversion rate, and a
genuinely generalised, cross-verified understanding of large material
advantage that didn't exist six runs ago. It also found, and then
confirmed twice, a real structural limit: self-play alone cannot teach
this architecture, at this scale, to value material below the magnitude
it gets repeatedly shown a decisive outcome for. Closing the arc here
isn't an admission that the approach failed — it's the point where every
cheap and moderate-cost lever has been pulled, tested against
pre-registered criteria, and the remaining path forward is a known,
quantified trade rather than an unexamined one.

---

## Annex A — `material_probe.csv` final state, side by side (run16 / run17 / run18)

Fraction of readings with the theoretically correct sign, terminal window
of each run:

| Position | run16 (final, 1580–2120) | run17 (final, 1220–1780) | run18 (final, 880–1280) |
|---|---|---|---|
| `missing_queen` | 37% | 93% | **100%** |
| `missing_rook` | 63% | 90% | 76% |
| `missing_bishop` | 0% | 3% | 24% |
| `missing_knight` | 0% | 3% | 33% |
| `missing_two_pawns` | 7% | 24% | 38% |
| `black_missing_queen` | 96% | 72% | 81% |
| `black_missing_rook` | 56% | 10% | 62% |

## Annex B — Eval history vs Stockfish depth 1 (every checkpoint measured)

| Run | Checkpoint (steps) | Result |
|---|---|---|
| run14 | 7,500 | 0/50 |
| run15 | 7,460 / 7,940 / 14,950 | 0/50 each |
| run17 | 720 / 4,900 / 7,500 | 0/50 each |

**Combined: 0 wins, 0 draws, in 300+ games, across every checkpoint
measured in this project's history.**

## Annex C — Full active-training-time accounting

| Run | Games | Active training time |
|---|---|---|
| run13_retune | 1,330 | 47.0h |
| run14 | 3,279 | 119.0h |
| run15 | 6,840 | 153.3h |
| run16 | 2,493 | 42.6h |
| run17 | 1,788 | 26.7h |
| run18 | 1,280 | 19.6h |

**Total: 17,010 games, ~408.2 active hours (~17.0 days) of continuous
GPU training across the arc.**
