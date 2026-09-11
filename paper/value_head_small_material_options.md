# Search Misranking Hanging-Piece Captures at Small/Medium Material — Findings and Options

**Project:** chess-ai
**Phase:** 3 — AlphaZero-style chess agent (HAL-4000)
**Authors:** Rob Kirkland, Ellis Ward
**Date:** 11 September 2026 (run20, step 18,725); revised 11 September 2026
after external review
**Status:** Decision document. Investigation direction agreed (see §6);
options 3 and 4 are the next work, option 5 to follow, option 2 on hold.

---

## 1. Summary

A newly-built tactical benchmark found that on 4 of 5 constructed
hanging-piece positions, HAL's search ends up choosing not to capture an
undefended piece, despite the capture being objectively correct by a wide
margin (Stockfish confirms 100–700+ centipawn swings each time). The
breakdown across those five positions is not uniform:

- **Two positions:** the raw policy prior correctly favours the capture,
  but search overturns it and lands on a worse move.
- **Two positions:** the policy is already wrong before search gets
  involved.
- **One position:** both the policy and search get it right (the largest
  material swing tested, a full queen).

The clearest and most actionable result is the first group: two cases
where a demonstrably correct policy choice is overturned by search into a
concrete blunder. That's a stronger and more specific finding than
"HAL misjudges small material" in general — it points at *search's own
move-ranking* on these positions, not necessarily at the value network in
isolation. This may be related to an already-documented, independent
finding that HAL's value network reads material deficits at queen/rook
scale reliably but bishop/knight/two-pawn scale weakly and noisily
(`material_probe_correction.md`), but the two findings haven't yet been
shown to share a single cause.

**What's confirmed:** the specific pattern in the specific positions
tested, reproduced with real move history, verified against Stockfish. In
every one of the four failures, the move search actually chose has a
higher accumulated value estimate (`Q`) at the root than the correct
move's `Q` — search is ranking these moves incorrectly.

**What's not confirmed:**
- *Why* `Q` comes out backwards. `Q` is an accumulated estimate built from
  whatever continuations search actually explored — priors, visit
  allocation, search depth, and batching all influence which leaves feed
  into it. A higher `Q` for the wrong move tells us search's ranking is
  wrong; it doesn't by itself isolate the value head as the cause over
  exploration or batching effects. Those haven't been ruled out (see §5,
  correcting an earlier draft of this document that claimed they had).
- Whether this is a strict function of material *magnitude* (large fine,
  small/medium broken) or something narrower to these specific lines —
  one success case (n=1) is not enough to establish a magnitude
  threshold, especially since capturing a piece changes board geometry,
  threats, and available replies as well as material.
- Whether it happens in real self-play games at a meaningful rate, or
  only in this hand-constructed benchmark.
- Whether it would improve on its own with more training.

## 2. How this was found

Working backward from the tactical benchmark built this week
(`chessai/tactical_benchmark.json`, `run_tactical_benchmark.py`):

1. `run_tactical_benchmark.py` scored the current checkpoint (run20, step
   18,725) at both the raw policy prior and a fixed search budget,
   separately, across 10 hand-authored positions (5 hanging-piece,
   2 mate-in-1, 1 defend-mate-in-1, 2 advantage-preservation) — each
   reached by a real, legal move sequence with genuine history, not an
   end-state FEN (the `material_probe` lesson this whole line of work is
   built on: an empty-history position tests something the network has
   never seen, regardless of what it's nominally checking).
2. One result stood out: `undefended_knight_on_e4` failed under search at
   both 200 and 600 simulations with the *identical* wrong move each
   time, despite the raw policy prior correctly picking the capture. That
   stability (not converging toward the right answer with 3x the search
   budget) is a different shape of problem than the other benchmark
   failures, which mostly resolved once given a proper (600-sim,
   training-matched) budget. Note that this only establishes persistence
   at 200 and 600 simulations specifically — it doesn't rule out the
   result changing at other search budgets or with a different allocation
   strategy, and matching the training budget doesn't by itself establish
   that the budget is adequate.
3. `diagnose_knight_e4_anomaly.py` inspected the actual search tree at
   that one position: the wrong move's own `Q` (the value estimate the
   search itself built up from evaluating that move's subtree) was
   *higher* than the correct move's `Q`. Stockfish independently confirmed
   the correct move is +400cp for White and the search's chosen move is
   -297cp — a ~700cp swing, a real blunder, not a defensible alternative.
4. To check whether this was an isolated quirk of that one line, three
   more hanging-piece positions were added — deliberately varied (mirrored
   colour, different piece values, different capturing pieces) — and
   `diagnose_search_disagreements.py` generalised the same root-tree
   analysis to run across all of them automatically.

## 3. The evidence

All five hanging-piece positions, checkpoint run20 (18,725 steps),
600-simulation search, `add_noise=False` (matching real evaluation/play):

| Position | Material at stake | Policy | Search's choice | Correct move's Q | Wrong move's Q |
|---|---|---|---|---|---|
| `undefended_knight_on_e4` | Knight (~3) | correct | wrong (`c4e6`) | -0.490 | **-0.138** |
| `undefended_bishop_on_f5` | Bishop (~3) | wrong | wrong (`h2h4`) | -0.411 | **+0.023** |
| `undefended_knight_on_d5_black_captures` | Knight (~3) | correct | wrong (`e7e6`) | +0.136 | **+0.426** |
| `undefended_pawn_on_e5` | Pawn (~1) | wrong | wrong (`g7g5`) | -0.616 | **+0.391** |
| `undefended_queen_on_a5` | Queen (~9) | correct | **correct** | — | — |

In every one of the four failures, the move search chose has a `Q` that
reads higher than the correct move's `Q` — search's own ranking is
backwards, checked directly rather than inferred (`diagnose_search_
disagreements.py`'s output is reproducible from the same checkpoint).
That's the strongest and most direct claim this evidence supports.

It is *not* strong enough evidence, on its own, to name a specific cause.
`Q` reading backwards is consistent with several distinct explanations —
a genuinely inaccurate leaf evaluation somewhere in the subtree, an
exploration/prior effect that shaped which continuations got visited, an
interaction between the two — and this benchmark hasn't separated them.
The magnitude pattern (queen succeeds, everything smaller fails) is
suggestive with n=1 for the "succeeds" case, but capturing a queen versus
capturing a knight or pawn also changes the resulting position's threats
and reply options, not just the material count — so it isn't yet possible
to say the material *magnitude* itself is what's doing the work.

Two things this doesn't yet tell us, worth being precise about: whether
the *policy* being wrong on 2 of 5 positions (`undefended_bishop_on_f5`,
`undefended_pawn_on_e5`) is the same underlying issue or a separate one
(those two fail even before search gets involved, unlike the other two —
`undefended_knight_on_e4`, `undefended_knight_on_d5_black_captures` —
where search actively makes a correct policy worse), and whether this
generalises beyond hand-constructed positions to HAL's actual self-play
games.

## 4. Why this matters for the run21 decision

Run21 (warm-started from run20, otherwise unchanged) was scoped as a
continuation to validate this session's measurement fixes on a clean
run, not to address a specific capability gap. If this search-ranking
weakness is real and structural, plain continued self-play — which is
exactly what has been running since run19 without this pattern
resolving — may not fix it on its own. Worth deciding deliberately
whether to proceed as planned, or investigate further first.

A short, explicitly measured run21 baseline is not blocked by this
finding. What should be avoided is committing to another long run without
first saving this benchmark's results at checkpoints along the way, so
the trajectory of this specific weakness is visible rather than only
knowable in hindsight.

## 5. Options

**Option 1 — Proceed with run21 as already planned, treat this as a
documented, accepted limitation for now.**
Cost: none beyond what's already planned. Risk: if this actively affects
real games (not just the constructed benchmark), it may be silently
capping playing strength in a way run21 won't measure or fix, and we'd
be running another full cycle without addressing a known issue.

**Option 2 — A targeted data/training intervention before run21: extend
`relabel_with_stockfish.py`/`curate_buffer.py`'s Stockfish-relabelled set
specifically toward "just captured material, what now" continuation
positions at the affected magnitudes.**
Directly targets a suspected root cause (weak calibration in exactly this
material-magnitude band) using infrastructure that already exists in the
project, rather than building something new. Cost: a real
generation-and-relabelling pass, similar effort to the original Option C
work — bounded, but not small, and delays starting run21 while it's
built and validated. Risk: the specific cause isn't established yet
(§3) — a training intervention aimed at "the value head" could miss if
the actual issue is more about exploration/search allocation, and would
need another round of this same benchmark to find out either way.
**Decision: on hold** — premature before the cause is better understood,
per the investigation in §6.

**Option 3 — Gather more evidence before committing to 1 or 2: expand
the tactical benchmark's magnitude coverage (more rook-scale and
queen-scale positions, both colours, and positions where the objectively
correct move is *declining* a capture, not just taking one) to properly
test whether this is a clean magnitude effect, a "search misranks
captures specifically" effect, or something narrower.**
Cheaper than Option 2, directly addresses the "n=1 for the one success"
gap in the current evidence, and guards against the benchmark trivially
rewarding "always capture" as a heuristic. Doesn't fix anything by
itself — an information-gathering step that could run in parallel with
starting run21 rather than blocking it. **Decision: proceed** (§6).

**Option 4 — Check whether this shows up in real self-play games, not
just the constructed benchmark**, by mining `games.csv` for actual
positions where a piece was left hanging and checking whether HAL took
it. Closes the gap between "a hand-built benchmark shows this" and "this
actually affects real play" — the thing that ultimately matters. Doesn't
block run21 either way; could run alongside it. **Decision: proceed**
(§6).

**Option 5 — Add ongoing tracking:** wire a lightweight version of this
benchmark into `chessai/logger.py` (same pattern as `material_probe`,
periodic during training) so this specific weakness's trajectory is
visible over the course of whatever run comes next, regardless of which
other option is chosen. Compatible with any of the above, not a
replacement for one. **Decision: proceed, framed as checkpoint-based
monitoring** rather than a per-step logger addition — save benchmark
results at checkpoints during any run so the trajectory is visible
without adding per-step overhead (§6).

Options 3, 4, and 5 are not mutually exclusive with proceeding on run21
(Option 1) — they're evidence-gathering and monitoring, not blockers.
Option 2 is the one that would actually delay starting run21, and is on
hold until §6's investigation narrows down the cause.

## 6. Agreed next investigation (external review, 11 Sept 2026)

Before choosing a remedy, the goal is to distinguish which of several
plausible contributing factors — inaccurate leaf values, exploration/prior
effects, search depth, batching — is actually driving the backwards `Q`
ranking. A deliberately small, four-step plan:

1. **Inspect the two policy-correct/search-wrong cases in detail.**
   `undefended_knight_on_e4` and `undefended_knight_on_d5_black_captures`.
   Record visits, priors, immediate post-move network values, and
   representative continuations for both the correct move and search's
   chosen move. Compare values from the same player's perspective
   throughout.
2. **Run a controlled search comparison.** Try serial search
   (`batch_sims=1`) against the current batching, keeping weights and
   simulation budget fixed. Tests whether batching materially changes the
   result — if the misranking persists identically under serial search,
   batching is not the explanation.
3. **Check network values against Stockfish on the actual sampled leaves.**
   Rather than only comparing root `Q`, evaluate the specific leaf
   positions search's tree actually visited against Stockfish. This is
   much stronger evidence of where — and whether — evaluation error enters
   the search, as opposed to inferring it indirectly from the root
   summary statistic.
4. **Expand to unseen positions and real games** (this is Option 3 and
   Option 4 above, run together with steps 1–3): rook and queen cases,
   both colours, and captures that should be *declined* — not only
   positions where capturing is correct. Otherwise the benchmark risks
   rewarding "always capture" rather than sound judgement. If Option 2
   later becomes justified by this investigation, any relabelling data
   should include both good and bad continuations, not only positions
   after successful captures, and the evaluation positions must stay
   separate from anything added to training data.

This investigation is scoped to run before deciding on Option 2, not
before starting a short, explicitly measured run21 baseline (§4).
