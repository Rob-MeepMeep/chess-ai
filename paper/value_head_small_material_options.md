# Value-Head Miscalibration at Small/Medium Material — Options Going Forward

**Project:** chess-ai
**Phase:** 3 — AlphaZero-style chess agent (HAL-4000)
**Authors:** Rob Kirkland, Ellis Ward
**Date:** 11 September 2026 (run20, step 18,725)
**Status:** Decision document — no option chosen yet. Written for external
review before deciding how (or whether) this changes the plan to start
run21.

---

## 1. Summary

A newly-built tactical benchmark found that HAL's search actively declines
to capture a hanging (undefended, free) piece in 4 of 5 constructed
positions — not because the policy doesn't know better (it correctly
favours the capture in every case tested), but because the search's own
internal value estimate rates *not* capturing as better than capturing,
despite the capture being objectively correct by a wide margin (Stockfish
confirms 100–700+ centipawn swings each time). The one exception was the
single largest material swing tested (a full queen). This is consistent
with — and may be the same underlying cause as — an already-documented,
independent finding that HAL's value network reads material deficits at
queen/rook scale reliably but bishop/knight/two-pawn scale weakly and
noisily (`material_probe_correction.md`).

**What's confirmed:** the specific pattern in the specific positions
tested, reproduced with real move history, verified against Stockfish,
and traced to the search's value estimates specifically (not the policy,
not an exploration/search-budget artifact — ruled out directly, see §3).

**What's not confirmed:** whether this is a strict function of material
*magnitude* (large fine, small/medium broken), or something narrower to
these five specific lines; whether it happens in real self-play games at
a meaningful rate, or only in this hand-constructed benchmark; and
whether it would improve on its own with more training, the way some
other metrics in this project's history have.

## 2. How this was found

Working backward from the tactical benchmark built this week
(`chessai/tactical_benchmark.json`, `run_tactical_benchmark.py`):

1. `run_tactical_benchmark.py` scored the current checkpoint (run20, step
   18,725) at both the raw policy prior and a fixed search budget,
   separately, across 10 hand-authored positions (5 hanging-piece,
   2 mate-in-1, 1 defend-mate-in-1, 2 material-conversion) — each reached
   by a real, legal move sequence with genuine history, not an end-state
   FEN (the `material_probe` lesson this whole line of work is built on:
   an empty-history position tests something the network has never seen,
   regardless of what it's nominally checking).
2. One result stood out: `undefended_knight_on_e4` failed under search at
   both 200 and 600 simulations with the *identical* wrong move each
   time, despite the raw policy prior correctly picking the capture. That
   stability (not converging toward the right answer with 3x the search
   budget) is a different shape of problem than the other benchmark
   failures, which mostly resolved once given a proper (600-sim,
   training-matched) budget.
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

Every one of the four failures shows the identical mechanism: the wrong
move's `Q` reads higher than the correct move's `Q`. This was checked
directly, not inferred — `diagnose_search_disagreements.py`'s output is
reproducible from the same checkpoint. The magnitude pattern (queen
succeeds, everything smaller fails) is suggestive with n=1 for the
"succeeds" case — not something to treat as confirmed on its own yet
(see §5, option 3).

Two things this doesn't yet tell us, worth being precise about: whether
the *policy* being wrong on 2 of 5 positions (`undefended_bishop_on_f5`,
`undefended_pawn_on_e5`) is the same underlying issue or a separate one
(those two fail even before search gets involved, unlike the other three
where search actively makes a correct policy worse), and whether this
generalises beyond hand-constructed positions to HAL's actual self-play
games.

## 4. Why this matters for the run21 decision

Run21 (warm-started from run20, otherwise unchanged) was scoped as a
continuation to validate this session's measurement fixes on a clean
run, not to address a specific capability gap. If this value-head
weakness is real and structural, plain continued self-play — which is
exactly what has been running since run19 without this pattern
resolving — may not fix it on its own. Worth deciding deliberately
whether to proceed as planned, or address something first.

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
Directly targets the suspected root cause (weak calibration in exactly
this material-magnitude band) using infrastructure that already exists
in the project, rather than building something new. Cost: a real
generation-and-relabelling pass, similar effort to the original Option C
work — bounded, but not small, and delays starting run21 while it's
built and validated. Risk: might not fix it if the issue is more
structural (representation/architecture) than a data-coverage gap; would
need another round of this same benchmark to find out either way.

**Option 3 — Gather more evidence before committing to 1 or 2: expand
the tactical benchmark's magnitude coverage (more rook-scale and
queen-scale positions, not just one queen case) to properly test whether
this is a clean magnitude effect or something narrower.**
Cheaper than Option 2, directly addresses the "n=1 for the one success"
gap in the current evidence. Doesn't fix anything by itself — a
information-gathering step that could run in parallel with starting
run21 rather than blocking it.

**Option 4 — Check whether this shows up in real self-play games, not
just the constructed benchmark**, by mining `games.csv` for actual
positions where a piece was left hanging and checking whether HAL took
it. Closes the gap between "a hand-built benchmark shows this" and "this
actually affects real play" — the thing that ultimately matters. Doesn't
block run21 either way; could run alongside it.

**Option 5 — Add ongoing tracking:** wire a lightweight version of this
benchmark into `chessai/logger.py` (same pattern as `material_probe`,
periodic during training) so this specific weakness's trajectory is
visible over the course of whatever run comes next, regardless of which
other option is chosen. Compatible with any of the above, not a
replacement for one.

Options 3, 4, and 5 are not mutually exclusive with proceeding on run21
(Option 1) — they're evidence-gathering and monitoring, not blockers.
Option 2 is the one that would actually delay starting run21.
