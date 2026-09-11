# Search Misranking Hanging-Piece Captures at Small/Medium Material — Findings and Options

**Project:** chess-ai
**Phase:** 3 — AlphaZero-style chess agent (HAL-4000)
**Authors:** Rob Kirkland, Ellis Ward
**Date:** 11 September 2026 (run20, step 18,725); revised 11 September 2026
after external review; revised 12 September 2026 with steps 1-4 results
**Status:** Steps 1-4 of the agreed investigation (§6) are complete. Option
5 (checkpoint-based monitoring) and a decision on Option 2 remain open.

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

1. **Inspect the two policy-correct/search-wrong cases in detail.** ✅ done
   — see §7. `undefended_knight_on_e4` and
   `undefended_knight_on_d5_black_captures`. Record visits, priors,
   immediate post-move network values, and representative continuations
   for both the correct move and search's chosen move. Compare values
   from the same player's perspective throughout.
2. **Run a controlled search comparison.** ✅ done — see §7. Try serial
   search (`batch_sims=1`) against the current batching, keeping weights
   and simulation budget fixed. Tests whether batching materially changes
   the result — if the misranking persists identically under serial
   search, batching is not the explanation.
3. **Check network values against Stockfish on the actual sampled leaves.**
   ✅ done — see §7. Rather than only comparing root `Q`, evaluate the
   specific leaf positions search's tree actually visited against
   Stockfish. This is much stronger evidence of where — and whether —
   evaluation error enters the search, as opposed to inferring it
   indirectly from the root summary statistic.
4. **Expand to unseen positions and real games.** ✅ done — see §9 and
   §10. (this is Option 3 and Option 4 above, run together with steps
   1–3): rook and queen cases,
   both colours, and captures that should be *declined* — not only
   positions where capturing is correct. Otherwise the benchmark risks
   rewarding "always capture" rather than sound judgement. If Option 2
   later becomes justified by this investigation, any relabelling data
   should include both good and bad continuations, not only positions
   after successful captures, and the evaluation positions must stay
   separate from anything added to training data.

This investigation is scoped to run before deciding on Option 2, not
before starting a short, explicitly measured run21 baseline (§4).

## 7. Results: steps 1 and 2 (run20, 18,725 steps, desktop, 600 sims)

**Step 2 (batched vs serial) — batching is not the explanation.** All 5
hanging-piece positions were run through search with `batch_sims=32`
(current default) and `batch_sims=1` (fully serial), same weights, same
600-simulation budget:

| Position | Correct | Batched (32) | Serial (1) |
|---|---|---|---|
| `undefended_knight_on_e4` | `d3e4` | fail: `c4e6` (Q=-0.138) | fail: `c4e6` (Q=-0.152) |
| `undefended_bishop_on_f5` | `d3f5` | fail: `h2h4` (Q=+0.023) | fail: `h2h4` (Q=-0.113) |
| `undefended_knight_on_d5_black_captures` | `f6d5` | fail: `e7e6` (Q=+0.426) | fail: `b8a6` (Q=+0.338) |
| `undefended_queen_on_a5` | `b6a5` | pass (Q=+0.866) | pass (Q=+0.867) |
| `undefended_pawn_on_e5` | `d6e5` | fail: `g7g5` (Q=+0.391) | fail: `f7f5` (Q=+0.396) |

Pass/fail is identical between batched and serial search on all 5
positions — serial search reproduces every failure. That rules out
in-tree batching (and the virtual-loss approximation it relies on) as
the explanation for these specific misrankings. Worth noting: on 2 of
the 3 failing positions, the *specific* wrong move chosen differs between
batched and serial (`e7e6` vs `b8a6`; `g7g5` vs `f7f5`) — so batching
does have some influence on which losing move search settles on, just
not on whether the position is misjudged at all.

**Step 1 (shallow value vs Stockfish, no search) — points at the value
head, not just search dynamics.** For the two policy-correct/search-wrong
positions, the raw network value one ply deeper — no search, just the
value head's own read of the position immediately after each candidate
move, flipped to the mover's perspective so it's directly comparable to
root `Q` — was checked against the correct and chosen moves:

| Position | Move | Search Q (600 sims) | 1-ply value, no search |
|---|---|---|---|
| `undefended_knight_on_e4` | `d3e4` (correct, +400cp per Stockfish) | -0.490 | **-0.595** |
| `undefended_knight_on_e4` | `c4e6` (chosen, -297cp per Stockfish) | -0.138 | **+0.795** |
| `undefended_knight_on_d5_black_captures` | `f6d5` (correct) | +0.136 | +0.605 |
| `undefended_knight_on_d5_black_captures` | `e7e6` (chosen) | +0.426 | +0.701 |

For `undefended_knight_on_e4`, the misranking is already fully present at
one ply with zero search: the network's own value head rates the move
Stockfish scores at +400cp as *bad* for White (-0.595) and the move
Stockfish scores at -297cp as *very good* for White (+0.795) — a
complete reversal, before search has contributed anything. Search's
600-sim `Q` for `c4e6` (-0.138) is notably less extreme than this raw
1-ply read (+0.795), meaning search partially self-corrected the initial
overestimate as it explored continuations — but not enough to overtake
`d3e4`'s Q, and not enough sims were spent to see whether more search
would keep correcting it further.

For `undefended_knight_on_d5_black_captures`, the effect is present but
much smaller: both moves get a favourable-looking 1-ply value, and the
correct move (`f6d5`, winning a whole knight) reads *lower* than the
declined capture (`e7e6`) by about 0.1 — the wrong direction, but a much
smaller gap than the `e4` case's near-1.4 swing.

**Reading across both steps together:** the evidence now more directly
implicates the value head itself, at least for `undefended_knight_on_e4`
— the reversal is visible in the network's own one-ply evaluation with no
search involved at all, and persists under both batched and serial
search. This doesn't yet establish *why* the value head reads this
particular post-capture position so wrongly (representation gap, a
training-data gap on ...Bxe6-type positions, something else), and the
`d5` case shows the same direction of error but far more mildly, so this
still isn't confirmed as a clean, general pattern across positions —
consistent with the paper's standing caution against generalising past
n=2. Step 3 (Stockfish against the actual deeper leaves search explored,
not just this one-ply read) is the natural next check: it would show
whether the miscalibration is confined to the immediate post-move
position or persists through the specific continuations search built
underneath it (e.g. the `c4e6 f7e6 ...` line search favoured 105/232
times for the wrong move).

## 8. Results: step 3 (leaves vs Stockfish, run20, 18,725 steps, desktop, 600 sims)

`check_leaves_against_stockfish.py` walked every node visited ≥5 times,
up to 4 plies below each candidate move, and compared search's `Q`, the
raw network value at that exact position (no further search), and
Stockfish (depth 16) — all in the same "value for whoever just moved"
convention. 34 nodes had a materially decisive Stockfish score (|eval|
≥ 0.3 pawns; 7 near-neutral nodes were excluded as not meaningfully
signed either way). Scoring each node as "correct" when the sign of
`net` (or `Q`) matches the sign of Stockfish's eval:

| | net correct | Q correct | nodes checked |
|---|---|---|---|
| `undefended_knight_on_e4` subtree | 5/21 (24%) | 9/21 (43%) | 21 |
| `undefended_knight_on_d5_black_captures` subtree | 8/13 (62%) | 4/13 (31%) | 13 |
| **combined** | **13/34 (38%)** | **13/34 (38%)** | 34 |

Two things this changes about the read on this problem:

**The miscalibration is not confined to the first ply — it runs through
the whole local neighbourhood of positions search explored.** Under both
candidate moves, at every depth checked (not just the immediate
post-move position), the raw network value disagrees with Stockfish's
sign roughly 60-75% of the time in the `e4` subtree. This is a much
broader claim than step 1 supported on its own: it isn't one bad leaf,
it's the network being unreliable across a whole family of related
positions reached from this opening line.

**Search's aggregation does not reliably improve on the raw value — it
helps in one subtree and hurts in the other.** In the `e4` subtree, `Q`
is right more often than the raw `net` read (43% vs 24%) — some
self-correction, consistent with what step 1's root-level numbers
suggested. But in the `d5` subtree, it's reversed: `Q` is right *less*
often than `net` (31% vs 62%) — search's own visit-allocation made
things worse in that case, not better. An earlier draft of this
investigation (§7) read the `e4` result as search "partially
self-correcting" the value head's error; that generalisation doesn't
hold up once the `d5` subtree is checked the same way, and is corrected
here. Combined across both, `net` and `Q` end up exactly tied (13/34
each) — there's no consistent direction to whether search helps or hurts
relative to a single raw value read; it depends on the specific tree.

One example of search making things worse on its own terms:
`f6d5 h2h4` (case 2) has a raw net value of -0.415, correctly signed
against Stockfish's -5.07 — but after 11 visits, search's own `Q` for
that same node is +0.7433, a bad misjudgement search *introduced* on top
of an already-correct shallow read. Whatever is driving this looks like
general noise in how search integrates values across a subtree, not a
single fixable direction of bias.

**What this does and doesn't establish.** It's now well supported (34
data points, not 1) that the value head is frequently, non-directionally
wrong across this specific local neighbourhood of post-tactical
middlegame positions — sometimes overrating the losing side, sometimes
underrating the winning side, in both the correct-move and the
wrong-move subtrees. It does not establish *why* (representation gap,
undertrained region of position-space, something structural) — that
needs the positions to be checked for how well-represented they and
their neighbours are in training data, which is out of scope for this
diagnostic. It also still doesn't establish how far this generalises
beyond these two specific opening lines — that's exactly what step 4
(Option 3: broaden the benchmark; Option 4: check real games) is for,
and is now the clear next step rather than an optional add-on.

## 9. Results: step 4, Option 3 (benchmark expansion)

`generate_tactical_benchmark.py` gained 4 new positions, all validated
the same way as the original 5 (a legal capture must exist at the target
square, and — newly enforced for these additions — `chess.Board.
is_attacked_by()` confirms the target really is undefended, not just
capturable):

- `undefended_rook_on_a1_black_captures` and `undefended_rook_on_a8` —
  rook-scale, both colours (the original set had none).
- `undefended_queen_on_h4` — a second queen-scale case, opposite colour
  and opposite capturing piece from the original `undefended_queen_on_a5`
  (there Black's pawn takes White's queen; here White's knight takes
  Black's). Directly addresses the "n=1 for the one success case" gap
  §1 and §3 flagged.
- `elephant_trap_qgd` — a new category, `declined_capture`: the
  well-documented "Elephant Trap" line in the Queen's Gambit Declined,
  where the tempting recapture `6.Nxd5??` looks like it just wins back a
  pawn but is a real piece-losing blunder (`6...Nxd5 7.Bxd8 Bb4+ 8.Qd2
  Bxd2+ 9.Kxd2 Kxd8`). Verified with Stockfish before trusting it from
  memory: the position is +35cp for White, the bait move collapses it to
  -361cp, and 32 of the other 35 legal replies (e.g. simple development)
  hold the position. Without a category like this, the benchmark could
  reward "always capture" as a trivial heuristic rather than sound
  judgement (external review, 11 Sept 2026) — this is a first example,
  not full coverage of that risk.

The benchmark is now 14 positions across 5 categories. `run_tactical_
benchmark.py` and `diagnose_search_disagreements.py` needed no changes —
both already route any category other than `advantage_preservation`
through the same "chosen move must be in `correct_moves`" check, so
`declined_capture` slots in automatically. `pytest` gained one new test
(`test_declined_capture_bait_move_is_excluded_from_correct_moves`) and
the two existing category-coverage tests were extended; 49/49 pass.

One honest limitation: `declined_capture` positions are, by construction,
easy to pass by chance when most legal replies are fine (32/36 here) — a
policy with no particular affinity for the bait move will avoid it most
of the time regardless of whether it "understands" the trap. This is
inherent to how real chess traps work (there are usually many safe
alternatives), not a flaw specific to this implementation, but it means
a single `declined_capture` pass/fail is a much weaker signal than a
`hanging_piece` one — worth more positions in this category before
reading much into any one checkpoint's score here.

## 10. Results: step 4, Option 4 (real self-play games)

`mine_real_hanging_pieces.py` scans a `games.csv` ply by ply and flags
every position where the side to move has a legal, genuinely undefended
capture available (same `is_attacked_by` check used to validate the new
benchmark positions above), then checks whether the move actually played
took the most valuable one available. This Mac happens to have local
copies of `logs/run19/games.csv` and `logs/run20/games.csv` already, so
this ran directly against real self-play data rather than waiting on the
desktop.

**Full games (137,636 opportunities across both logs' games):**

| Piece | n | took it | take rate |
|---|---|---|---|
| queen (9) | 4,709 | 3,640 | 77.3% |
| rook (5) | 12,990 | 6,323 | 48.7% |
| bishop (3) | 16,915 | 7,483 | 44.2% |
| knight (3) | 16,176 | 6,466 | 40.0% |
| pawn (1) | 86,846 | 17,473 | 20.1% |

**Restricted to the greedy phase only (ply ≥ 31, past `TEMP_MOVES=30`
where self-play switches from stochastic sampling + Dirichlet noise to
plain argmax — the same selection mode the benchmark itself measures):**

| Piece | n | took it | take rate |
|---|---|---|---|
| queen (9) | 3,589 | 2,930 | 81.6% |
| rook (5) | 9,620 | 5,331 | 55.4% |
| bishop (3) | 10,947 | 5,526 | 50.5% |
| knight (3) | 10,562 | 4,787 | 45.3% |
| pawn (1) | 56,180 | 12,639 | 22.5% |

(Correction, 12 Sept 2026: the first version of this table reported a
single "bishop" row at each value=3 level. Knight and bishop share the
same material value and `mine_real_hanging_pieces.py` originally grouped
by value alone, silently merging the two under whichever piece name
happened to be picked first — not wrong in overall shape, but mislabeled.
Fixed to group by piece type; the two split out close to each other as
shown above, bishop consistently a few points higher than knight in both
cuts.)

Two things worth noting immediately: the magnitude-ordered pattern
(queen > rook > bishop, pawn excluded — see below) holds in real games
across tens of thousands of instances, not just the 5-position
benchmark's n=1 queen case. And the pattern is if anything *slightly
stronger* in the greedy-only cut than the full-game numbers, which rules
out "this is just training-time exploration noise from Dirichlet noise
and stochastic sampling" as an explanation for the whole effect — it's
still there, and slightly worse, when only greedy (argmax) choices are
counted.

**The pawn row should not be read the same way as the others.** A
material-only, defense-based check like this one can't tell a genuine
blunder from a materially-free pawn that has a legitimate positional
reason to decline (king safety, development, structure) — real games are
full of those, unlike the hand-authored benchmark where every "correct"
answer was designed to have no such downside. A ~20-23% pawn take rate
may partly or mostly reflect *sound* declines, not misjudgement. Knight
value and above is a much cleaner signal, since it's harder to have an
innocent positional reason to leave a whole minor piece or more
uncontested.

**Spot-checked against Stockfish, not just asserted.** The detector's
"undefended and capturable" rule also can't distinguish a real blunder
from a case where declining was correct for a reason it can't see (a
`declined_capture`-style trap, or something else). `verify_missed_hangs.py`
sampled 25 random knight-value-and-above misses and compared Stockfish's
eval after the capturing move against its eval after the move actually
played:

- Full-game sample (25 positions): **14/25 (56%) confirmed real blunders**
  (capturing would have scored ≥150cp better), median swing +185cp. 6/25
  were "fine to decline" (the played move actually scored *better* than
  capturing — a real tactical reason existed). The rest were mild.
- Greedy-phase-only sample (25 positions, ply ≥ 31): **16/25 (64%)
  confirmed real blunders**, median swing +328cp — if anything a
  stronger result than the full-game sample.
- One striking example from the greedy-phase sample: `logs/run20/
  games.csv` game 1378, ply 12 — Stockfish rates the position after
  taking the free knight at +241cp, but the move actually played leads
  to a position Stockfish scores as essentially lost (a swing north of
  +10,000cp on the scale used here). This is a concrete instance of
  exactly the kind of large, avoidable blunder the tactical benchmark
  was built to detect — found in a real game, not a constructed one.

**Reading steps 1-4 together:** the search-misranking pattern step 1-3
found in two hand-authored positions is not an artefact of those
specific lines. It shows up at scale in real self-play games, correlates
with material magnitude in the direction the original benchmark
suggested (queen best, then rook, then bishop), persists (slightly
worse, if anything) when training-time exploration noise is excluded,
and a solid majority of a random sample of flagged instances are
confirmed by Stockfish to be genuine, substantial blunders rather than
detector false positives. This is now a well-supported claim across a
large, real sample — not proof of a single root cause (steps 1-3's own
finding was that the value head is noisy in a way search doesn't
reliably correct, not that it's wrong in one consistent direction), but
strong enough to treat "HAL leaves real material on the board in real
games at a materially significant rate" as established rather than
speculative going into any decision about Option 2.

## 11. run21 overnight check (330 games, ~1,650 steps, 12 Sept 2026)

run21 (warm-started from run20, per §4/§6) ran overnight: 330 games,
steps 0 → ~1,650 (run20 finished at 18,725, so this is under 10% more
training on top of that). Only the logged CSVs were pulled, not the
checkpoint itself (`checkpoints/` stays local, same as every prior run)
— so this is a game-log-only check, not a check against run21's actual
trained weights.

**Standard training metrics look healthy, nothing alarming:** `avg_loss`
flat around 1.70-1.71, policy/value loss both flat, white/black win
balance roughly even (~45-55% split each interval), average game length
73-80 plies, end reasons dominated by material adjudication as usual.
`material_probe` readings (missing_queen ≈ -0.60, missing_rook ≈ -0.47,
missing_bishop ≈ -0.08, missing_knight ≈ +0.02, missing_two_pawns ≈
-0.35) are flat across all 16 readings in this window — no meaningful
movement either direction, which is expected this early and isn't a red
flag on its own.

**`mine_real_hanging_pieces.py` on run21's own 330 games (no checkpoint
needed, just the move log) closely tracks the run19+run20 baseline:**

| Piece | run19+run20 (137,636 opps) | run21, 330 games (9,601 opps) |
|---|---|---|
| queen | 77.3% | 75.8% |
| rook | 48.7% | 51.0% |
| bishop | 44.2% | 46.2% |
| knight | 40.0% | 41.0% |
| pawn | 20.1% | 22.0% |

Every row is within a couple of points of the baseline — no sign of
either improvement or further drift yet. That's the expected result, not
a null finding worth alarm: run21 has no training intervention in it
(Option 2 is still on hold), so there was never a reason to expect ~330
games / ~1,650 steps to move a pattern that was stable across run19 and
run20's much larger combined sample. This overnight run is functioning
correctly as the "short, measured baseline" it was scoped to be — it
just hasn't run long enough yet to be informative one way or the other
on the search-misranking question specifically.

**What would make this check more informative:** a checkpoint-based
comparison (running `run_tactical_benchmark.py` and the step 1-3
diagnostics against run21's actual trained weights, not just its game
log) would need `checkpoints/run21_hal_chess.pt` transferred over as
well. Given how little the logged metrics have moved, that's probably
not worth doing yet at this step count — better to let it accumulate
substantially more training first, or use the next check-in to decide
whether it's time to revisit Option 2 instead of waiting longer on
unmodified self-play.
