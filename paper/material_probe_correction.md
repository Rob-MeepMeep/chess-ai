# The Generalisation Gap Was a Measurement Bug

**Project:** chess-ai
**Phase:** 3 — AlphaZero-style chess agent (HAL-4000)
**Authors:** Rob Kirkland, Ellis Ward
**Date:** 3 September 2026 (run19, game ~1190, step 5,950)
**Status:** Resolved. Corrects `material_probe.csv`'s reading for
`missing_bishop`, `missing_knight`, and `missing_two_pawns` across the
entire run16–19 arc, and closes out run19's decision protocol on
different grounds than it was written to expect.

---

## 1. Summary

`material_probe` — the diagnostic run16 introduced to check whether the
value head has learned "material matters" as a general concept — has
been testing the network on board positions that cannot occur in real
chess: the starting position with one piece deleted, zero moves played,
empty move history. Removing a piece requires a capture, which requires
history; this exact combination never appears anywhere in self-play,
Stockfish-relabelled, or any other training data the network has ever
seen. On the current run19 checkpoint, that construction reads near zero
(or wrong-signed) for `missing_bishop`/`missing_knight`/`missing_two_pawns`
— the finding that drove rung 1b, run17's Stockfish relabelling, and
both run18 and run19's alpha escalation. On the **same checkpoint**,
given real mid-game positions with real history at the identical material
magnitudes, those same categories read a clean, correctly-signed -0.30 to
-0.37, in a smooth, monotonic curve alongside every other magnitude from
±1 to ±20+. The network already knew this. The probe couldn't ask it the
question properly.

`chessai/logger.py` is fixed (commit `d6ffb24`) to use real in-context
positions instead. `run20` is set up as a clean continuation from run19's
checkpoint so `material_probe.csv` has no legacy discontinuity from game 0.

## 2. Background — what material_probe was, and what it drove

`MATERIAL_PROBE_POSITIONS` (`chessai/logger.py`, introduced run16 as
generalisation-gap Option A) tested seven fixed FENs every 20 games: the
standard opening with one piece removed, per category
(`missing_queen`, `missing_rook`, `missing_bishop`, `missing_knight`,
`missing_two_pawns`, and two "black is down material" mirrors), each
evaluated with `agent.get_value(board, [])` — empty history. Since
run16, its readings drove the arc's central narrative:

| Run | What it tested | material_probe's small-piece verdict |
|---|---|---|
| run16 | Option A: material-count input plane | large material generalised, small material got worse |
| run17 | Rung 1b: second self-play adjudication tier | declined further — RED, self-play-only exhausted |
| run18 | Option C: Stockfish-relabelled positions, α=0.2 | queen/rook solved, bishop/knight/pawns still failing |
| run19 | Option C escalated, α=0.05 | flat/declining through game 1190 — trending RED |

Four runs, each treating a flat-or-declining `missing_bishop`/`missing_knight`/
`missing_two_pawns` reading as ground truth about the network's actual
capability, and each responding with a real, costly intervention (a new
adjudication tier, an hour of Stockfish evaluation, two rounds of alpha
retuning).

## 3. The bug

`chessai/logger.py`, pre-fix:

```python
MATERIAL_PROBE_POSITIONS = {
    "missing_bishop": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1",
    ...
}
...
def record_material_probe(self, game_num, agent):
    for key, fen in MATERIAL_PROBE_POSITIONS.items():
        board = chess.Board(fen)
        values[key] = agent.get_value(board, [])   # <- empty history
```

Compare `relabel_with_stockfish.py`, which builds the positions the
network actually trains on: `state = encode([pos_board] + pos_history)`,
sampled from real self-play games at plies 8–28, with up to 3 frames of
genuine prior-move history filling 48 of the network's 55 input planes.
The probe's positions have zero moves played and all 48 history planes
zeroed — a shape that cannot arise from a legal game once material has
already changed, and one the network has never been trained on, at any
magnitude, in any run.

A related note already existed in `generalization_gap_options.md`
(13 August, run15 era), specifically about the `missing_queen` FEN: *"The
test position ... is structurally close to unreachable through legal
self-play ... It has no near neighbour anywhere in the seed buffer or in
organic self-play data."* That observation was correct and specific to
one FEN — it wasn't generalised at the time to the other six categories
`material_probe` added at run16, and no one revisited whether the same
argument applied to all of them equally. It did. Option A (the material
plane, plane 54) was built afterward as a mitigation and genuinely helped
large-magnitude generalisation, but it didn't touch the measurement
problem for small magnitudes — the probe stayed broken underneath it.

## 4. How it was found

Three diagnostic scripts, run in sequence against the run19 checkpoint
(step 5,950), each ruling out one explanation before the next:

1. **`diagnose_material_labels.py`** — checked whether the Stockfish
   labels the network trains on are themselves too weak/noisy at small
   material magnitudes to be learnable. They aren't: `sf_value` at
   |mat|=3 has mean -0.371 (std 0.575), a real signal, just noisier than
   |mat|=5's -0.667 (std 0.462) — and the small-magnitude buckets have
   *more* examples (786-823) than the large ones (395-496), ruling out
   underrepresentation too.
2. **`diagnose_network_output.py`** — ran the trained checkpoint forward
   on every cached Stockfish-relabelled position and correlated its
   prediction against the true label, per magnitude. Correlation was
   0.90-0.96 uniformly across the entire range, including |mat|=2-3 —
   the network was already tracking the target well on realistic
   positions. This is the result that reframed the question: if the
   network can do this, why does `material_probe` say it can't?
3. **`diagnose_probe_construction.py`** — ran the same checkpoint on
   both the exact synthetic FENs `material_probe.csv` had been logging,
   and on fresh real positions at the same magnitudes sampled independently
   from run19's own `games.csv` (not the same positions used in #2, so
   not just re-measuring memorisation). Direct side-by-side result:

   | category | synthetic (old probe) | real, same checkpoint |
   |---|---|---|
   | missing_queen | -0.888 | -0.741 |
   | missing_rook | -0.725 | -0.526 |
   | missing_bishop | **-0.003** | **-0.372** |
   | missing_knight | **+0.016** (wrong sign) | **-0.372** |
   | missing_two_pawns | **-0.031** | **-0.301** |

   Queen and rook read somewhat weaker under real positions than
   synthetic (large-magnitude material is unambiguous enough that raw
   material dominates regardless of context, so the synthetic shape
   still "works" there by accident). Bishop, knight, and two-pawns don't
   just read weaker — the old probe reads statistically indistinguishable
   from zero (knight even flips sign) while the real reading is a clean,
   confident deficit signal, essentially matching the ~-0.37 that
   `diagnose_material_labels.py` established as the real target's true
   mean at that magnitude.

## 5. What this confirms, and what it doesn't

**Confirmed:** on the run19 checkpoint at step 5,950, the network has
genuinely learned to value bishop/knight/two-pawn deficits correctly in
realistic positions. The measurement, not the training, produced the
flat/declining readings that drove run17's RED verdict and run18/19's
alpha escalation.

**Not confirmed — worth being precise about:** this was only tested
directly against run19's current checkpoint. `MATERIAL_PROBE_POSITIONS`
was unchanged from run16 through run19, so the same construction bug
almost certainly affected every one of those runs' readings too — but
that's an inference from an unchanged bug, not a re-measurement.
Checkpoints for run16/run17/run18 (if still on disk) haven't been run
through `diagnose_probe_construction.py`. Whether the "gap" was *never*
real, or whether it was real early on and self-play/Option A/Option C
each closed some of it before the probe ever could have shown it, is an
open historical question. It doesn't change what to do next (§7), but it
should stay open rather than get quietly assumed either way — if it
matters later (e.g. for a paper claiming Option C's specific
contribution), the fix is to run `diagnose_probe_construction.py`-style
checks against any surviving earlier checkpoints, not to guess.

## 6. Disposition of run19's decision protocol

`run19_decision_protocol.md` pre-registered GREEN (sustained small-piece
improvement within ~1,000 games) against RED (flat/declining, stop
tuning α, move to Option B or accept the gap). Neither verdict applies as
written: the data the protocol was measuring wasn't measuring what it
claimed to. The protocol is marked **resolved, not RED** — see the
resolution section appended to that document. The practical consequence
(§7) happens to land close to what a real GREEN would have recommended,
but for a different reason: not "α=0.05 worked," but "there was nothing
here to fix at this α, and no evidence α needed escalating past run18's
0.2 in the first place." That distinction matters if anyone later asks
whether the alpha-escalation experiment itself succeeded — it wasn't a
clean test of that question either way, since the measurement was broken
throughout.

## 7. What changes going forward

- **`chessai/logger.py`** now uses real in-context positions: 30 move-
  sequences per category, sampled from run19's `games.csv` (positions
  where exactly one piece type differs by the category's exact amount,
  all else equal, White to move — see `generate_material_probe_positions.py`),
  replayed to reconstruct real history at read time, averaged per
  category. Commit `d6ffb24`.
- **`STOCKFISH_ALPHA` stays at 0.05.** No evidence it's hurting (queen/rook
  still read correctly on real positions), and changing it now would
  confound whether any future reading reflects the fix or a new alpha
  value.
- **No further alpha tuning planned.** The lever this arc spent four runs
  on wasn't shown to be broken — go build against the corrected probe
  before deciding anything else needs fixing.
- **`run20`** warm-starts from run19's checkpoint (`warm_start_run20.py`),
  same architecture, same buffer (run19's accumulated replay buffer
  carried forward, not the original curated seed — nothing about the
  data pipeline changed). The only reason for a new run number: a clean
  `material_probe.csv` from game 0, since run19's own file mixes old
  synthetic-FEN readings before this fix with real-position readings
  after it, not directly comparable within one file.
- **`paper/phase3_rl_arc_closing_report.md`** gets a short addendum
  pointing here — its run16/17/18 table rows describing "small-piece
  positions still failing" should be read as *probe readings*, not
  confirmed capability, pending the re-measurement noted in §5.

## References

- `chessai/logger.py` — the fix itself (`MATERIAL_PROBE_POSITIONS` comment
  and `record_material_probe`)
- `generate_material_probe_positions.py` — builds the real-position data set
- `diagnose_material_labels.py`, `diagnose_network_output.py`,
  `diagnose_probe_construction.py` — the three-step diagnosis (§4)
- `run19_decision_protocol.md` — resolution appended
- `generalization_gap_options.md` §1 — the original, narrower observation
  about `missing_queen`'s construction (13 August)
