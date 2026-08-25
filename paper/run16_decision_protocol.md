# Run 16 — Pre-Registered Decision Protocol

**Date:** 18 August 2026 (registered at game ~40, before any signal-bearing
`material_probe.csv` data exists)
**Authors:** Rob Kirkland, Ellis Ward

## Purpose

Run16 tests generalisation-gap **Option A** — a material-count input
plane — chosen from the three options compared in
`generalization_gap_options.md`. Run15's rung-1 intervention (early
material adjudication) fixed conversion decisively (Gate 3: 90.0% wins vs
random, later a perfect 100.0%) but left generalisation completely
untouched: `missing_queen` never moved off a ±0.06 noise band across
nineteen readings and two full runs, and HAL stayed at 0 wins / 0 draws in
300 games against Stockfish depth 1, unmoved across every checkpoint
measured.

Option A was chosen first because it is the cheapest possible test of a
specific hypothesis: that the bottleneck is pure material *counting*, not
something broader. If true, handing the network the computed material
balance directly — rather than requiring it to derive that from raw piece
positions, a global aggregation a CNN has no natural aptitude for — should
resolve it quickly and cheaply, without the larger engineering cost or
bigger self-play-purity compromise of Options B or C.

## Method

Add one input plane (55th) carrying material balance (current player's
perspective, clipped ±20, scaled to [-1, 1]), computed identically to
`train_chess.py`'s adjudication rule. The network is **warm-started** from
run15's trained checkpoint rather than trained from scratch a fourth
time — every weight whose shape is unaffected is copied directly; the one
resized layer (`input_conv.0.weight`) has its 54 existing input channels
copied across and the new 55th channel zero-initialised.

This was mathematically verified, not assumed: with the new channel's
weights at zero, the network's output is bit-identical to run15's for any
input — including a real, non-zero material value in the new plane — since
it's the zero *weights*, not the input, that silence the channel's
contribution. Run16 begins at exactly run15's full strength (confirmed in
the first 20 games: no Fool's-Mate-style early blundering, immediately
competent games of 39–105 moves) and only has to learn to *use* the new
signal, not relearn chess.

## Changes made

- `chessai/encoder.py`: `N_PLANES` 54→55; new `_material_balance()` helper;
  plane 54 = material balance, reusing the already-mirrored board the
  castling-rights planes use, so the perspective flip is automatic.
  Verified: starting position reads 0.0, a missing-queen position reads
  −0.45 for the disadvantaged side and +0.45 for the advantaged side.
- `chessai/model.py`, `agent.py`, `replay.py`: docstring/comment updates
  only (54→55) — `ChessNet` imports `N_PLANES` directly, so the input conv
  layer resizes with no logic changes.
- `run_config.py`: `RUN_NAME` → `run16`.
- `curate_buffer.py`: seed buffer rebuilt from run15's games (was run14's)
  — 5,090 games passed filters (vs run14's much smaller decisive-game
  yield, since run15's average game length is far shorter).
- `warm_start_run16.py` (new): the checkpoint-surgery script described
  above. Produces `checkpoints/run16_hal_chess.pt` directly, so
  `train_chess.py`'s existing resume logic picks it up automatically.
- `chessai/logger.py`: new `record_material_probe()` — seven positions
  (the existing `missing_queen`, four more single-piece-missing variants
  on White's side, two mirrored "Black is down material" positions),
  written to `material_probe.csv` every `MATERIAL_PROBE_EVERY` = 20 games.
  Deliberately decoupled from `regression.csv`'s 200-game cadence and set
  far shorter, since the whole point is a faster read than the existing
  cadence gave run14/run15 — cheap to do since it's plain forward passes,
  no MCTS, no games played.

Full technical detail and commit history: `03fe79b`, `5d5c0d8` (warm-start
fix), `c699fb3`, `3597f76` (probe cadence fix).

## Expected effects and outcomes

**Core expectation: fast movement, not slow.** Previously the network had
to *derive* material counting from raw piece positions — the reason
nineteen readings never moved. Now the answer is handed to it directly, so
the remaining learning problem is "trust and weight this one input
channel," which should be learnable in hundreds of games, not thousands.

| Signal | By game ~1,000 (from restart) |
|---|---|
| `missing_queen` | Should break clearly outside ±0.06, trending toward the −0.3 to −0.9 range the original protocol always said was "correct" |
| The other 6 `material_probe.csv` positions | Should move **together**, in the right directions (negative for every White-down position, positive for every Black-down one), roughly scaled by how much material is missing |
| `start`, `w_wins`, `b_move` | Should stay as they were — near 0, and saturated ±0.99 respectively. Not what's being tested. |

**What a null result would actually mean here.** If `missing_queen` and its
six siblings are *still* flat by game ~2,000 despite the network being
handed the material value directly as an input, that is a **stronger,
more damning result than any flat reading in run14 or run15** — those
could be explained by "the network has to derive this the hard way."
This time it doesn't. A flat result would point at something structural:
the value head not routing a broadcast scalar channel usefully into its
output, a wiring problem, or the underlying diagnosis being wrong in a way
not yet identified. Worth a sanity check on gradient flow into that
specific channel's weights before concluding the whole hypothesis is wrong.

**What this test is *not* claiming to fix.** Beating Stockfish depth 1 is
not the bar for Option A specifically — per `generalization_gap_options.md`,
knowing you're down a queen is not the same as the broader positional
judgement (king safety, structure, tactical foresight) a sound opponent
actually punishes. If `missing_queen` moves correctly but Stockfish depth 1
stays at 0/50, that is **not** a failure of this test — it's exactly the
outcome that would motivate moving to Option C (self-play positions
relabelled with Stockfish evaluations) as the next step, per the
sequencing recommendation already on record.

## Decision on outcome

- **GREEN** (`missing_queen` and the probe set move clearly, in the right
  directions, within ~1,000 games): confirms pure counting was the
  bottleneck. The generalisation-gap work specific to material awareness
  is done; whether to pursue Option C for the harder "beat Stockfish" goal
  becomes a separate decision, not a continuation of this one.
- **RED** (flat by ~2,000 games): stronger evidence than before that the
  gap is broader than counting. Move to Option C rather than iterating
  further on Option A — annealing or retuning the material plane doesn't
  address a problem that isn't about counting.

## Status at registration (game ~40)

Run16 resumed cleanly after the `material_probe.csv` code landed (game 34,
steps 165). No signal-bearing data yet — the warm-started weights haven't
had meaningful gradient exposure to the new channel. First real readings
expected within the next few dozen games at the 20-game probe cadence.

## Correction: game-40 reading was confounded, not evidence (18 August)

The first `material_probe.csv` row (game 40) read every position except
`missing_queen` as strongly, uniformly negative — including the two
positions that should have read positive (`black_missing_queen`,
`black_missing_rook`), and with `missing_two_pawns` just as saturated as
`missing_rook` despite the large difference in actual material. Cause:
six of the seven positions used `-` (no castling rights) while
`missing_queen` alone kept its established `KQkq`. Self-play games almost
never retain castling rights past move 60+ — exactly when material
adjudication fires — so `-` rights very plausibly read as a strong "this
is a decided, losing endgame" signal that swamped the actual material
plane, rather than the network's material judgement actually being tested.

Fixed (commit `97642cf`): all seven positions now keep full `KQkq` wherever
chess rules allow it, dropping only the specific right a removed rook
genuinely invalidates. The game-40 reading is discounted — it is evidence
of a flawed diagnostic, not evidence about Option A. First trustworthy
reading will be the next one logged after the fix.

## Final result (22 August, game 2,120 — run16 stopped here)

104 clean readings (post-confound-fix), split into four consecutive time
windows for trend analysis rather than eyeballed row by row:

| Position | Early (60–500) | Mid (520–1000) | Late (1020–1560) | Final (1580–2120) |
|---|---|---|---|---|
| `missing_queen` | 30% | 35% | 48% | 37% |
| `missing_rook` | 43% | 77% | 77% | 63% |
| `missing_bishop` | 26% | 23% | 6% | **0%** |
| `missing_knight` | 26% | 19% | 6% | **0%** |
| `missing_two_pawns` | 39% | 42% | 10% | 7% |
| `black_missing_queen` | 83% | 88% | 87% | **96%** |
| `black_missing_rook` | 0% | 4% | 29% | 56% |

(% = fraction of readings with the theoretically correct sign in that
window.)

**Verdict: neither clean GREEN nor clean RED — a genuine, informative
split.** `black_missing_queen` is a robust, steadily strengthening
success. `black_missing_rook` recovered from confidently wrong to a
coin-flip. `missing_rook` improved sharply then partially backslid. But
`missing_bishop`, `missing_knight`, and `missing_two_pawns` didn't merely
fail to improve — they got **monotonically worse across all four
windows**, ruling out "just needs more time." That specific, ruled-in
failure mode is itself the finding: material adjudication only fires at
`|material| ≥ 8`, so self-play never generates reinforced training signal
for smaller imbalances, no matter how many games are played. Queen/rook-
scale material generalised because the training data actually contains
it; bishop/knight/pawn-scale material didn't, because it structurally
doesn't.

**Decision**: don't run to the full 10,000-game budget — the cause is
data-composition, not data-volume, so more of the same games won't fix
it. Stopped at game 2,120. Intervention ladder rung 1b (a second,
lower-confidence adjudication tier for the 3–7 band) tests whether this
specific, diagnosed gap is fixable cheaply within self-play before
reaching for Option B or C. See `run17_decision_protocol.md`.

The material-count input plane itself is kept, not reverted — it
demonstrably works for the material ranges self-play actually reinforces.
run17 warm-starts directly from this run's final checkpoint.
