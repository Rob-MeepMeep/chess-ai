# The Generalisation Gap — Options Going Forward

**Project:** chess-ai
**Phase:** 3 — AlphaZero-style chess agent (HAL-4000)
**Authors:** Rob Kirkland, Ellis Ward
**Date:** 13 August 2026 (written at run15 Gate 3, game ~3,027)
**Status:** Decision document — no option implemented yet. This is a
comparison to weigh, not a recommendation to act on unread.

---

## 1. What Gate 3 established (updated 18 August, game 6,840)

Run15's intervention (early material adjudication) worked decisively.
Cap-draw share held at 5–8% across 6,800+ games against run14's climb to
77%; win rate vs random hit 90.0% at Gate 3 and has since **reached a
perfect 100.0% (50/50, zero draws, zero losses)** at the most recent
checkpoint (step 29,950). There is no headroom left in this benchmark —
rung 1 has fully saturated what it can teach against a blundering
opponent. Full detail in `run15_decision_protocol.md`.

Two things still have **not** moved, across the entire project, and the
extra ~4,000 games and doubled training steps since Gate 3 have only
strengthened the case that rung 1 was never going to touch either of them:

- **`missing_queen` regression**: nineteen readings now, two full runs,
  never once left a ±0.06 noise band (latest: +0.0124, +0.0158, −0.0136,
  +0.0056, +0.0081). The test position — the standard starting setup with
  White's queen removed and nothing else touched — is structurally close
  to unreachable through legal self-play (it would require the queen to
  vanish before any other move is played anywhere on the board). It has no
  near neighbour anywhere in the seed buffer or in organic self-play data.
  The network has never been shown anything shaped like it.
- **Stockfish depth 1**: **0 wins, 0 draws, in 300 games**, across run14
  and all five run15 evals, completely flat from step 0 through step
  29,950 — across three widely-spaced checkpoints *after* Gate 3 alone
  (14,950 → 22,450 → 29,950), with zero trend in either direction, even as
  the vs-random win rate climbed from ~10% to a perfect 100% over that
  same span. This is no longer "hasn't moved yet" — it's a flat line
  sampled five separate times across a doubling of training. Random
  punishes blunders; depth-1 Stockfish makes essentially none. Everything
  rung 1 fixed was about converting clear advantages faster. Beating a
  sound (if shallow) opponent requires positional judgement HAL has not
  demonstrated anywhere in this project's data.

**One practical consequence of the 100% ceiling**: the case for simply
letting run15 keep grinding toward its full 10,000-game budget before
deciding is now much weaker than it was at Gate 3. With conversion against
random fully maxed out, additional self-play games are likely to look very
similar to the ones already collected — diminishing returns on exactly the
axis rung 1 was designed to improve. The seed-buffer argument from earlier
(more decisive games → a better buffer via `curate_buffer.py`'s move-count
filter) still holds in principle, but most of its value has probably
already been banked. This is a reasonable point to start on one of the
options below rather than wait for game 10,000.

The working diagnosis (see chat log, 13 August): this is very likely a
**scale and inductive-bias problem**, not a bug. AlphaZero trained on
~44 million self-play games; we are at ~3,000–10,000. A convolutional
value head has no built-in mechanism for summing material across the
board — it has to discover that as an emergent pattern from enormous
incidental exposure, and at our scale that exposure may simply never
accumulate enough for the pattern to crystallise. Rung 2 of the original
intervention ladder (annealing `perm_ratio`) does not address this — it
only changes which data gets memorised harder, not whether the network is
capable of deriving a general counting rule from it.

## 2. Option A — Material-count input plane

**What it is**: add one input channel to the encoder — a broadcast scalar
plane carrying material balance (current player's perspective, clipped and
scaled), computed the same way `train_chess.py`'s adjudication rule already
computes it. `N_PLANES` goes from 54 to 55; `ChessNet`'s first conv layer
picks up the new size automatically since it imports `N_PLANES` directly.

**Scope** (full detail in chat, 13 August):
- `chessai/encoder.py`: `_material_balance()` helper + one new plane,
  ~15 lines.
- `chessai/model.py`: no code change, one docstring update.
- A new run (`run16`), started either fresh or **warm-started** from
  run15's checkpoint — copy every weight whose shape is unaffected (all
  residual blocks, both heads) directly, and for the one changed layer,
  copy the 54 existing input-channel weights across and zero-initialise
  the 55th. Warm-starting means the network begins identical to run15 and
  only has to learn to *use* the new feature, not relearn everything else.

**Cost**: hours, not days. No new data pipeline, no new training loop.

**What it plausibly fixes**: the specific, narrow capability of knowing
material balance in unfamiliar configurations — directly targets
`missing_queen`.

**What it does not obviously fix**: whether HAL can beat Stockfish depth 1.
Knowing you're down a queen is not the same as understanding king safety,
pawn structure, or tactical foresight — the broader positional judgement
that a sound opponent actually punishes. This is a plausible partial fix,
not a guaranteed path to the harder goal.

**Philosophical cost**: small and scoped. It replaces exactly one derived
arithmetic operation — the equivalent of giving a mental-arithmetic student
a calculator for one step while they still work out the reasoning around
it. Every other part of HAL's play — strategy, tactics, when to convert,
how to attack — is still entirely self-taught through self-play. The
project's core claim ("HAL learned to generalise from self-play") stays
true everywhere except this one narrow, well-understood blind spot.

## 3. Option B — Supervised value-head pretraining on Stockfish evaluations

**What it is**: before (or interleaved with) self-play, train the value
head on a large set of positions labelled with real Stockfish evaluations
(cheap and instant to generate — no self-play compute involved). Centipawn
scores get squashed to the network's [-1, +1] range (e.g.
`tanh(cp / 400)`), and the network learns to imitate Stockfish's judgement
directly, before self-play/MCTS sharpens tactics on top of an already-
competent value head.

**Scope**: materially larger than Option A.
- A position-sampling strategy (self-play games, random legal-ish
  positions, or an existing open game database).
- A batch pipeline running Stockfish over that set (already installed —
  used by `eval_chess.py` — but running it at volume is new infrastructure).
- A new supervised training script, separate from `train_chess.py`'s RL
  loop, plus the calibration work to get the label scale right.
- Some decision about how self-play training resumes afterward (fine-tune
  on top of the pretrained value head, most likely).

**Cost**: days of engineering, plus compute for generating and training on
the labelled set, before any self-play resumes.

**What it plausibly fixes**: this is the more promising path for the
*harder* goal specifically. Stockfish's evaluation function encodes real
chess understanding — material, king safety, structure, activity — not
just counting. This is the established, recognised technique other
from-scratch chess NN projects (Leela Chess Zero's early bootstrapping,
Maia) have used to solve exactly this kind of gap. If the actual target is
"give HAL a real chance against a sound opponent," this is the path that
plausibly gets there; Option A plausibly does not.

**Philosophical cost**: substantial, and worth being honest about. HAL's
foundational sense of what a good position looks like would originate from
imitating Stockfish's judgement, not from anything HAL discovered through
its own self-play experience. Self-play would still refine play afterward,
but the core evaluation knowledge traces back to an external, hand-tuned
expert system. This is meaningfully closer to "the self-play project
didn't get there on its own, so we borrowed a real engine's understanding"
than Option A is. The "HAL taught itself chess" claim would no longer hold
for the value head's foundational knowledge, even if the policy head and
later refinement remained self-play-driven.

## 4. Option C — Relabel self-play positions with Stockfish evaluations

**What it is**: the middle path, raised in chat on 13 August. The core
problem with the self-play value signal isn't just "no material counting"
— it's that every position in a game trajectory gets the *same* final
outcome label (z), regardless of whether that position was five moves in
or fifty. That's inherently noisy for anything but the tail end of a
decisive game. Instead of importing synthetic or externally-sourced
positions (Option B), take the positions HAL's own self-play *already*
produces — nothing invented, nothing sourced elsewhere — and have
Stockfish evaluate each one individually, at real depth (not the weak
depth-1 setting used for the eval opponent). Use that per-position
evaluation as the value target instead of, or alongside, the whole-game
outcome.

**Scope**:
- A new script (extends the `curate_buffer.py` pattern rather than
  replacing it): replay `games.csv` move lists with python-chess, sample
  positions per game (every Nth ply, or a handful at random — not every
  ply; that's expensive and the positions within one game are highly
  correlated), query Stockfish for a centipawn score, convert via
  `tanh(cp / 400)` to the network's [-1, +1] range.
- **The real design fork — full replacement vs. blend**:
  - *Full replacement*: the sampled position's target becomes the
    Stockfish eval outright, discarding z for that position.
  - *Blend*: `loss = α · mse(v, z) + (1−α) · mse(v, stockfish_eval)`,
    letting self-play outcome keep a voice. A refinement worth
    considering: weight by distance from the game's end — z is reliable
    near the finish and progressively noisier further back, so early/mid
    positions could lean more on the Stockfish eval while late positions
    lean more on z. α becomes a dial, not a switch.
- **No architecture change** — same 54-plane encoder, same network shape.
  This is a training-signal change, not a model change, so it can be
  applied directly to run15's existing checkpoint via continued
  fine-tuning. No warm-start surgery required, unlike Option A.

**Cost**: medium — more than Option A (a new relabeling script, a change
to how the training loss is computed), less than Option B (no separate
synthetic-position pipeline, no architecture change, reuses data already
being generated). The new cost is wall-clock: evaluating a sampled subset
of positions across thousands of games at real Stockfish depth is real
compute, though far less than evaluating every ply of every game.

**What it plausibly fixes**: broader than Option A. This attacks the root
noisy-label problem directly, across the entire diversity of positions
self-play naturally produces — not just material counting. It plausibly
improves the value head's positional judgement generally (king safety,
structure, activity — whatever Stockfish's evaluation captures), which
makes it the strongest of the three candidates for genuinely narrowing the
gap to beating Stockfish depth 1, while remaining grounded entirely in
positions HAL's own self-play discovered.

**What it doesn't fix**: still can't produce positions self-play
structurally cannot reach — the exact `missing_queen` FEN still won't
arise from real games no matter how good the labels on reachable positions
get. If `missing_queen` stays the only generalisation probe, it may show
little movement even if overall judgement improves substantially — this
is the strongest argument yet for building the diverse held-out test set
proposed earlier, since a single unreachable position is a poor detector
of real but partial progress.

**Philosophical cost**: real, but more moderate and more controllable than
Option B. The *what* — which positions HAL ever learns from — stays 100%
self-play. The *how* — the value target for some fraction of positions —
is Stockfish-influenced, and the degree is a tunable dial (α) rather than
a binary switch. A modest blend favouring self-play (α ≈ 0.7–0.8) is a
noticeably smaller compromise than full replacement, and arguably smaller
than it first sounds — it doesn't change what HAL is fed as input at all
(unlike Option A), only refines the quality of the training signal for
experience HAL already generated itself.

## 5. Side-by-side

| | A — material plane | B — Stockfish pretraining | C — relabel self-play positions |
|---|---|---|---|
| Engineering cost | Hours | Days | Medium — script + loss change |
| Architecture change | Yes (new run, warm-startable) | No | No — fine-tunes run15 directly |
| Positions HAL learns from | 100% self-play | Synthetic / external | 100% self-play |
| Value signal source | Self-play outcome (unchanged) | Stockfish | Stockfish, self-play outcome, or a blend (tunable) |
| Fixes `missing_queen`? | Likely, directly | Likely, as a side effect | Uncertain on this specific metric — see caveat above |
| Closer to beating Stockfish d1? | Uncertain — narrow fix | More plausible | Most plausible — fixes the noisy-label problem broadly |
| Self-play-from-scratch claim | Survives almost intact | Materially compromised | Partially compromised, tunable via α |
| Reversibility / risk | Very low | Higher — bigger investment before knowing the outcome | Moderate — cheap to try a light blend first |

## 6. A sequencing recommendation (not a decision)

Given the stated goal is genuinely to beat Stockfish, not just to get a
Lichess number, I'd weight this differently than a pure diagnostic
ordering:

- **Option A** is still worth doing first and cheaply, purely as a
  diagnostic — it isolates whether pure counting was ever the issue,
  in hours, at very low cost either way.
- **Option C**, specifically a light blend (α favouring self-play), is
  the strongest candidate for the actual stated goal. It doesn't require
  throwing away run15's checkpoint or architecture, it's grounded entirely
  in HAL's own experience, and it directly targets the noisy whole-game-
  label problem rather than one narrow symptom of it. This is where I'd
  actually expect the biggest real improvement in playing strength.
- **Option B** (full synthetic pretraining) becomes worth its larger cost
  only if C's lighter touch turns out insufficient — i.e., if even
  Stockfish-grounded self-play positions can't get the value head to a
  place where HAL competes with a sound opponent.

This still doesn't remove the values question — a light blend in Option C
is a smaller compromise than full Option B, but it's still real, and it's
still Rob's call how much of "HAL taught itself chess" is worth preserving
against wanting to actually beat Stockfish. The sequencing above is about
cost and evidence order, not about pre-deciding that trade for you.
