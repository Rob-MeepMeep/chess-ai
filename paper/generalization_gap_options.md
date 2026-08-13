# The Generalisation Gap — Options Going Forward

**Project:** chess-ai
**Phase:** 3 — AlphaZero-style chess agent (HAL-4000)
**Authors:** Rob Kirkland, Ellis Ward
**Date:** 13 August 2026 (written at run15 Gate 3, game ~3,027)
**Status:** Decision document — no option implemented yet. This is a
comparison to weigh, not a recommendation to act on unread.

---

## 1. What Gate 3 established

Run15's intervention (early material adjudication) worked decisively.
Cap-draw share held at 5–7% across 3,000+ games against run14's climb to
77%; win rate vs random hit 90.0% against run14's 23.1% at the same
checkpoint. Full detail in `run15_decision_protocol.md`.

Two things did **not** move, across the entire project, and rung 1 was
never going to touch either of them:

- **`missing_queen` regression**: fourteen readings, two full runs,
  never once left a ±0.06 noise band. The test position — the standard
  starting setup with White's queen removed and nothing else touched — is
  structurally close to unreachable through legal self-play (it would
  require the queen to vanish before any other move is played anywhere on
  the board). It has no near neighbour anywhere in the seed buffer or in
  organic self-play data. The network has never been shown anything shaped
  like it.
- **Stockfish depth 1**: **0 wins, 0 draws, in 200 games**, across run14
  and all three run15 evals, completely flat from step 0 through step
  14,950 — even as the vs-random win rate climbed from ~10% to 90% over
  that same span. Random punishes blunders; depth-1 Stockfish makes
  essentially none. Everything rung 1 fixed was about converting clear
  advantages faster. Beating a sound (if shallow) opponent requires
  positional judgement HAL has not demonstrated anywhere in this project's
  data.

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

## 4. Side-by-side

| | Option A — material plane | Option B — Stockfish pretraining |
|---|---|---|
| Engineering cost | Hours | Days |
| Architecture change | Yes (new run, warm-startable) | No (same input shape) |
| Fixes `missing_queen`? | Likely, directly | Likely, as a side effect of broader fix |
| Meaningfully closer to beating Stockfish d1? | Uncertain — narrow fix | More plausible — imports real evaluation judgement |
| Self-play-from-scratch claim | Survives almost entirely intact | Materially compromised for the value head |
| Reversibility / risk if it doesn't work | Very low — cheap to test and discard | Higher — bigger investment before knowing the outcome |

## 5. A sequencing recommendation (not a decision)

Option A is cheap enough to treat as a diagnostic rather than a
commitment: build it, run it, see what happens to `missing_queen`.

- If it moves sharply toward the expected −0.5 to −0.9 range, that
  confirms the bottleneck really was pure counting — a small, well-scoped
  fix solved it, no need for Option B's larger investment or its bigger
  philosophical cost.
- If it only partially fixes it, or fixes this specific metric while
  other blind spots remain, that is real evidence the gap is broader than
  counting — and would make the case for Option B's larger investment much
  stronger than it is today, on data rather than a guess.

This sequencing doesn't resolve the values question, though, and it isn't
meant to: **how much the "HAL taught itself chess" narrative matters
relative to wanting a stronger engine is a call only Rob can make.** If the
project's identity is genuinely about self-play from first principles, that
argues for stopping at Option A regardless of whether it fully closes the
gap to beating Stockfish. If the goal has shifted toward "build something
that actually plays well," Option B is the more honest bet on getting
there — at the cost of the purity the project has maintained through
fifteen runs so far.
