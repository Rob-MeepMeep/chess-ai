# Phase 3b Proposal — Architecture Scaling and Hardware Transition

**Authors:** Rob Kirkland, Ellis Ward  
**Project:** chess-ai  
**Phase:** 3b (continuation of Phase 3)  
**Status:** Proposal — June 2026  
**Trigger:** Planned hardware upgrade to AMD RX 9070 XT desktop (16GB VRAM, 32GB DDR5)

---

## 1. Where We Are

Phase 3 has been running on a MacBook Pro M5 Pro (24GB unified memory, MPS backend)
since its first run. After twelve training runs we have:

- A working AlphaZero-style agent (MCTS + residual network) trained entirely from self-play
- **Best performance:** 26% win rate vs random (Run 10), 18% in Run 12 (in progress)
- All major training bugs resolved: MCTS backup sign, cap draw contamination, geometry trap,
  colour bias, policy mirroring, value resign
- A mature infrastructure: tiered replay buffer, permanent endgame partition (745 positions),
  Dirichlet noise, regression logging, eval watcher
- A clear ceiling: the current architecture (160 channels, 10 blocks, 100 MCTS simulations)
  has likely approached its practical learning ceiling on this hardware

Run 12 is the final run on this architecture. Its purpose is to confirm the Stage 2 resign
transition (value-only) and test the endgame conversion buffer (K+R vs K, K+Q vs K+P). When
it closes, the natural next move is to scale up.

---

## 1b. Observed Constraints in the Current Architecture

Run 12 (in progress, game 2090+) has been a controlled experiment: the same 160ch/10-block
network, same 100 simulations, but with two targeted fixes (Stage 2 resign, expanded endgame
buffer). The results clarify where the ceiling is and why.

### The cap draw structural gap

The most persistent problem across Runs 9–12 is conversion failure. In eval vs random, HAL
draws ~76–84% of games — not because the positions are balanced, but because HAL wins material
and then cannot force a result within the 200-move eval cap. In self-play training, 40–50% of
games hit the 150-move cap with clear material imbalances.

This has two consequences:
1. **Slow learning.** Cap-draw games receive soft outcomes (±0.8) rather than decisive ones (±1.0).
   The value head receives ambiguous feedback on positions it should clearly evaluate as wins or losses.
2. **Self-reinforcing.** As the buffer fills with cap-draw games, the value head is trained to
   evaluate endgame-like positions as ±0.8, which suppresses resignation, which produces more cap draws.

The K+R vs K and K+Q vs K+P permanent buffer positions (added in Run 12) directly target this.
Early results (game 2000) show no measurable improvement in cap draw rate yet — the canonical
positions need more time to generalise to the varied king placements in real games.

### b_move oscillation and its cause

The value head's evaluation of K+Q vs K (Black to move) oscillates with a ~400-game period
throughout Run 12, with dip troughs at games ~1000, ~1400, ~1677 and recoveries at ~1200, ~1600,
~2000. The cause: when the rolling buffer is in a draw-heavy phase, the cap-draw soft outcomes
(±0.8) pull the value head away from the ±1.0 canonical signal. The permanent positions fight
back and win, producing the recovery.

This oscillation is manageable — b_move stays above the 0.95 resign threshold throughout — but
it indicates the permanent buffer weight (25% of each batch) is not quite enough to anchor the
value head against the rolling buffer's influence. **Proposed fix for Run 13:** increase
`PERM_FRACTION` from 0.25 to 0.33 in `chessai/replay.py`.

### Signal quality vs model capacity

The honest question about the current ceiling is: is it capacity-limited or signal-quality-limited?

**Evidence for capacity-limited:**
- Win rate vs random has plateaued at 18–26% across Runs 9–12 despite training for 10,000+ games
- Opening, middlegame, and endgame patterns compete for the same limited representational space —
  the network oscillates between mastering one and forgetting another
- vs Stockfish depth 1: maximum 10% W/D (5 draws in Run 10) from a ~1100 ELO opponent with
  principled positional play the small network cannot represent

**Evidence for signal-quality-limited:**
- The cap-draw problem contaminates 40–50% of self-play training positions with ambiguous outcomes
- 100 simulations per move produces coarser policy targets than 200 simulations; subtle move
  quality differences don't separate reliably at this resolution
- Opening policy lock-in (a2a3 in Run 10, h2h3 in Run 12) suggests the policy is converging
  on local optima that more diverse training signal would break

**Most likely:** both contribute, in proportions we cannot determine without the experiment.
Phase 3b addresses both simultaneously. If the new architecture trajectory is steep (crossing
26% within 2,000 games), capacity was the binding constraint. If the ramp is slow, signal
quality matters more and further sim-count or buffer changes are the next lever.

### Opening policy lock-in

Despite Dirichlet noise in training, greedy eval consistently converges to a single first move:
a2a3 (Run 10), h2h3 (Run 12). The noise diversifies training games effectively, but the policy
distribution is too peaked to survive greedy (argmax) selection at eval time. This is not a
bug — it reflects a genuine policy preference — but it caps eval quality by making HAL's
opening entirely predictable. A larger network with more representational capacity should
maintain a more distributed policy, making this less likely to recur.

---

## 2. The Hardware Transition

**New machine:** 3XS Edge RX desktop  
**CPU:** AMD Ryzen 5 9600X  
**RAM:** 32GB DDR5  
**GPU:** AMD RX 9070 XT — 16GB GDDR6 dedicated VRAM  
**Storage:** 2TB M.2 SSD  
**OS:** Windows 11

This is a meaningful upgrade in every dimension that matters for training:

| | MacBook Pro M5 Pro | 3XS Edge RX |
|---|---|---|
| GPU memory | 24GB unified (shared) | 16GB dedicated |
| GPU type | Apple MPS | AMD RDNA 4 |
| RAM | 24GB (shared w/ GPU) | 32GB DDR5 (separate) |
| Training | ~35s/game (100 sims) | TBD — expect significantly faster |

The 24GB unified figure for the M5 Pro is misleading — CPU and GPU share that pool, so
in practice the GPU has less available than the headline suggests, and memory bandwidth
is shared. The RX 9070 XT has its own dedicated 16GB with its own bus.

**Backend:** PyTorch with ROCm on WSL2 (Ubuntu). AMD's ROCm stack exposes the GPU as
`"cuda"` to PyTorch — a two-line device detection change in `train_chess.py` and
`eval_chess.py` covers it. Everything else in the codebase is backend-agnostic.

---

## 3. The Architecture Change

### Current architecture

```
Input:        54 planes × 8×8
Tower:        10 residual blocks, 160 channels
Policy head:  4096 move logits (all from-to combinations)
Value head:   single scalar, tanh activation
MCTS sims:    100 (training), 50 (eval vs random), 200 (vs Stockfish)
Batch size:   256
Replay buffer: 75,000 rolling + 745 permanent
```

### Proposed architecture

```
Input:        54 planes × 8×8  (unchanged)
Tower:        20 residual blocks, 256 channels
Policy head:  4096 move logits  (unchanged)
Value head:   single scalar, tanh  (unchanged)
MCTS sims:    200 (training), 100 (eval vs random), 400 (vs Stockfish)
Batch size:   512
Replay buffer: 200,000 rolling + 745+ permanent
```

This is the AlphaZero network scale. The parameter count rises from ~4.6M to ~23M —
a 5× increase. All other components (encoder, MCTS logic, replay buffer, training loop)
are unchanged.

### Why this scale

AlphaZero used 256 channels and 20 blocks for chess. That's not coincidence — at this
scale the residual tower has enough representational capacity to learn opening theory,
tactical patterns, and endgame technique simultaneously. The current 160ch/10-block
network is half this in every dimension; we have been asking a smaller model to learn
the same game.

The 16GB VRAM fits this comfortably. Even at batch size 512, the activations for a
256ch/20-block network occupy well under 4GB. The VRAM ceiling for this architecture
is not a practical constraint.

### Simulation count

Doubling MCTS simulations from 100 to 200 is arguably as important as the network
scale. Each simulation is a guided search through possible continuations; more
simulations means:

- Stronger self-play move quality (better training signal)
- More accurate policy targets (MCTS visit counts)
- Better separation between good and bad moves in ambiguous positions

The cost is proportional training time per game, but on dedicated GPU hardware this
should remain manageable.

---

## 4. Transition Strategy — Fresh Weights, Seeded Buffer

### Why not transfer the Run 12 weights

The existing weights are sized for 160 channels and 10 blocks. Neural network weights
cannot be resized — the tensor dimensions are part of the model definition. To use the
new architecture, we start from random initialisation.

### Why this is less painful than it sounds

The failure modes that made early runs (1–7) slow and difficult were almost entirely
code bugs, not bootstrapping noise:

| Problem | Cause | Status |
|---|---|---|
| Draw collapse | Cap draw returning 0.0 outcome | Fixed in code — permanent |
| Geometry trap | RESIGN_MATERIAL asymmetry | Fixed in code (Stage 2 resign) |
| MCTS backup sign | Inverted W accumulation | Fixed in code — permanent |
| Colour bias | Colour plane in encoder | Fixed in code — permanent |
| Policy mirroring | Black policy not mirrored | Fixed in code — permanent |
| Value resign winner | Wrong side declared winner | Fixed in code — permanent |

None of these can recur. The new network starts with all fixes in place.

What will recur, briefly:
- **Bootstrapping phase** (~200–400 games) where the value head hasn't converged on K+Q vs K
- **Opening policy noise** before Dirichlet noise breaks local minima
- **Low win rate** against random in the first 500 games

All of these are manageable and faster to resolve than before, because:

1. **The permanent buffer seeds immediately.** curate_buffer.py can build a new seed buffer
   from Run 12 games the moment Run 12 closes. The new network's first batch includes all
   745 canonical endgame positions — it begins learning K+Q vs K, K+R vs K, and K+Q vs K+P
   from game 1, not game 800.

2. **The Dirichlet noise is already configured.** Opening policy diversity is built in from
   the start; we don't need a separate run to add it.

3. **More simulations = stronger early signal.** At 200 sims, the MCTS policy targets are
   more accurate from the first game. The network bootstraps faster.

Expected trajectory on the new architecture with seeded buffer: past the 26% milestone
(current project best) within 2,000–3,000 games, rather than the ~10,000 games it took
to get there on the current architecture.

---

## 5. Implementation Plan

### Before starting

- [ ] Close Run 12 (after at least game 2000, eval confirms stable value head)
- [ ] Run curate_buffer.py on Run 12 data → `checkpoints/run13_seed_buffer.pt`
- [ ] Set up WSL2 + Ubuntu + ROCm on the Windows machine
- [ ] Verify `torch.cuda.is_available()` returns True under ROCm
- [ ] Install Stockfish Windows build, verify eval_chess.py can call it

### Code changes (minimal)

**chessai/model.py** — parameterise channel count and block count:
```python
# Change defaults
N_CHANNELS = 256   # was 160
N_BLOCKS   = 20    # was 10
```

**train_chess.py / eval_chess.py** — device detection update:
```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

**train_chess.py** — config updates:
```python
N_SIMULATIONS  = 200    # was 100
BATCH_SIZE     = 512    # was 256
BUFFER_CAPACITY = 200000  # was 75000
RUN_NAME       = "run13"
BUFFER_LOAD    = "checkpoints/run13_seed_buffer.pt"
CKPT_LOAD      = None   # fresh weights
```

### Regression positions

The same four canonical positions carry forward unchanged:
- Start position (expect ~0.0)
- K+Q vs K, White to move (expect near +1)
- K+Q vs K, Black to move (expect near -1)
- White missing queen (expect < 0)

Add a fifth if missing_queen oscillation persists: K+R vs K (White to move, expect near +1)
— to confirm the new endgame buffer positions are landing.

---

## 6. What We Are Testing

Phase 3b has two distinct questions:

**Question 1: Does architecture scale matter at this project scope?**
The current agent learned to beat random ~26% of the time from a 160ch/10-block network.
Does doubling both dimensions produce a meaningfully stronger agent in the same number of
self-play games? If the answer is yes, the limiting factor has been model capacity. If the
answer is no (similar ceiling, similar rate), the limiting factor is elsewhere — simulation
quality, training data diversity, or the inherent ceiling of self-play at this scale.

**Question 2: Can we reach a Stockfish depth 1 draw rate above 20%?**
This is the Phase 3 threshold for escalating to depth 3. The current architecture has
reached 10% W/D (5 cap draws, Run 10). A 256ch/20-block network with 200 simulations
is the most direct path to crossing 20%.

---

## 6b. Realistic Expectations — Benefit by Benefit

This section records the honest pre-experiment expectations, so the Phase 3b results can be
evaluated against them rather than against vague optimism.

### Architecture (160ch/10b → 256ch/20b)

The 5× parameter increase gives the network representational capacity it currently lacks:
the ability to learn opening theory, middlegame patterns, and endgame technique in the same
model without oscillating between them. The current network's oscillation behaviour (b_move
cycling, missing_queen instability, opening lock-in) is consistent with a model that has
insufficient capacity to hold all three domains simultaneously.

**Honest uncertainty:** we don't know if the current ceiling is capacity-limited or
signal-quality-limited. If the cap draw noise is the dominant problem, a bigger network
trained on the same noisy signal may just learn the same mistakes more confidently.

**Diagnostic:** if the new architecture crosses 26% vs random within 2,000 games (seeded
start), capacity was the binding constraint. If it takes 8,000–10,000 games to match the
current best, signal quality matters more and further sim-count or buffer changes are the
next lever.

### MCTS simulations (100 → 200)

Arguably the most impactful single change for training signal quality. At 200 sims per move
the policy targets (visit count distributions) are sharper — good moves get visited
proportionally more, ambiguous positions get better resolution. This means every self-play
game generates a cleaner training target.

**Important:** the MCTS loop is partially CPU-bound (Python tree traversal). The GPU only
accelerates the neural network forward passes within each simulation. Speedup from better
hardware applies to the inference fraction only, not the tree traversal.

### Hardware (M5 Pro → RX 9070 XT)

The benefit is less dramatic than the headline specs suggest. The 24GB M5 Pro unified figure
is shared with the CPU — in practice the GPU has less than the headline. The RX 9070 XT has
dedicated 16GB with its own bus. But the MCTS Python overhead doesn't shrink with faster hardware.

**Practical estimate:** a 256ch/20b network at 200 sims on the RX 9070 XT will take similar
wall-clock time per game to the current 100 sims on M5 Pro. The gain is not speed — it is that
you get 2× the simulation quality for the same time budget, and the larger network doesn't
become the bottleneck (the M5 Pro would be memory-bandwidth-constrained at 256ch/20b).

### Buffer (75k → 200k)

More diverse training data, less correlated batches. Primarily dampens the b_move oscillation
observed throughout Runs 10–12. However, with 745 permanent positions in a 200,000-capacity
buffer, permanent positions are only 0.37% of the buffer by count. **Increasing PERM_FRACTION
from 0.25 to 0.33 is more important than the buffer capacity change** — it determines how
aggressively the canonical positions anchor the value head against the rolling buffer's influence.

### Summary table

| Improvement | Mechanism | Realistic benefit |
|---|---|---|
| 256ch/20 blocks | More representational capacity | Crossing the 26% vs random ceiling; stronger Stockfish resistance |
| 200 simulations | Sharper policy targets | Faster convergence, better opening diversity, cleaner training signal |
| AMD GPU | Faster inference | Handles larger network without becoming the bottleneck |
| 200k buffer | Less correlated training | Reduced oscillation amplitude |
| PERM_FRACTION 0.33 | Stronger canonical anchoring | Reduced b_move oscillation specifically |
| Seeded buffer | Endgame knowledge from game 1 | Faster value head convergence than fresh-buffer runs |

**Realistic targets:**

| Metric | Current best | Phase 3b (within 10k games) |
|---|---|---|
| Win rate vs random | 26% (Run 10) | 40–50% |
| vs Stockfish depth 1 W/D | 10% (Run 10) | Cross 20% threshold |
| vs Stockfish depth 3 W/D | 0% | Possible with extended training |

---

## 7. Longer Term

Phase 3b sets up Phase 4: getting HAL a real Lichess ELO rating via a UCI wrapper and
the Lichess bot API. A Phase 3b network trained at AlphaZero scale on dedicated hardware
is a more credible candidate for a rated bot than the current one.

After Phase 3b, if Phase 4 is the goal, the immediate next step is the UCI wrapper —
a text protocol adapter that translates `position`/`go`/`bestmove` commands into calls
to `agent.choose_move()`. The architecture and training changes here don't affect that
path.

---

*Rob Kirkland, Ellis Ward — chess-ai project, June 2026*
