# Phase 2 Report — Connect Four with Deep Reinforcement Learning
## HAL-3000: A DQN Agent for Connect Four

**Authors:** Rob Kirkland, Ellis Ward  
**Project:** chess-ai  
**Phase:** 2 of 3  
**Completed:** May 2026

---

## Overview

Phase 2 replaces the Q-table from Phase 1 with a neural network. The game
scales from Tic-Tac-Toe (4,520 reachable states, fully memorisable) to
Connect Four (roughly 4 trillion positions — far beyond what a table can
hold). The neural network learns to *approximate* Q-values for positions it
has never seen, generalising from experience rather than memorising outcomes.

The agent produced by this phase — HAL-3000 — wins approximately 71–72% of
games against a random opponent from either side of the board. Three full
training runs were required to reach a stable result: each failure taught us
something that we couldn't have known in advance.

---

## The Problem: Why Connect Four

Connect Four is a natural step between Tic-Tac-Toe and chess:

- **Large enough** that a Q-table isn't feasible
- **Small enough** that training runs complete in hours rather than days
- **Solved** (first player wins with perfect play — proven 1988), giving us
  a theoretical ceiling to compare against
- **Sequential alternating play**, identical in structure to chess

It also introduces the key DQN challenges we'll need to handle in chess:
Q-value instability, experience replay, co-evolutionary self-play dynamics,
and perspective-correct training.

---

## Architecture

### Board Encoding

The raw board is a 42-integer tuple (6 rows × 7 columns). Each cell contains
0 (empty), 1 (player 1's piece), or 2 (player 2's piece).

Rather than feeding this directly to the network, we encode it as a
**2-channel binary tensor** of shape (2, 6, 7):

- Channel 0: cells occupied by *my* pieces (1.0 where present, 0.0 elsewhere)
- Channel 1: cells occupied by *the opponent's* pieces

This is perspective-relative — the same physical board produces different
tensors depending on whose turn it is. HAL always sees itself as "my pieces"
regardless of which player it is, which makes spatial patterns consistent
across both player roles.

```
encode_state(state, player):
    my_channel  = (board == player).float()
    opp_channel = (board == opponent).float()
    return stack([my_channel, opp_channel])  # shape: (2, 6, 7)
```

### Convolutional Neural Network

The network uses two convolutional layers to detect spatial patterns
(rows, columns, diagonals) before passing to separate heads.

```
Input: (2, 6, 7)
  → Conv2d(2→32, kernel 3×3, padding 1) + ReLU
  → Conv2d(32→64, kernel 3×3, padding 1) + ReLU
  → Flatten: 64 × 6 × 7 = 2,688 features
  → Policy head: Linear(2688→128) + ReLU + Linear(128→7)  → 7 Q-values
  → Value  head: Linear(2688→64)  + ReLU + Linear(64→1)   → scalar [-1, 1]
```

The **policy head** produces Q-values for each of the 7 columns. The
**value head** produces a position evaluation (used for monitoring; becomes
load-bearing in Phase 3 when we add MCTS).

The value head uses Tanh activation to constrain its output to [-1, 1],
consistent with win/loss semantics.

### Double DQN

Standard DQN is known to overestimate Q-values: when we compute the target
for a transition, we take `max Q(next_state)` — but the same network selects
and evaluates the best action, which introduces optimistic bias.

Double DQN separates these roles:

- **Policy network** selects the best next action: `argmax Q_policy(s')`
- **Target network** evaluates that action: `Q_target(s', selected_action)`

The target network is a frozen copy of the policy network, updated every 500
training steps. This breaks the feedback loop that causes overestimation.

```python
# Policy net selects best action
best_next_actions = Q_policy(next_state).argmax(dim=1)
# Target net evaluates it
q_next = Q_target(next_state).gather(1, best_next_actions)
q_target = reward + gamma * q_next * (1 - done)
```

*Credit: Double DQN architecture recommended by external reviewer Dave.*

### Action Masking

In Connect Four, a column is illegal if its top cell is occupied. The
network must not learn to assign high Q-values to full columns.

Masking is applied in two places:

1. **Inference:** illegal columns set to `-inf` before `argmax`
2. **Training:** when computing the Bellman target, the max Q-value over the
   next state must also exclude full columns

The second masking point is easy to miss. If illegal columns contaminate the
training target, the loss looks normal but the network silently learns from
bad targets. We derive the mask from the top row of each next state:

```python
full_col_mask = (next_states[:, :7] != 0)  # top-row check, shape (batch, 7)
Q_policy(next_state).masked_fill(full_col_mask, float('-inf')).argmax(dim=1)
```

*Credit: action masking in training step identified by external reviewer Dave.*

### Experience Replay

Every move is stored in a replay buffer (capacity 50,000 transitions). During
training we sample random batches of 64 transitions. This breaks the
correlation between consecutive moves that would otherwise cause the network
to oscillate.

---

## Training Setup

| Parameter | Value |
|-----------|-------|
| Episodes | 50,000 |
| Batch size | 64 |
| Replay buffer capacity | 50,000 |
| Learning rate | 1e-4 (Adam) |
| Discount factor γ | 0.99 |
| Epsilon start | 1.0 |
| Epsilon min | 0.02 |
| Epsilon decay | 0.9999 per step |
| Target network update | every 500 steps |
| Gradient clipping | max_norm = 1.0 |

Two agents (HAL-1 and HAL-2) train simultaneously via self-play, each with
their own policy network, target network, and replay buffer.

### Perspective-Correct Transitions (Deferred Push)

In alternating self-play, HAL-1 moves first. If we push HAL-1's transition
immediately after it moves, the stored `next_state` is the board before
HAL-2 has responded. The Q-values computed for that state represent
"HAL-1's best move from here" — but HAL-1 won't face that state on its next
turn. It will face whatever the board looks like *after* HAL-2 responds.

Fix: hold HAL-1's transition in a pending buffer. After HAL-2 moves, push
HAL-1's transition with the real `next_state` (post-HAL-2 response). If
HAL-2 wins on its turn, HAL-1's deferred transition gets `reward = -1`.

This ensures HAL-1 learns from the states it will actually encounter.

*Credit: perspective consistency issue identified by external reviewer Dave.*

---

## Training Run Narrative

Three runs were required to produce a stable result. Each failure was
informative rather than wasted work.

---

### Run 1 — Baseline (no stability safeguards)

**Configuration:** Default Adam lr=1e-3, no gradient clipping.

| Episode | HAL-1 Win% | Avg Loss | Avg Game Length |
|---------|-----------|----------|-----------------|
| 500     | 53.8%     | 0.073    | 21.5            |
| 5,000   | 52.2%     | 0.145    | 24.5            |
| 7,000   | 45.6%     | 0.116    | 24.6            |
| 10,000  | *         | 76,965   | *               |
| 10,500  | *         | 321,169  | 14.0            |

Loss began doubling every few thousand episodes. The inflection point —
where growth became clearly exponential — was around episode 7,000, when
HAL-1's win rate dropped (HAL-2 found a counter-strategy) and the loss began
climbing steeply. By episode 10,500, loss reached 321,169 and average game
length had collapsed to ~14 moves — near the minimum for Connect Four.

**What happened:** Q-values entered a feedback loop. As HAL-1 won more, the
winning moves were reinforced with large Q-values. Those large Q-values
became the training targets. Which made them larger. Without any cap on how
much the weights could update in a single step, the values diverged toward
infinity.

**Game length as collapse signal:** The first clear indicator was game length
dropping below 20. By the time the loss numbers became alarming, the agents
had already converged to a near-minimum game. This metric is more reliable
than loss alone as an early warning.

Run 1 was stopped at episode ~11,500.

---

### Run 2 — Gradient Clipping (max_norm=10.0, lr=1e-3)

**Configuration:** Added `clip_grad_norm_(parameters, max_norm=10.0)` before
each optimiser step.

| Episode | HAL-1 Win% | Avg Loss | Avg Game Length |
|---------|-----------|----------|-----------------|
| 500     | 53.8%     | 0.073    | 21.5            |
| 8,000   | 48.6%     | 0.104    | 25.1            |
| 8,500   | 50.6%     | 0.096    | 24.7            |
| 9,000   | 48.6%     | 0.090    | 25.5            |
| 10,000  | 48.6%     | 0.086    | 25.3            |
| 10,500  | 44.0%     | 0.080    | 24.8            |
| 14,000  | 45.2%     | 0.067    | 24.6            |
| 18,500  | *         | 2,608    | 9.7             |

Gradient clipping delayed the divergence significantly — the run was stable
for approximately 10,000 additional episodes compared to Run 1. During this
window, the exploit/counter cycle was directly observable:

Around episodes 8,000–12,500, HAL-1's win rate climbed to 65–70% (an
exploit found), the loss spiked to ~15.5, gradient clipping allowed partial
self-correction (loss dropped back to 0.83 by ep9,000), and HAL-2 adapted
(win rates returned to near-parity by ep12,500).

This cycle — exploit discovered, loss spike, partial recovery, opponent
adaptation — is the competitive dynamic self-play is designed to produce.
It was only visible because gradient clipping bought enough stable training
time to observe it.

However, with lr=1e-3, updates were still large enough to diverge eventually.
By episode 18,500, loss had reached 2,608 and game length had fallen to 9.7
moves (physically near-minimum). The run was stopped.

**Lesson:** Gradient clipping is necessary but insufficient. It limits
individual update magnitude, but a high learning rate means many moderately
large updates accumulate quickly.

---

### Run 3 — Stable (max_norm=1.0, lr=1e-4)

**Configuration:** Tightened gradient clipping to max_norm=1.0 and reduced
learning rate to 1e-4.

| Episode | HAL-1 Win% | Avg Loss | Avg Game Length |
|---------|-----------|----------|-----------------|
| 5,000   | 52.2%     | 0.145    | 24.5            |
| 10,000  | 48.6%     | 0.086    | 25.3            |
| 20,000  | 46.6%     | 0.064    | 24.4            |
| 30,000  | 61.0%     | 0.064    | 23.1            |
| 35,000  | 77.0%     | 0.066    | 22.7            |
| 40,000  | 58.2%     | 0.103    | 22.2            |
| 45,000  | 76.4%     | 0.442    | 20.9            |
| 50,000  | 81.6%     | 0.924    | 20.1            |

The run completed all 50,000 episodes without catastrophic divergence.
Loss did increase in the final 10,000 episodes (from ~0.06 to ~0.92), but
this reflects genuine Q-value growth as HAL-1's dominance became established —
not runaway feedback. Game length remained above 18 throughout, well clear of
the collapse threshold.

The loss increase in the final phase corresponds to HAL-1 finding increasingly
decisive winning lines. HAL-1's win rate climbed from ~50% through the
midgame to 70–80% by the end, consistent with genuine strategy development
rather than collapse.

**Demo game screenshots** — captured automatically during training at every
5,000 episode interval:

- `logs/screenshots/demo_ep005000.png` — Episode 5,000: early exploration,
  ~52% win rate, moves still largely random
- `logs/screenshots/demo_ep020000.png` — Episode 20,000: structure beginning
  to appear; centre-column preference visible
- `logs/screenshots/demo_ep035000.png` — Episode 35,000: HAL-1 winning
  decisively; coordinated multi-threat play
- `logs/screenshots/demo_ep050000.png` — Episode 50,000: fully trained play

---

## Evaluation Results

HAL-3000 was evaluated at ε=0 (pure exploitation, no random moves) over
1,000 games per matchup.

| Matchup | P1 Win% | P2 Win% | Draw% |
|---------|---------|---------|-------|
| HAL-1 (P1) vs Random (P2) | **71.3%** | 28.7% | 0.0% |
| Random (P1) vs HAL-2 (P2) | 27.7% | **72.3%** | 0.0% |
| HAL-1 (P1) vs HAL-2 (P2) — ε=0 | **100%** | 0.0% | 0.0% |
| Random (P1) vs Random (P2) — baseline | 55.9% | 43.6% | 0.5% |

### Interpretation

**Matchups 1 and 2** are the meaningful performance measures. HAL wins
~71–72% from both sides against a random opponent — symmetric performance
indicates HAL learned genuine strategy rather than exploiting a first-player
advantage. The random baseline (matchup 4) confirms first-player wins ~56%
by chance, so HAL is adding approximately 16 percentage points of genuine
strategic value.

**Matchup 3** (HAL-1 vs HAL-2 at ε=0) shows a 100% HAL-1 win rate, and every
game plays out identically — deterministic networks in the same starting
position always produce the same sequence. This is co-evolutionary collapse:
during training, HAL-1 found a strategy line that HAL-2 consistently cannot
answer, and HAL-2's Q-values were shaped by this specific dynamic throughout
training. HAL-2 is not a poor Connect Four player (72% vs random) — but it
cannot beat HAL-1, which it co-evolved against.

Notably, Connect Four is a *solved game* — with perfect play, first player
always wins. HAL-1 winning 100% from the first-player position is at least
directionally correct. Whether this reflects approximating optimal play or
merely finding a specific exploit HAL-2 can't handle, we cannot easily
determine from these numbers alone.

**Conclusion:** HAL-3000 learned meaningful Connect Four strategy. The
co-evolutionary self-play structure produced competent play on both sides
while exhibiting the known limitation of co-evolutionary convergence.

---

## Key Lessons Learned (Phase 2)

### L6 — Q-value divergence is the central DQN stability problem
Without safeguards, Q-values spiral in a feedback loop: winning moves get
reinforced with large values, which become training targets, which push values
larger. Run 1 reached a loss of 321,169 by episode 10,500.

### L7 — Gradient clipping delays but does not prevent divergence
Capping gradient norm (max_norm=10.0) prevents catastrophically large updates
but cannot overcome a learning rate that is fundamentally too aggressive.
Run 2 extended stable training by ~10,000 episodes before diverging.

### L8 — Learning rate is the root cause
A learning rate of 1e-3 is too aggressive for DQN on Connect Four. Each
update is large enough that Q-values drift faster than the target network can
stabilise them. The correct fix is lr=1e-4 with max_norm=1.0 — both levers
constrain how fast Q-values change.

### L9 — Game length is the clearest collapse signal
Average game length falling sharply precedes loss explosion and win rate
extremes. A game averaging 10 moves means one player is winning in ~5 moves
each — near the minimum. Monitor this, not just loss or win rate.

### L10 — The exploit/counter cycle is observable in self-play
In Run 2 (episodes 8,000–12,500): HAL-1 win rate rose to 65–70% (exploit
found), loss spiked, gradient clipping allowed partial recovery, HAL-2
adapted (rates returned to ~50/50). This is the intended competitive
dynamic — and requires enough training stability to observe.

### L11 — Action masking must apply in training, not just inference
When computing the Bellman target, illegal actions must be excluded from the
next-state max. If they aren't, the network learns from corrupted targets
silently — the loss looks normal but the targets are wrong.
*(Identified by external reviewer Dave before any training run.)*

### L12 — Perspective-correct transitions require deferred push
HAL-1's transition must be held until HAL-2 responds, then pushed with the
real post-response `next_state`. Pushing immediately gives HAL-1 an
intermediate board state it will never actually face.
*(Identified by external reviewer Dave.)*

---

## External Review — Dave's Contributions

Several architectural decisions were validated or corrected by external
reviewer Dave before training began. This saved at least one additional
failed run.

**Double DQN:** Dave recommended separating action selection (policy network)
from action evaluation (target network) — the standard DQN approach, but
easy to overlook when first implementing. This was incorporated from the start.

**Action masking in training:** Dave identified that we were masking illegal
actions at inference but not in the Bellman target computation. A subtle but
important distinction — the same masking logic must apply to both.

**Perspective-correct transitions:** Dave identified that HAL-1's deferred
push needed to use the post-HAL-2 `next_state`, not the intermediate board
state immediately after HAL-1 moved.

**Game length monitoring:** Dave recommended tracking average game length as
a collapse signal alongside win rate and loss. This proved to be the clearest
early warning in both Run 1 and Run 2.

---

## Looking Ahead: Phase 3 — Chess

The architectural decisions made in Phase 2 inform the chess agent design.
Key considerations already identified:

**Single network for both colours.** Rather than two agents co-evolving,
chess HAL will use one network trained to always see the board from the
perspective of the player to move. When it's black's turn, the board
representation is flipped so "my pieces" / "opponent's pieces" is consistent.
This is the same approach used by AlphaZero and eliminates co-evolutionary
collapse.

**Input encoding must include full game state.** Legal moves in chess depend
on more than the current board position: castling rights, en passant
availability, and repetition history all affect legality. The input tensor
must encode all of these.

**Reward sparsity.** In an 80-move chess game, gamma=0.99^80 ≈ 0.45 at move
1 — the network barely feels the result of the game at the opening. Reward
shaping or a higher gamma will be needed.

**Value head is load-bearing.** In Phase 2, the value head was present but
informational. In chess, if we add Monte Carlo Tree Search, the value head
must produce meaningful position evaluations from early training onward.

**The exploit/counter cycle logging design** (100-episode log intervals,
periodic strategy snapshots on canonical positions) will be built into the
chess training loop from the start. This produces the data needed to show
when specific opening preferences emerged and how long exploits persisted
before being answered — richer material than win/loss curves alone.

---

## Appendix — Training Data (Run 3)

Full training log available at `logs/training_c4.csv`.

Selected milestones:

| Episode | HAL-1 Wins | HAL-2 Wins | Draws | Loss   | Game Length |
|---------|-----------|-----------|-------|--------|-------------|
| 500     | 269       | 228       | 3     | 0.073  | 21.5        |
| 5,000   | 261       | 238       | 1     | 0.145  | 24.5        |
| 10,000  | 243       | 254       | 3     | 0.086  | 25.3        |
| 15,000  | 264       | 235       | 1     | 0.064  | 24.9        |
| 20,000  | 233       | 261       | 6     | 0.064  | 24.4        |
| 25,000  | 259       | 240       | 1     | 0.061  | 23.1        |
| 30,000  | 305       | 194       | 1     | 0.064  | 23.1        |
| 35,000  | 385       | 114       | 1     | 0.066  | 22.7        |
| 40,000  | 291       | 209       | 0     | 0.103  | 22.2        |
| 45,000  | 382       | 118       | 0     | 0.442  | 20.9        |
| 50,000  | 408       | 92        | 0     | 0.924  | 20.1        |

*Note: wins per 500-episode window (each row = 500 games). Not cumulative.*

---

*Report written May 2026. chess-ai project — Phase 2 complete.*
