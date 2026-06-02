# HAL Phase 1: Q-Learning on Tic-Tac-Toe
## A Brief Technical Report

**Project:** chess-ai  
**Phase:** 1 — Reinforcement Learning Foundations  
**Date:** May 2026  
**Authors:** Rob Kirkland, Ellis Ward

---

## 1. Introduction

This report documents Phase 1 of the chess-ai project — an ongoing effort to build a chess-playing agent using reinforcement learning (RL). Rather than beginning with chess directly, Phase 1 uses Tic-Tac-Toe as a controlled environment to establish and validate the core RL pipeline. The game is simple enough to reason about completely, yet rich enough to surface real learning dynamics.

The agent is named HAL.

---

## 2. What We Built

### 2.1 Architecture

The system comprises three components:

**`tictactoe/env.py` — The Environment**  
A complete Tic-Tac-Toe rulebook. The board is represented as a 9-tuple of integers (0 = empty, 1 = X, 2 = O). The environment exposes a standard RL interface: `reset()` returns an initial state; `step(action)` applies a move and returns `(new_state, reward, done, winner)`. It has no knowledge of strategy — it only enforces rules.

**`tictactoe/agent.py` — HAL (Q-Learning Agent)**  
HAL maintains a Q-table: a dictionary mapping board states to an array of 9 Q-values, one per cell. Higher Q-value = HAL's belief that playing that cell is good. HAL selects actions using an epsilon-greedy policy — exploring randomly with probability ε, exploiting his best known move otherwise. After each move, HAL updates his Q-table using the Bellman equation:

```
Q(s, a) ← Q(s, a) + α · (r + γ · max Q(s', a') − Q(s, a))
```

where α is the learning rate, γ is the discount factor, r is the reward received, and s' is the resulting state.

**`train_ttt.py` — Training Loop**  
Two instances of HAL (HAL-X and HAL-O) play each other for N episodes in self-play. After each game: the winner receives +1, the loser receives −1, draws receive 0. Epsilon decays each episode from 1.0 toward a configured minimum. Every 1,000 games, win/draw rates are logged to CSV. Every 5,000 games, a demo game is rendered to the terminal showing HAL's board evaluations in real time.

### 2.2 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate (α) | 0.1 |
| Discount factor (γ) | 0.9 |
| Epsilon start | 1.0 |
| Epsilon decay | 0.9999 per episode |
| Epsilon minimum | 0.02 (final run) |
| Training episodes | 200,000 (final run) |

---

## 3. Experiments

### 3.1 Initial Run — Baseline Failure

The first training run (50,000 episodes, ε_min = 0.05) produced a surprising result: HAL-X won ~85% of games, with draws below 10%, and every demo game played the exact same sequence of moves.

Diagnosis revealed two compounding bugs:

**Bug 1 — Broken tie-breaking.** When Q-values are equal (as they are for unseen states, initialised to 0), Python's `max()` consistently returns the first element. HAL-X always played cell 0; HAL-O always played cell 1. The same game was played on repeat — no exploration occurred.

**Bug 2 — Q-value propagation failure.** Because the same positions were visited repeatedly, Q-values never propagated back to the opening. The starting board showed all-zero Q-values even after 50,000 games.

**Fix:** Tie-breaking was changed to randomly select among all actions sharing the maximum Q-value. This immediately produced diverse games and proper exploration.

### 3.2 Second Run — With Fix (50,000 episodes, ε_min = 0.05)

After fixing tie-breaking, draws climbed from ~13% (early random play) to ~47% by episode 50,000. HAL discovered 4,519 unique board states and stopped finding new ones around episode 15,000 — indicating the full reachable game tree had been explored.

### 3.3 Final Run — Extended Training (200,000 episodes, ε_min = 0.02)

Extending training to 200,000 episodes with a lower epsilon floor showed clear improvement through episode ~40,000, at which point draws plateaued at approximately 50%. No meaningful improvement was observed from episode 40,000 to 200,000.

---

## 4. Results

### 4.1 Training Performance (final run, episodes 195,000–200,000)

| Outcome | Rate |
|---------|------|
| X wins | ~32% |
| O wins | ~18% |
| Draws | ~50% |

### 4.2 Evaluation Performance (ε = 0, pure exploitation, 1,000 games)

| Outcome | Count | Rate |
|---------|-------|------|
| X wins | 319 | 31.9% |
| O wins | 184 | 18.4% |
| Draws | 497 | 49.7% |

### 4.3 Comparison Against Baselines

| Agent | Draw Rate |
|-------|-----------|
| Two random agents | ~13% |
| HAL (trained, ε=0.02) | ~50% |
| HAL (trained, ε=0) | ~50% |
| Perfect play (theoretical) | 100% |

HAL significantly outperformed random play. The near-identical results at ε=0.02 and ε=0 suggest the gap is not caused by exploration noise but by unconverged Q-values in the opening.

---

## 5. Key Learnings

**L1 — Tie-breaking matters.**  
A single-line bug — defaulting to the first action when Q-values are tied — caused catastrophic co-evolutionary collapse. Both agents played identical games for 50,000 episodes and learned almost nothing. In larger state spaces this kind of silent failure would be much harder to detect.

**L2 — Q-value propagation is slow.**  
With α=0.1 and γ=0.9, rewards propagate backwards one step per episode. Opening-position Q-values received diluted signals (γ^6 ≈ 0.53 of the terminal reward) and never converged even after 200,000 games. This is a fundamental property of discounted Q-learning over multi-step games.

**L3 — More training has diminishing returns.**  
Once HAL explored all 4,520 reachable states (~15,000 episodes), no new states were discovered. The subsequent 185,000 episodes produced no measurable improvement. Training time is only valuable while new experience is being gained.

**L4 — Epsilon noise is not the main problem.**  
Reducing ε_min from 5% to 2% produced only a modest improvement in draw rate. The evaluation at ε=0 confirmed that the remaining non-draws stem from unconverged Q-values, not exploration noise.

**L5 — Self-play can create local equilibria.**  
Because both agents learn simultaneously, they can converge to a stable but suboptimal strategy pair. HAL-X learned to exploit HAL-O's specific defensive patterns rather than discovering universally optimal play.

---

## 6. Conclusion

Phase 1 successfully demonstrated the core RL pipeline: an environment, an agent, a training loop, and evaluation. HAL improved from 13% draws (random baseline) to ~50% draws — a genuine learning result.

The ceiling reached here is inherent to tabular Q-learning on this problem. Extending the same approach to chess is not feasible: chess has approximately 10^44 reachable positions compared to Tic-Tac-Toe's 4,520. A Q-table would require more memory than exists in the observable universe.

Phase 2 will replace the Q-table with a neural network — allowing HAL to *generalise* across positions he has never seen, rather than memorising each one individually. The training loop, reward structure, and self-play architecture established in Phase 1 carry forward unchanged.

---

## Annex A — Training Hyperparameter History

| Run | Episodes | ε_min | Peak Draw Rate | Notes |
|-----|----------|-------|----------------|-------|
| 1 (broken) | 50,000 | 0.05 | ~10% | Tie-breaking bug; HAL-X wins 85% |
| 2 (fixed) | 50,000 | 0.05 | ~47% | Post tie-breaking fix |
| 3 (final) | 200,000 | 0.02 | ~53% | Plateaued at ep ~40,000 |

---

## Annex B — Q-Value Convergence Sample

States with highest learned Q-values (HAL-X, final checkpoint):

```
State: (1, 2, 1, 2, 1, 0, 0, 2, 0)  — X one move from winning
Q-values: [0, 0, 0, 0, 0, 0, 1.0, 0, 1.0]   (cells 6 and 8 win)

State: (1, 1, 2, 2, 1, 0, 0, 0, 2)  — X can win or must block
Q-values: [0, 0, 0, 0, 0, 0, -0.44, 1.0, 0]  (cell 7 wins; cell 6 loses)

State: (0, 0, 0, 0, 0, 0, 0, 0, 0)  — Empty board
Q-values: [0, 0, 0, 0, 0, 0, 0, 0, 0]         (no preference learned)
```

The empty board Q-values confirm that opening strategy was not learned — HAL's endgame knowledge is strong but opening knowledge is absent.

---

## Annex C — Selected Training Log (Final Run)

| Episode | X Win % | O Win % | Draw % | Epsilon | States |
|---------|---------|---------|--------|---------|--------|
| 1,000 | 57.3% | 30.6% | 12.1% | 0.905 | 2,683 |
| 5,000 | 57.2% | 31.2% | 11.6% | 0.607 | 4,402 |
| 10,000 | 49.7% | 32.2% | 18.1% | 0.368 | 4,513 |
| 15,000 | 44.1% | 24.3% | 31.6% | 0.223 | 4,520 |
| 20,000 | 42.8% | 19.5% | 37.7% | 0.135 | 4,520 |
| 30,000 | 34.7% | 19.6% | 45.7% | 0.050 | 4,520 |
| 40,000 | 32.0% | 17.4% | 50.6% | 0.020 | 4,520 |
| 100,000 | 32.8% | 16.0% | 51.2% | 0.020 | 4,520 |
| 200,000 | 34.8% | 17.8% | 47.4% | 0.020 | 4,520 |

*Draw rate plateaued at ~50% from episode 40,000 onward. State count stabilised at 4,520 from episode ~15,000.*
