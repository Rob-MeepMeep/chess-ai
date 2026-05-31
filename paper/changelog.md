# chess-ai Changelog
**Authors:** Rob Kirkland, Ellis Ward

A record of significant decisions, config changes, and architectural pivots across the project.

---

## Phase 0

### [14cc773] FastAPI service skeleton
- Random move endpoint at `/move`, health check at `/health`
- Proves Electron ↔ Python pipeline

---

## Phase 1 — Tic-Tac-Toe Q-learning

### [549d589] Initial Q-learning agent (HAL)
- Q-table with epsilon-greedy exploration
- Self-play training loop

### [8803805] Fix tie-breaking in action selection
- **Problem:** `max()` always returned first element on equal Q-values — HAL always played cell 0
- **Fix:** Randomly choose among all actions sharing the max value

---

## Phase 2 — Connect Four DQN (HAL-3000)

### [a5b4b18] Connect Four neural network agent
- CNN with policy + value heads, Double DQN, experience replay
- Action masking for full columns

### [572481b] Fix DQN training instability
- **Problem:** Q-value divergence — loss reached 321,169 by episode 10,500
- **Fix:** Lower learning rate (1e-3 → 1e-4), tighten gradient clipping (max_norm=10 → 1)

---

## Phase 3 — Chess AlphaZero-style (HAL-4000)

### [cb66a49] Phase 3 architecture — encoding layer, move index, residual network
- Board encoding: 55 planes × 8×8
- Move encoding: UCI ↔ index 0–4095
- ResNet: 128 channels, 8 blocks, policy + value heads

### [1862169] MCTS — tree search guided by neural network
- UCB selection, virtual loss for batched simulations
- Dirichlet noise at root for training exploration

### [7e2a374] Replay buffer, agent, logger
- Trajectory buffer storing (state, policy, outcome)
- Rich logging: win rates, game lengths, opening sequences, MCTS snapshots

### [27d2cb1] Training loop complete
- Self-play loop with checkpointing and resume

### [6bc8e31] Batched MCTS, faster checkpoints, better logging
- Performance optimisations for M3 hardware

### [4d56f3a] Game 1090 milestone — HAL integration into chess-trainer
- FastAPI `/move` endpoint updated to use trained ChessAgent
- chess-trainer hal-integration branch: HAL playable + Watch mode

### [c2ee1f6] Resignation logic — fix reward signal collapse
- **Problem:** All games hitting move cap — no decisive outcomes in replay buffer
- **Fix:** Material resignation (`RESIGN_MATERIAL=9`) and value resignation (`RESIGN_THRESHOLD=-0.95`)

### [b725872] Game-end distribution logging, value regression test, training refinements
- `end_reasons.csv` — per-100-game breakdown of how games ended
- Value head regression test in `eval_chess.py` — 4 canonical positions
- `MAX_GAME_MOVES` 100 → 150
- `resign_cause` tracking (material vs value)

### Run 3 abandoned — resign logic never fired (game 400, 2026-05-30)
- **Problem 1 (resign streak):** `mat_from_mover` was calculated per-side-to-move,
  so with white down 6 the streak went 1 (white's turn) → 0 (black's turn, black is UP 6)
  → 1 → 0... and never reached 3. All 400 games hit the 150-move cap.
- **Fix:** Use `abs(mat) > RESIGN_MATERIAL` — checks the absolute imbalance regardless
  of whose turn it is. Winner determined by which side has more material at resignation.
- **Problem 2 (logger key mismatch):** `end_reasons.csv` showed all zeros because window
  dict keys were plural (`"cap_draws"`, `"checkmates"`) but `end_reason` strings are
  singular (`"cap_draw"`, `"checkmate"`). The `if key in self._window` check never matched.
- **Fix:** Renamed window keys to match end_reason strings exactly.

### Run 2 abandoned — draw-collapse confirmed (game 200, 2026-05-29)
- **Problem:** Value head completely draw-collapsed. All 200 games ended as cap-draws,
  giving every position an outcome of 0. Value head MSE loss already minimised at zero
  — gradient flat, value head learns nothing. MCTS evaluations meaningless.
- **Evidence:** K+Q vs lone K evaluated at -0.07 (expect near +1). 0% win rate vs random.
- **Fix for Run 3:** Lower `RESIGN_MATERIAL` 9 → 5 (rook) and `RESIGN_CONSECUTIVE` 5 → 3
  to generate decisive games much earlier in training.

### Model scaled up for MacBook Pro M5 Pro (2026-05-29)
- `N_CHANNELS` 128 → 160, `N_BLOCKS` 8 → 10
- `N_SIMULATIONS` 50 → 100
- Hardware: MacBook Air M3 16GB → MacBook Pro M5 Pro 24GB

### Run 4 concluded — 2300 games, 12,155 steps (2026-05-31)
- Value head active: value resigns grew from 10% → 48-58% of games
- Cap draws essentially eliminated: 13% → 1%
- Average game length: 84 → 43.7 moves
- Loss: 3.78 → 3.61 (windowed average)
- 8 checkmates in 2300 games
- **Problem 1 — black wins 55-60% persistently:** policy developed bad white habits
  (king marches to centre) which self-play never punished. Buffer contaminated with
  systematically biased data. HAL lost 22% vs random as white, 15% as black.
- **Problem 2 — resign ends games too early:** with RESIGN_CONSECUTIVE=3, network never
  needed to learn closing technique. Only 8 checkmates in 2300 games — 0% win rate
  vs random at eval because HAL can't deliver checkmate reliably.
- **Root cause of both:** resign thresholds were correctly lowered to bootstrap the
  value head in Run 3/4, but once the value head was active the thresholds should
  have been walked back. They weren't, allowing bad habits to compound.

### Run 5 abandoned — black bias confirmed structural (game 100, 2026-05-31)
- 54B / 31W / 15D in first 100 games with fresh buffer and run4 weights
- Bias present from game 1 — source: plane 48 ("colour to move") allowed the network
  to develop colour-specific associations across run4's 2300 games
- Decision: remove plane 48, start run6 from random weights

### Remove colour plane from encoder (2026-05-31)
- **Problem:** plane 48 (all 1s = white to move, all 0s = black) let the network
  learn colour-specific behaviour. Over run4's 2300 games it associated white with
  losing, causing persistent 55-63% black win rate.
- **Fix:** removed plane 48 entirely. Encoder: 55 planes → 54 planes.
  Network is now genuinely colour-blind — identical positions look identical
  regardless of which colour is to move. Matches true AlphaZero convention.
- First conv layer input channels: 55 → 54 (picked up automatically via N_PLANES)

### Run 6 — start config (2026-05-31)
- Fresh random weights, fresh buffer — no inherited bias
- Encoder: 54 planes (colour plane removed)
- `RESIGN_MATERIAL` = 7, `RESIGN_CONSECUTIVE` = 5 (retained from Run 5)
- Goal: balanced W/B outcomes, more checkmate data, working eval win rate

---

*Updated throughout the project. For full diff history see git log.*
