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

### Selective buffer seeding — curate_buffer.py (2026-06-02)
- **Problem:** Bootstrapping. Early self-play generates noise — both sides random,
  training signal meaningless for the first ~500 games. The value head learns nothing
  useful until the buffer fills with decisive games. Every run wastes the first few
  hundred games re-discovering basic patterns.
- **Solution:** `curate_buffer.py` — reads run6's game log, filters for quality,
  replays surviving games to extract positions, adds canonical endgame positions with
  correct outcomes, saves as a seed buffer for the next run.
- **Filters:** game_num >= 3000, n_moves 20-100, decisive end_reasons only
  (material_resign, checkmate). Excludes cap draws and overconfident short games.
- **Canonical positions:** K+Q vs K, K+R vs K in multiple orientations with
  ground-truth outcomes ±1. Repeated 200× each so the value head gets direct signal
  for positions it rarely sees in self-play.
- **Result (run6 at game 4850):** 808 games × ~46 moves = 37,335 positions +
  1,600 canonical = 38,935 positions (78% of 50k capacity).
- **Usage:** Run `venv/bin/python3 curate_buffer.py` after a training run,
  then set `BUFFER_LOAD = "checkpoints/run7_seed_buffer.pt"` in train_chess.py.

### Run 7 — buffer seeding validated, cap draw collapse (game 1200, 2026-06-06)
- b-move value reached −0.950 at game 990 — first time value head correctly evaluated K+Q vs K endgame. Proved buffer seeding works: Run 7 bootstrapped meaningfully from game 1 rather than spending hundreds of games re-discovering resign thresholds.
- **Problem:** Cap draws scored 0.0 outcome regardless of material balance. A game ending at 50-move cap with White up a queen still scored 0/0 draw for both sides — directly contradicting the value head's resign signal. After game 1000 a cap draw spike (likely caused by early value head confusion) flooded the buffer with 0.0 outcomes and collapsed training.
- Stopped game 1200. Fix needed before Run 8: assign soft outcomes (±0.8) to cap draws where one side is materially ahead.

### [d7b6f83] Prepare Run 8 — cap draw fix, 25% canonical partition, seed buffer
- **Cap draw soft outcome:** cap draws now score +0.8 / −0.8 when `abs(material) > 3` at the move cap, rather than 0.0. Prevents material-advantage positions from being labelled as draws and contaminating the value head.
- **Tiered replay buffer:** permanent partition (25% of each batch) reserved for canonical endgame positions — K+Q vs K, K+R vs K in multiple orientations. These positions are never evicted as the rolling buffer cycles. Ensures the value head always sees ground-truth outcomes even as self-play distribution shifts.
- Seed buffer built from Run 7 games (decisive only, move 20–100) + canonical positions.

### [05e20df] Fix critical MCTS backup sign — value stored from wrong perspective
- **Problem (affects Runs 1–7):** During MCTS backup, the value estimate was negated *after* being accumulated into `node.W`. The correct AlphaZero convention is to negate *before* storing, so each node records value from the perspective of the player who moved to reach that node. With the sign inverted, every visit to a node updated its statistics in the wrong direction — the policy head was effectively learning to prefer the worst moves.
- **Effect:** All prior runs trained on backwards value signal. This explains 0% win rate vs random across seven runs despite the value head converging to reasonable regression values.
- **Fix:** flip sign before `node.W += value`, not after. One-line change with run-wide consequences.

### [3c5270a] eval_chess.py — background-safe eval for concurrent training
- Added `--cpu` flag to force CPU device during eval, freeing the MPS backend for a concurrent training loop.
- Added `--regression-only` flag for a ~5-second value head sanity check that can run safely at any point without interrupting training.

### Run 8 — first wins, Geometry Trap (game 2010, 10,160 steps, 2026-06-07)
- Win rate vs random: 4% (game 560) → 12% (game 1000) → 18% (game 1500) → 12% (game 2010, regression trough).
- ~240 checkmates across the run. Notable: Fool's Mate (game 823), Scholar's Mate (game 830), Na5# (game 1886), Bg7# (game 1509).
- Loss: 3.093 (new low across all runs). b-move: −0.921 (new high). w-wins: +0.048 (oscillating throughout — see Geometry Trap).
- **Geometry Trap identified:** w-wins oscillated near zero throughout Run 8 while b-move converged cleanly. Root cause: `RESIGN_MATERIAL` cuts games before White ever has the material advantage to trigger decisive win signals. White's winning endgames are underrepresented in the buffer; training data asymmetry prevents the value head from learning White wins.
- vs Stockfish depth 1: 0% throughout. No draws.

### [93057d3] Run 9 infrastructure — RESIGN_MATERIAL=3, diverse endgame buffer
- `RESIGN_MATERIAL` lowered 7 → 3. A 3-pawn material advantage (minor piece up) now triggers resignation, generating White win signals much earlier in games.
- Diverse K+Q vs K endgame seeding: 256 positions at varied board locations with ground-truth ±1 outcomes added to the permanent buffer partition. Gives the value head direct, repeated signal for the most important endgame the agent needs to learn to convert.
- Seed buffer built from Run 8 games (decisive, games 1000–2010, both colours).

### Run 9 — Geometry Trap resolved, f2f3 local minimum (game 1000, 15,135 steps, 2026-06-07)
- **Geometry Trap resolved:** w-wins reached +0.9990 at game 410. First time value head correctly evaluated a White-winning K+Q vs K endgame. The RESIGN_MATERIAL=3 change fixed the training data asymmetry — White winning positions now enter the buffer.
- First White wins vs random: 8% as White at game 1000 (was 0% across all prior runs).
- Black regression: 24% → 12% as Black; 4 losses to random — policy in mid-transition between old and new training distribution.
- **f2f3 opening local minimum:** greedy eval opened `1. f2f3` in 21–70% of games throughout. Self-play partners learned to exploit the open diagonal; value signal reinforced f2f3 avoidance in some games, then re-selected it others. Policy converged to a local minimum.
- Loss: 1.6–1.9 (well below Run 8). Zero cap draws. W/B balance 25/25 at game 800.
- Decision: Dirichlet noise at root breaks opening lock-in. Stop Run 9 at game 1000; implement noise for Run 10.

### [0d60be1] Run 10 infrastructure — temperature scheduling, Run 9 seed buffer
- **Temperature scheduling:** first 30 plies of each self-play game use stochastic multinomial sampling over the policy distribution. From ply 31 onward, greedy argmax. Hard binary cutoff — no decay schedule. Increases training diversity in the opening and early middlegame without sacrificing endgame quality.
- `N_SIMULATIONS` halved: 200 → 100. Doubles game throughput; MCTS quality sufficient at this stage.
- `ReplayBuffer` capacity expanded: 50,000 → 75,000 rolling positions.
- Checkpoint and log paths updated: `run10_hal_chess.pt`, `logs/run10/`.
- Seed buffer built from Run 9 games 800–1000 (decisive, both-colour wins).

### [b88d867] Policy mirroring for canonical encoding, value resign winner fix
- **Policy mirroring:** MCTS was computing prior probabilities using canonical (always-White-to-move) board encoding, but storing them in the original board's move index. For Black, the canonical flip means the move indices no longer correspond to the unflipped board's legal moves — policy priors were targeting wrong squares. Fixed with `mirror_policy()` applied in `mcts.py` and `train_chess.py` to remap Black's policy vector back to the original board's coordinate space.
- **Value resign winner:** when a resign was triggered by the value head (not material balance), the winner was determined by material balance anyway. This could assign the wrong colour as winner in positions where the value head and material signal disagreed. Fixed to use the value head's sign for value-triggered resigns.

### [b14eab7] Replay buffer sample — explicit ratio, edge case guard, shuffle
- **Buffer sample crash:** `random.sample(rolling_buffer, k)` raised `ValueError` when the rolling buffer had fewer positions than requested sample size during early training. Fixed with explicit guard: if rolling buffer undersized, draw all available positions.
- Batch composition now uses explicit ratio: 75% rolling, 25% permanent. Shuffle applied to combined batch before training. Prevents gradient order bias from consistent position ordering.

### Run 10 — eval improvements (game ~5400)
- **`hal_move_noisy_at(n_sims)`:** new eval move function that applies Dirichlet noise at the MCTS root while keeping greedy (argmax) move selection. Breaks determinism in Stockfish eval games without degrading play quality — consecutive eval games see different opening exploration without temperature-driven quality loss.
- **`add_noise` parameter in `choose_move()`:** allows noise injection independently of the `greedy` flag. Previously noise was tied to training mode only; now eval can selectively add noise regardless of greedy/stochastic setting.
- Stockfish depth 1 eval now uses `hal_move_noisy_at(200)` — 200 simulations, greedy + noise.
- **First draws vs Stockfish depth 1:** 3 draws at game ~5400 eval. First time in the project HAL has not lost every game against a rated engine.

### Run 10 — Scholar's Mate trap resolved (game ~5960 → game 7500)
- **Game ~5960:** 17/25 eval games as White vs Stockfish depth 1 ended in Scholar's Mate against HAL — greedy policy consistently opened f2f3/g2g3/g4, exposing the h5-e8 diagonal. Trap worsened across the run as Black self-play policy learned to exploit it and reinforced the signal from the other side.
- **Self-play game 6030 (45,105 steps):** HAL as Black delivered Scholar's Mate in 8 plies (`b1c3 a7a5 f2f3 c7c6 g2g3 e7e5 g3g4 Qh4#`). Generated direct −1 training signal on the f3/g3/g4 sequence for White.
- **Game 7500:** 0/25 Scholar's Mates. The greedy first-move shifted from f2f3 to a2a3 (66% of games) — a2a3 doesn't expose the diagonal. f3/g3/g4 sequence received enough −1 signal from self-play losses and Stockfish defeats to drop out of greedy selection.

### Run 10 — first White draw vs Stockfish depth 1 (game ~9050, 60,270 steps)
- Eval game 59 (as White): 174 moves, drawn by 50-move rule. Opening: `f2f3 e7e5 e2e4 g8e7 c2c4 e7c6 d2d3 f8b4`. Dirichlet noise in `hal_move_noisy_at` escaped a2a3 to f2f3, but HAL followed with central development (e4/c4/d3) rather than the Scholar's Mate continuation.
- First time HAL has drawn as White against any rated engine. (First ever project draw vs Stockfish was Black draw, game ~5400.)
- Win rate vs random at same eval: 14% (trough from negative start value bias, −0.057). Not structural collapse — same trough pattern seen at game ~2600 before recovery.

### [1241f60] Disable Stockfish depth 3/5 eval pending depth 1 threshold
- Depth 3 and depth 5 matchups commented out of `eval_chess.py`. All results were 0/0/25 (no wins, no draws) — no signal, only eval time cost.
- Re-enable thresholds: add depth 3 when HAL reaches 20% W/D vs depth 1; add depth 5 when 20% W/D vs depth 3.
- Current Stockfish eval: depth 1 only (`for depth in [1]`).

### Run 11 — buffer curation pipeline (2026-06-11)
- **`extract_buffer_candidates.py`:** extracts FEN positions from Run 10 decisive games (game 2000+) where abs(material) ≥ 5 at plies 8–28 and the material-advantaged side won. Output: `paper/buffer_candidates.json` (~300 candidates).
- **External agent review:** all 300 candidates reviewed by Claude — 193 accepted, 107 rejected with rationale. Output: `paper/buffer_candidates_reviewed.json`.
- **`curate_buffer.py` updated for Run 11:** loads 40 static canonical positions + 256 diverse K+Q vs K + 193 agent-reviewed mid-game positions = 489 permanent positions total. The 193 mid-game positions are the fix for the missing_queen oscillation seen throughout Run 10 — they anchor the value head to real material-advantage outcomes in mid-game positions, not just canonical endgames.

### Run 11 — regression logging (2026-06-11)
- **`record_regression()` in `chessai/logger.py`:** evaluates value head on 4 canonical positions (start, w_wins, b_move, missing_queen) and appends to `logs/run11/regression.csv` every 200 games. Gives a continuous curve of value head health through the run instead of requiring manual spot-checks.
- **`train_chess.py`:** `REGRESSION_EVERY = 200` config constant; `logger.record_regression(game_num, agent)` called in main loop.
- Positions match `eval_chess.py` REGRESSION_POSITIONS exactly so logged values are directly comparable to manual eval output.

### Run 11 — eval watcher (2026-06-11)
- **`eval_watcher.py`:** standalone script that polls `logs/run11/games.csv` every 30 seconds and fires `eval_chess.py` whenever a 1500-game boundary is crossed. Run in a second terminal alongside `train_chess.py`. Initialises `last_eval_at` from current game count to avoid catch-up evals when started mid-run. Stops cleanly on Ctrl+C.

### Run 11 — dashboard extended (2026-06-11, 2026-06-12)
- **`dashboard.py`** extended from 4 to 7 CSV types:
  - **`openings.csv`:** first-move distribution over time (stacked bar) + a2a3 lock-in line chart + top-10 opening sequences table. Quantifies the a2a3 → b2b4 transition in Run 11.
  - **`end_reasons.csv`:** material resign vs value resign proportions over time + checkmate rate trend. Shows when the value head becomes the primary resign signal.
  - **`snapshots.csv`:** top-1 MCTS visit share at start position over time (policy confidence curve) + which move was top-1 per snapshot window + latest canonical position table.
- CSV upload crash fixed: `on_bad_lines='skip'` + try/except for malformed rows (e.g. Run 8 games.csv with mid-run column count change).
- `st.dataframe()` `use_container_width` deprecation fixed: updated to `width='stretch'`.

### Run 11 — complete (2026-06-24)
- **5389 games, steps 64,925–91,775.** Best result: 18% vs random (matching game 3000 peak); best White: 20% (trough phase, step 87,375); best Black: 24% (peak phase, step 79,875).
- **Findings:** missing_queen oscillation persists with transient spikes (−0.714 at step 90,780, project record). Cap draw rate (~80% vs random) is structural — confirmed as the primary remaining gap. White policy improved measurably across the run independent of oscillation phase. Start position bias developed late (−0.1345 at final eval). vs Stockfish depth 1: max 8% W/D; no formal draws.
- **Run 11 closed.** Run 12 plan: remove RESIGN_MATERIAL (Stage 2 resign), add endgame conversion positions to permanent buffer (K+Q vs K+P, K+R vs K), monitor start position bias.

### Run 11 — Fool's Mate visualisation (2026-06-24)
- Game 3721: Scholar's/Fool's Mate in 5 half-moves (`1. d4 f6 2. e4 g5 3. Qh5#`) visualised as HTML board replay artifact.
- Board-by-board progression of all 6 positions with animated attack diagonal and pulsing king-in-check highlight. Saved to Desktop as `game3721.html`.

### Run 11 — Windows portability notes (2026-06-24)
- `paper/windows_port_notes.md` added: assessment of changes required to run chess-ai on the 3XS Edge RX (AMD Ryzen 5 9600X / AMD RX 9070 XT 16GB / Windows 11).
- Key changes: device detection block in `train_chess.py` and `eval_chess.py` (add CUDA/ROCm path before MPS fallback); no `caffeinate` on Windows; Windows Stockfish binary; venv activation command. AMD GPU options: torch-directml (easiest) or WSL2+ROCm (best performance).

---

*Updated throughout the project. For full diff history see git log.*
