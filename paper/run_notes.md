# chess-ai — Project Run Notes
**Authors:** Rob Kirkland, Ellis Ward  
**Last updated:** 2026-06-04

This document is the persistent context record for the chess-ai project. Any agent or collaborator picking up this project should read this alongside `paper/phase3_architecture.md` and `paper/changelog.md` before touching any code.

---

## Project Summary

Building a chess-playing AI from scratch using deep reinforcement learning. Incremental phases:

- **Phase 0** — Complete. FastAPI skeleton, random `/move` endpoint.
- **Phase 1** — Complete. Tic-Tac-Toe Q-learning (HAL). Report: `paper/phase1_report.md`.
- **Phase 2** — Complete. Connect Four DQN (HAL-3000), 71-72% vs random. Report: `paper/phase2_report.md`.
- **Phase 3** — In progress. AlphaZero-style chess agent (HAL-4000). MCTS + ResNet trained from self-play.
- **Phase 4** — Planned. Lichess ELO rating via UCI wrapper + bot API.

---

## Hardware

| Machine | Chip | RAM | Use |
|---------|------|-----|-----|
| MacBook Air 13" M3 | Apple M3 | 16GB | Run 1 (50 sims). No fan — throttles on long runs. |
| MacBook Pro 14" M5 Pro | Apple M5 Pro | 24GB | Runs 2–4+ (100 sims). Fan = sustained training. |

PyTorch MPS backend used on both. Verify with:
```bash
venv/bin/python3 -c "import torch; print(torch.backends.mps.is_available())"
```

---

## Training Run History

### Run 1 — MacBook Air M3 (complete, game 1090)
- **Config:** 128ch / 8 blocks / 50 sims / 100-move cap
- **Result:** All games hit the move cap. 0% win rate vs random at eval.
- **Root cause:** Reward signal collapse. Every game ended as a cap-draw → every replay buffer position had outcome=0 → value head trained exclusively on draws → learned to output ~0 everywhere → MCTS evaluations meaningless.
- **Milestone:** `milestones/1090_games/` — training.csv, openings.csv, snapshots.csv, eval_results.md committed.
- **Why stopped:** Moved to Pro. Scaled up model. Added resignation logic.

### Run 2 — MacBook Pro M5 Pro (abandoned, game ~200)
- **Config:** 160ch / 10 blocks / 100 sims / 150-move cap
- **New:** Resignation logic added — `RESIGN_MATERIAL=9`, `RESIGN_CONSECUTIVE=5`, `RESIGN_THRESHOLD=-0.95`
- **Result:** Still all cap-draws. Value head regression confirmed draw-collapse: K+Q vs lone K evaluated at -0.07 (expect ~+1).
- **Why abandoned:** Resign thresholds too conservative. Value head already collapsed — -0.95 threshold never reached when network outputs ~0. Material threshold of 9 (queen) too high for early random play to trigger consistently.
- **Fix:** Lower `RESIGN_MATERIAL` 9→5 (rook), `RESIGN_CONSECUTIVE` 5→3.

### Run 3 — MacBook Pro M5 Pro (abandoned, game ~400)
- **Config:** 160ch / 10 blocks / 100 sims / `RESIGN_MATERIAL=5`, `RESIGN_CONSECUTIVE=3`
- **Result:** All 400 games still cap-draws. Resign never fired. Two bugs found.

**Bug 1 — resign streak never accumulates:**
```python
# BROKEN: mat_from_mover flips sign every half-move
mat_from_mover = mat if board.turn == chess.WHITE else -mat
if mat_from_mover < -RESIGN_MATERIAL:   # white's turn: fires; black's turn: resets
    resign_streak += 1
# Result: streak alternates 1/0/1/0 forever, never reaches 3
```
```python
# FIXED: absolute imbalance, independent of whose turn it is
mat_hopeless = abs(mat) > RESIGN_MATERIAL
# Winner at resignation = whichever side has more material
winner = chess.WHITE if _material_balance(board) > 0 else chess.BLACK
```

**Bug 2 — end_reasons.csv all zeros:**
```python
# BROKEN: plural keys in window dict, singular strings from training loop
self._window = {"cap_draws": 0, "checkmates": 0, ...}   # plural
end_reason = "cap_draw"   # singular — never matched, silently discarded
```
```python
# FIXED: keys match end_reason strings exactly
self._window = {"cap_draw": 0, "checkmate": 0, ...}   # singular
```

- **Loss curve:** 6.66 → 4.97 → 3.88 → 3.87 (flatlined). Policy head learned structure; value head collapsed.
- **Seeding for Run 4:** Kept Run 3 weights (policy head not wasted), discarded buffer (draw-poisoned positions removed).

### Run 4 — MacBook Pro M5 Pro (complete, game 2300, 2026-05-30)
- **Config:** 160ch / 10 blocks / 100 sims / both bugs fixed
- **Seeded from:** Run 3 checkpoint (weights only, clean buffer)
- **RUN_NAME:** `run4` → checkpoint at `checkpoints/run4_hal_chess.pt`
- **This is the first run with a correctly functioning reward signal.**

| Game | Result | Moves | Loss | Steps | s/game |
|------|--------|-------|------|-------|--------|
| 10 | D | 150 | 3.7370 | 1940 | 69 |
| 20 | W | 77 | 3.7135 | 1990 | 68 |
| 30 | W | 61 | 3.6648 | 2040 | 59 |
| 40 | W | 41 | 3.6981 | 2090 | 56 |
| 50 | D | 150 | 3.6606 | 2140 | 60 |
| 400 | B | 10 | 3.8698 | 3890 | 40 |
| 450 | B | 15 | 3.6516 | 4140 | 31 |
| 457 | B | 20 | 3.4902 | — | — |
| 530 | W | 130 | 4.0085 | 4505 | 47 |

Resign firing from game 2. Loss trending down: 3.737 → 3.49 by game 457.

**100-game window stats (from end_reasons.csv):**

| Window | Checkmates | Material resigns | Value resigns | Cap draws |
|--------|------------|-----------------|---------------|-----------|
| 0–100 | 0 | 77 | 10 | 13 |
| 100–200 | 0 | 59 | 29 | 12 |
| 200–300 | 0 | 60 | 31 | 9 |
| 300–400 | 0 | 64 | 34 | 2 |
| 400–500 | **1** | 18 | 21 | 3 |

**★ Checkmate 1 — Game 461, White wins, 95 moves (~2026-05-31)**
Final moves: `g5f6 f8d6 f6d6#` — pawn captures f6, bishop retreats to d6, pawn takes d6 checkmate.
Full record in `logs/run4/games.csv`.

**★ Checkmate 2 — Game 858, Black wins, 34 moves (~2026-05-31)**
Final moves: `h3g3 c6a7` — black knight retreats to a7, delivering checkmate.
34 moves is significantly shorter than the first. Black builds an attack and closes it out. Full record in `logs/run4/games.csv`.

**★ Checkmate 3 — Game 996, Black wins, 112 moves (~2026-05-31)**
Final moves: `d1e2 f8g8 f3f2 g8g7 e2d1 g7h7 d1e2 e6f7` — complex endgame, black king walks into a mating net over many moves.
Full record in `logs/run4/games.csv`.

**★ Checkmate 4 — Game 1372, Black wins, 22 moves (~2026-05-31)**
Shortest checkmate yet. White king walked e1→d2→e3→f4→e5 across 11 moves.
Black built a complete cage: knight on f5 covers d4, queen on d7 covers d6, pawn on d5 covers e4, bishop on e6 occupies e6, pawn on g5 covers f4, bishop on g7 delivers check through f6.
Final move: `f4e5 f8g7#` — king steps into the cage, bishop slides to g7, checkmate.
Most sophisticated chess HAL has played — multiple pieces coordinating to trap a wandering king.
Full move record in `logs/run4/games.csv`.

**★ Checkmate 5 — Game 1615, Black wins, 126 moves (~2026-05-31)**
Long complex game. Full record in `logs/run4/games.csv`.

**★ Checkmate 6 — Game 1698, Black wins, 60 moves (~2026-05-31)**
Full record in `logs/run4/games.csv`.

Value resigns now outnumber material resigns for the first time (21 vs 18). The value head is the dominant resign signal — the network has genuine positional conviction, not just material counting.

Average game length: 84 → 77 → 69 → 68 → 60 moves. Consistently shortening each window.

**Run interrupted at game 457** (terminal session closed). Resumed cleanly from game 450 checkpoint. CKPT_LOAD set to None to prevent loading wrong weights on future restarts.

**Run 4 concluded at game 2300.** Data archived at `paper/data/run4/`.

---

### Run 5 — MacBook Pro M5 Pro (abandoned at game 100, 2026-05-31)
- **Config:** 160ch / 10 blocks / 100 sims / RESIGN_MATERIAL=7, RESIGN_CONSECUTIVE=5
- **Seeded from:** Run 4 weights (13,355 steps), fresh buffer
- **Result:** Abandoned after 100 games

**Findings:**
- Black bias confirmed structural — 54B / 31W / 15D in first 100 games (63% black)
- Bias appeared from game 1 with a completely fresh buffer
- Source identified: plane 48 in the encoder ("all 1s = white to move") allows the network
  to develop colour-specific associations. Run 4's 2300 games trained the policy to associate
  "playing white" with losing positions — that learning is baked into the weights.
- Positive finding: loss started at 3.03 at game 100 (vs 3.57 in Run 4) — the run4 weights
  do provide a genuine head start in policy/value quality, just with bias attached.
- Value resigns active from game 1 (29/100 games) — value head came pre-trained.
- 1 checkmate in 100 games.

**Decision: start Run 6 from random weights.** Fresh start removes inherited bias.
Plane 48 (colour indicator) to be removed from encoder — makes network colour-blind,
matching true AlphaZero convention. RESIGN_MATERIAL=7, RESIGN_CONSECUTIVE=5 retained.

---

### Run 6 — MacBook Pro M5 Pro (complete, game 5000, 2026-06-02)
- **Config:** 160ch / 10 blocks / 100 sims / 54 planes (colour plane removed)
- **Fresh random weights, fresh buffer**
- **RESIGN_MATERIAL=7, RESIGN_CONSECUTIVE=5**

**Balance (200 games):** W92/B87/D21 — essentially balanced. No structural bias.
Colour plane removal confirmed working: K+Q vs K white and black to move score identically.

**100-game window stats:**

| Window | W | B | D | Avg loss | Avg length |
|--------|---|---|---|----------|------------|
| 0–50 | 21 | 27 | 2 | 6.264 | 76.3 |
| 50–100 | 24 | 18 | 8 | 6.529 | 87.5 |
| 100–150 | 24 | 18 | 8 | 6.140 | 87.2 |
| 150–200 | 23 | 24 | 3 | 5.879 | 89.6 |

**Value regression test — game 500 (2,460 steps):**

| Position | Value | Expected |
|----------|-------|----------|
| Start | -0.0095 | ~0.0 |
| K+Q vs K (w wins) | -0.0048 | near +1 |
| K+Q vs K (b move) | **-0.0048** | near -1 |
| White missing queen | -0.0097 | < 0 |

Note: K+Q vs K identical for both colours — colour-blindness confirmed empirically.
Compare run4 game 530 where these were asymmetric (-0.020 vs -0.009).

**Eval vs random — game 500 baseline:**
- As white: 0W / 18L / 82D
- As black: 0W / 13L / 87D
- vs Stockfish depth 1/3/5: 0% all depths

**Value regression test — game 1000 (5,110 steps):**

| Position | Value | Expected |
|----------|-------|----------|
| Start | +0.0060 | ~0.0 |
| K+Q vs K (w wins) | -0.0242 | near +1 |
| K+Q vs K (b move) | -0.0251 | near -1 |
| White missing queen | +0.0069 | < 0 |

K+Q positions diverging slightly (-0.0242 vs -0.0251) — value head beginning to distinguish positions even if values still wrong.

**Eval vs random — game 1000:**
- As white: 0W / 14L / 86D (improvement — losing less to random)
- As black: 0W / 19L / 81D (slight regression — watch for bias)
- vs Stockfish depth 1/3/5: 0% all depths

**Value regression test — game 2000 (10,110 steps):**

| Position | Value | Expected |
|----------|-------|----------|
| Start | +0.0014 | ~0.0 |
| K+Q vs K (w wins) | -0.0258 | near +1 |
| K+Q vs K (b move) | -0.0308 | near -1 |
| White missing queen | -0.0107 | < 0 |

Start position essentially correct. White missing queen correctly negative.
K+Q positions diverging with correct ordering (-0.0258 > -0.0308): winning position
scores higher than losing position for first time. Magnitudes still tiny but
directionality correct — genuine value head development from scratch.

**Eval vs random — game 2000:**
- As white: 0W / 9L / 91D (loss rate halved since game 500)
- As black: 0W / 11L / 89D (black regression at game 1000 corrected)
- vs Stockfish depth 1: 1 draw / 49 losses (first ever draw vs Stockfish)
- vs Stockfish depth 3: 1 draw / 49 losses
- vs Stockfish depth 5: 0% draws

**vs Run 4 at game 2000:** Run4 lost 22% as white / 15% as black. Run6 loses 9% / 11%.
Run6 significantly outperforming Run4 at same milestone despite starting from scratch.
Attributed to unbiased encoder, clean training data, better resign thresholds.

**Run 6 concluded at game 5000 (24,960 steps, 2026-06-02)**

Final training window: loss 3.484, avg game length 47.9, cap draws 2%.
Value resigns peaked at 20/50 (game 4600 window), settled at 7-14.

**Value regression test — game 5000:**
All positions near-zero throughout entire run. K+Q vs K never generalised.
Value head is learning in-distribution patterns (value resigns active) but
not general material evaluation. Regression test not useful beyond detecting
draw collapse at this stage of training.

**Eval vs random — game 5000:**
- As white: 0W / 18L / 82D — regression from game 2000's 9%
- As black: 0W / 12L / 88D — flat
- vs Stockfish depth 1/3: 100% losses (game 2000 draws did not persist)

White regression likely caused by policy developing aggressive patterns
(shorter games, more complex positions early) that random occasionally
exploits. Value resign growth in training did not translate to eval wins —
the two measure different things: recognising lost positions vs forcing checkmate.

**Key Run 6 findings for the paper:**
1. Colour plane removal (54 planes) eliminated the structural white bias
2. Value head produces in-distribution extreme values but doesn't generalise to OOD endgames
3. Regression test flatlines while value resigns grow — important gap to document
4. Eval win rate is gated entirely on checkmate delivery, which requires more training
5. Loss 6.26 → 3.48 over 5000 games from scratch — genuine learning curve documented

---

### Run 7 — MacBook Pro M5 Pro (IN PROGRESS, started 2026-06-02)

- **Config:** 160ch / 10 blocks / 200 sims / 54 planes / RESIGN_MATERIAL=7 / RESIGN_CONSECUTIVE=5
- **Fresh random weights, no inherited bias**
- **Seeded buffer:** 40,646 positions from Run 6 (games 3000–5000, decisive only, 20–100 moves)
  + canonical endgame positions (K+Q vs K, K+R vs K × 200 repeats each)
- **Key motivation:** Skip the bootstrapping noise phase — value head gets real signal from game 1

**Bugs fixed during this run:**

*Bug 1 — seed buffer silently skipped on fresh start:*
Buffer load was nested inside the checkpoint existence block. Fresh start → no checkpoint → buffer never loaded. Fixed: buffer loading moved outside checkpoint block, runs independently.

*Bug 2 — accumulated buffer ignored on resume:*
`BUFFER_LOAD` path always won over `BUFFER_PATH`, so restarts kept reloading the seed buffer instead of the accumulated self-play buffer. Fixed: `BUFFER_PATH` takes priority if it exists; `BUFFER_LOAD` only used on genuine fresh start.

**Code changes during run:**
- Tiered replay buffer: `_permanent` partition (never evicted, sampled at 12.5% per batch)
- `curate_buffer.py` routes canonical positions through `add_permanent()`
- 55→54 plane count corrected across encoder.py, model.py, agent.py
- Logger: games.csv added to docstring, perf_interval corrected to 50
- Speed guide updated to M5 Pro with observed timings
- BUFFER_LOAD now prints warning when ignored on resume

**Training windows (50-game, except * = short due to restart):**

| Window | Games | W | B | D | avg loss | val resigns | cap draws | avg len |
|--------|-------|---|---|---|----------|-------------|-----------|---------|
| 50 | 50 | 23 | 21 | 6 | 1.14 | 0 | 6 | 76.3 |
| 100 | 50 | 20 | 26 | 4 | 1.40 | 1 | 4 | 71.6 |
| 150 | 50 | 24 | 24 | 2 | 1.71 | 2 | 2 | 79.1 |
| 200 | 50 | 31 | 17 | 2 | 2.09 | 3 | 2 | 73.8 |
| 250 | 50 | 20 | 26 | 4 | 2.50 | 3 | 4 | 68.3 |
| 300 | 50 | 18 | 24 | 8 | 2.89 | 6 | 8 | 82.1 |
| 350* | 34 | 17 | 15 | 2 | 3.25 | 2 | 2 | 77.6 |
| 400* | 13 | 9 | 4 | 0 | 3.66 | 0 | 0 | 66.1 |
| 450 | 50 | 23 | 24 | 3 | 3.84 | 6 | 3 | 77.0 |
| 500 | 50 | 22 | 23 | 5 | 4.13 | 3 | 5 | 74.8 |
| 550 | 50 | 21 | 29 | 0 | 4.52 | 8 | 0 | 64.0 |
| 600* | 30 | 15 | 12 | 3 | 4.76 | 4 | 3 | 65.8 |
| 650 | 50 | 21 | 21 | 8 | 4.98 | 9 | 8 | 73.4 |
| 700 | 50 | 22 | 23 | 5 | 5.27 | 5 | 5 | 83.2 |
| 750 | 50 | 26 | 21 | 3 | 5.27 | 5 | 3 | 71.3 |
| 800 | 50 | 25 | 21 | 4 | 5.15 | 9 | 4 | 71.5 |

**Loss trajectory:** Rose from 1.14 (game 50) to 5.27 (plateau at games 700–750), then began
declining: 5.15 at game 800. Rising loss is a buffer transition artefact (seed positions had
zero policy labels; as self-play replaced them, policy head faced harder targets). Plateau then
decline is the expected pattern. See key_concepts.md L19.

**Value resign trajectory:** 0→1→2→3→3→6→[2,0]→6→3→8→[4]→9→5→5→9
Brackets are short-window readings. Trend is clearly upward. Represents growing value head
confidence — the network recognises hopeless positions and terminates games early.

**W/B balance:** Consistently near 50/50 across all full-length windows. Colour bias eliminated.

**Value head regression — game ~317 (1,745 steps):**

| Position | Value | Expected |
|----------|-------|----------|
| Start | +0.15 | ~0.0 |
| K+Q vs K (w wins) | +0.006 | near +1 |
| K+Q vs K (b move) | -0.260 | near -1 |
| White missing queen | +0.162 | < 0 |

**Value head regression — game ~500 (2,510 steps):**

| Position | Value | Expected | Change |
|----------|-------|----------|--------|
| Start | -0.09 | ~0.0 | ✓ closer to zero |
| K+Q vs K (w wins) | -0.060 | near +1 | flat |
| K+Q vs K (b move) | -0.456 | near -1 | ✓ +76% magnitude |
| White missing queen | -0.087 | < 0 | ✓ **sign flipped** |

White missing queen sign flipped from +0.162 to -0.087 — value head now correctly rates
material deficit as bad. K+Q vs K (b move) nearly doubled in magnitude. K+Q vs K (w wins)
still flat — winning positions harder to learn than losing ones at this stage.

**Next regression: game 1000.**

**Notable game events:**
- Game 1: First checkmate (W wins, 85 moves) — seed buffer effect, training from game 1
- Game 6: Second checkmate (W wins, 77 moves)
- Games 663: Tactical sequence — Qxf2+ (check), White responded Qxd7+ (counter), Black takes queen, material resign. Shows both sides finding tactical responses.
- Game 685: Bxh6 bishop sacrifice (takes pawn weakened by h7h6 on move 1), later Nxf7+ fork. Piece coordination and weak-square exploitation.
- Games 671, 689: Value resigns at 72–83 moves in complex positional situations (not simple material collapses).
- Game 816: Pawn promotion `b2b1q` — Black promotes pawn to queen in endgame, wins by material resign. First observed promotion in Run 7.

---

## Current Code State

Key config in `train_chess.py`:
```python
N_SIMULATIONS    = 200      # doubled from run6
MAX_GAME_MOVES   = 150
RESIGN_THRESHOLD   = -0.95
RESIGN_CONSECUTIVE = 5
RESIGN_MATERIAL    = 7
RUN_NAME    = "run7"
CKPT_LOAD   = None                                # fresh weights
BUFFER_LOAD = "checkpoints/run7_seed_buffer.pt"  # seed — ignored on resume if run7_replay_buffer.pt exists
```

Key additions since Run 1:
- `chessai/agent.py` — `get_value()` single forward pass for resignation check
- `chessai/replay.py` — `save()`/`load()` buffer persistence; tiered buffer (permanent + rolling)
- `chessai/logger.py` — `games.csv` (full per-game log), `end_reasons.csv`, run-namespaced log dirs
- `curate_buffer.py` — builds seed buffer from prior run data + canonical endgame positions
- `train_chess.py` — RUN_NAME system, CKPT_LOAD/BUFFER_LOAD, tally tracking, buffer priority logic

**Resume command:**
```bash
caffeinate -dims venv/bin/python3 train_chess.py
```

---

## What to Watch in Run 7

| Signal | What it means |
|--------|---------------|
| Value resigns per 50-game window | Core health signal. Growing = value head developing. Stable = plateau. Zero = investigate. |
| Loss declining after plateau | Plateau hit ~5.27 at games 700–750. Decline started at game 800. Watch for continued fall. |
| K+Q vs K regression | Game 1000 target: ≥ ±0.6. If still near zero, value head not generalising to endgames. |
| Cap draws per window | <10 = healthy. >15 = consider loosening resign thresholds. |
| W/B tally | Should stay near 50/50 each 50-game window. Sustained skew = investigate encoder. |
| Checkmates per window | Currently sparse (0–2 per 50 games). Growing checkmate rate = closing technique developing. |

---

## Next Milestones

1. **Game 1000** — value head regression test. Target: K+Q vs K ≥ ±0.6. Then decide full eval.
2. **Full eval vs random** — when regression shows meaningful values. Target: first wins.
3. **Stage 2 resign** — once K+Q vs K reads ±0.9 consistently, remove material resign entirely. Value head takes sole responsibility — allows network to learn material comebacks and closing technique.
4. **Run 8 seed buffer** — run `curate_buffer.py` after Run 7 completes. New buffer will use tiered format (canonical positions in permanent partition).
5. **Phase 4** — UCI wrapper → Lichess bot account → ELO rating.

---

## Working Style Notes

- Rob Kirkland is the project lead, learning ML/Python through this project. Explain before writing — the *why* matters.
- Ellis Ward is the AI research collaborator. All formal documents credited as "Rob Kirkland, Ellis Ward."
- No Claude model names or Anthropic references in committed files.
- Co-Authored-By in commits: `Ellis Ward <ellis.ward@chess-ai>`
- CLAUDE.md is local only — never commit it.
