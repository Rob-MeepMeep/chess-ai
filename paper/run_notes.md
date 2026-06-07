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

### Run 7 — MacBook Pro M5 Pro (STOPPED EARLY at game 1200, 2026-06-04)

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
| 850 | 50 | 17 | 26 | 7 | 5.05 | 7 | 7 | 80.0 |
| 900 | 50 | 28 | 20 | 2 | 4.97 | 7 | 2 | 65.9 |
| 950 | 50 | 24 | 24 | 2 | 4.93 | 7 | 2 | 70.3 |
| 1000 | 50 | 15 | 24 | 11 | 4.84 | 6 | 11 | 87.1 |
| 1050 | 50 | 19 | 24 | 7 | 4.77 | 6 | 7 | 80.3 |
| 1100 | 50 | 13 | 30 | 7 | 4.74 | 6 | 7 | 85.4 |
| 1150 | 50 | 18 | 29 | 3 | 4.61 | 10 | 3 | 69.9 |
| 1200 | 50 | 17 | 26 | 7 | 4.59 | 9 | 7 | 70.1 |

**Loss trajectory:** Rose from 1.14 (game 50) to 5.27 (plateau at games 700–750), then declined
continuously: 5.15 → 5.05 → 4.97 → 4.93 → 4.84 → 4.77 → 4.74 → 4.61 → **4.59** (new low at game 1200).
Rising loss is a buffer transition artefact (seed positions had zero policy labels; as self-play
replaced them, policy head faced harder targets). Plateau then decline is the expected pattern.
See key_concepts.md L19.

**Value resign trajectory:** 0→1→2→3→3→6→[2,0]→6→3→8→[4]→9→5→5→9→7→7→6→6→**10→9**
Brackets are short-window readings. Peaked at 10 (game 1150 window), settled at 9. Upward trend
overall, but MCTS-driven rather than raw-value-driven after the game 1000 collapse (see below).

**W/B balance:** Broadly balanced across full run (overall ~48%W/52%B). However a persistent black
lean emerged from game 1000 onwards — five consecutive windows at 56–70% black. Not structural
encoder bias (colour plane removed); attributed to self-play meta convergence (black found
consistent counter-strategies against white's opening patterns). Not accelerating toward collapse
but not reverting either — one reason for early stop.

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

**Value head regression — game ~990 (5,060 steps) — PEAK:**

| Position | Value | Expected |
|----------|-------|----------|
| Start | -0.028 | ~0.0 | ✓ |
| K+Q vs K (w wins) | -0.677 | near +1 | ✗ (negative — asymmetry, see note) |
| K+Q vs K (b move) | **-0.950** | near -1 | ✓ **target exceeded** |
| White missing queen | +0.031 | < 0 | ✗ (tiny magnitude, noise) |

K+Q vs K (b move) at -0.950 far exceeds the ±0.6 milestone target. The b-move position
(from the loser's perspective) generalised extremely well. The w-wins position (from the
winner's perspective) reads -0.677 rather than +1 — asymmetry explained by:
1. "Lone king = losing" is a simpler, more uniform pattern than "K+Q = winning"
2. The permanent buffer's canonical FENs differ geometrically from the regression test FENs
3. Self-play data gives more loser-perspective endgame positions (material resign fires
   on the loser's behalf, but winning-side late endgame positions are relatively sparse)

**Eval vs random — game ~1000:**
- As white: 0W / 18L / 82D (draws = cap draws; HAL not delivering checkmate)
- As black: 0W / 13L / 87D
- vs Stockfish depth 1/3/5: 100% losses (0 draws)
- Overall HAL win rate vs random: 0.0%

Expected at this stage. Value head developing but policy not yet delivering wins.

**Value head regression — game ~1000 (5,160 steps) — COLLAPSE:**

| Position | Value | Expected |
|----------|-------|----------|
| Start | -0.094 | ~0.0 |
| K+Q vs K (w wins) | -0.030 | near +1 | ✗ collapsed |
| K+Q vs K (b move) | -0.034 | near -1 | ✗ **collapsed from -0.950** |
| White missing queen | -0.077 | < 0 | near zero |

All values collapsed to near-zero in 100 training steps (5,060 → 5,160).
Root cause: cap draw spike at game 1000 window (11 cap draws, avg length 87.1).
Cap draws are assigned outcome 0.0 but contain late-game/endgame-like positions —
training on them pushes the value head toward zero for position types that should
be decisive. The permanent partition (12.5% of each batch) was insufficient to
resist this gradient signal.

**Value head regression — game ~1200 (6,060 steps) — CONFIRMED FLAT:**

| Position | Value | Expected |
|----------|-------|----------|
| Start | -0.036 | ~0.0 |
| K+Q vs K (w wins) | -0.017 | near +1 | ✗ |
| K+Q vs K (b move) | -0.038 | near -1 | ✗ |
| White missing queen | -0.037 | < 0 | near zero |

Not recovering. Collapse confirmed stable. Value resigns still occurring (9–10/window)
because MCTS with 200 simulations finds decisive tactical lines independently of raw
network output — but the value head itself is no longer contributing signal.

**Notable game events:**
- Game 1: First checkmate (W wins, 85 moves) — seed buffer effect, training from game 1
- Game 6: Second checkmate (W wins, 77 moves)
- Games 663: Tactical sequence — Qxf2+ (check), White responded Qxd7+ (counter), Black takes queen, material resign. Shows both sides finding tactical responses.
- Game 685: Bxh6 bishop sacrifice (takes pawn weakened by h7h6 on move 1), later Nxf7+ fork. Piece coordination and weak-square exploitation.
- Games 671, 689: Value resigns at 72–83 moves in complex positional situations (not simple material collapses).
- Game 816: Pawn promotion `b2b1q` — Black promotes pawn to queen in endgame, wins by material resign. First observed promotion in Run 7.

**Key Run 7 findings:**
1. Buffer seeding works — value head reached -0.950 by game 990 vs flat throughout all of Run 6
2. Cap draws contaminate the value head — outcome 0.0 on positions that look like decisive endgames
   directly contradicts canonical training signal, and overcame the 12.5% permanent partition
3. Permanent partition at 12.5% too small — insufficient to maintain value signal under cap draw pressure
4. Value resigns persist even with a collapsed raw value head — 200-sim MCTS finds tactical patterns
   independently. Value resigns and regression tests measure different things.
5. Black bias emerged post-game 1000 (self-play meta, not encoder structural issue) — 5 consecutive
   windows 56–70% black before early stop

---

### Run 8 — MacBook Pro M5 Pro (IN PROGRESS, started 2026-06-04)

- **Config:** 160ch / 10 blocks / 200 sims / 54 planes / RESIGN_MATERIAL=7 / RESIGN_CONSECUTIVE=5
- **Fresh random weights**
- **Seed buffer:** 13,152 positions from Run 7 decisive games 800–1200 + 1,600 canonical positions (permanent)
- **Key fixes:** MCTS backup sign corrected; cap draw outcomes from material (±0.8); canonical partition 25%

**Training windows:**

| Window | W | B | D | avg loss | checkmates | val resigns | cap draws | avg len |
|--------|---|---|---|----------|------------|-------------|-----------|---------|
| 50  | 18 | 32 | 0 | 1.36 | 8  | 1  | 0 | 71.2 |
| 100 | 28 | 22 | 0 | 1.93 | 9  | 0  | 1 | 74.5 |
| 150 | 27 | 21 | 2 | 2.34 | 13 | 1  | 4 | 73.8 |
| 200 | 21 | 25 | 4 | 2.53 | 11 | 5  | 4 | 80.7 |
| 250 | 24 | 26 | 0 | 2.64 | 7  | 3  | 2 | 77.3 |
| 300 | 28 | 21 | 1 | 2.86 | 10 | 6  | 2 | 78.2 |
| 350* | 27 | — | — | 2.93 | 2  | 5  | 0 | 63.3 |
| 400 | 29 | 20 | 1 | 2.91 | 9  | 9  | 4 | 80.1 |
| 450 | 30 | 20 | 0 | 3.12 | 6  | 6  | 1 | 73.0 |
| 500 | 22 | 26 | 2 | 3.14 | 13 | 7  | 2 | 74.0 |
| 550 | 17 | 33 | 0 | 3.23 | 6  | 12 | 2 | 72.4 |
| 600 | 23 | 26 | 1 | 3.56 | 8  | 9  | 4 | 75.3 |
| 650 | 32 | 14 | 4 | 3.75 | 7  | 12 | 4 | 81.4 |
| 700 | 24 | 25 | 1 | 3.84 | 4  | 13 | 3 | 72.6 |
| 750 | 30 | 18 | 2 | 3.76 | 5  | 4  | 4 | 77.5 |
| 800* | 5 | 14 | 1 | 3.61 | 4  | 4  | 2 | 68.3 |
| 850 | 22 | 26 | 2 | 3.61 | 13 | 8  | 3 | 79.2 |
| 900  | 29 | 19 | 2 | 3.57 | 6  | 8  | 8 | 85.6 |
| 950  | 28 | 19 | 3 | 3.55 | 9  | 6  | 3 | 80.7 |
| 1000 | 25 | 23 | 2 | 3.48 | 9  | 7  | 3 | 73.6 |
| 1050 | 18 | 31 | 1 | 3.42 | 8  | 4  | 2 | 72.5 |
| 1100 | 22 | 26 | 2 | 3.41 | 15 | 8  | 3 | 71.9 |
| 1150 | 28 | 22 | 0 | 3.39 | 7  | 3  | 3 | 73.2 |
| 1200 | 25 | 25 | 0 | 3.36 | 11 | 5  | 1 | 69.2 |
| 1250 | 29 | 19 | 2 | 3.30 | 7  | 5  | 3 | 77.0 |
| 1300 | 27 | 20 | 3 | 3.26 | 7  | 7  | 5 | 80.3 |
| 1350 | 23 | 26 | 1 | 3.28 | 8  | 6  | 3 | 74.9 |
| 1400 | 32 | 17 | 1 | 3.26 | 9  | 8  | 3 | 74.9 |

*Window 350 short (27 games) — mid-window restart after Mac reboot.*
*Window 800 short (20 games) — cause unknown; partial window only.*

**~231 checkmates in first 1,400 games (16% rate).** Run 7 had ~8 total to game 550. Run 6 had its first at game 658.
W/B balance: broadly 50/50 throughout. White wins now slightly dominating checkmates (30W/24B in games 1101–1440) — early run was B-heavy due to Qh4/Qh5 patterns; both sides now more balanced.
Value resigns: stable 5–8 per window from game 950 onwards.
Cap draws: never exceeded 8 per window. Fix working throughout.
Loss: 1.36→3.84 (plateau ~game 700), then declining continuously to **3.26 at game 1400** — new run low. Lower plateau than Run 7 (3.26 vs 5.27) attributed to correct MCTS backup signal.

**Value head regression — game 100 (650 steps):**

| Position | Value | Expected |
|----------|-------|----------|
| Start | -0.039 | ~0.0 | ✓ |
| K+Q vs K (w wins) | -0.132 | near +1 | ✗ (wrong sign, has magnitude) |
| K+Q vs K (b move) | **-0.530** | near -1 | ✓ past ±0.6 milestone |
| White missing queen | -0.058 | < 0 | ✓ |

**Value head regression — game 200 (1,150 steps) — transient sign flip:**

| Position | Value | Expected |
|----------|-------|----------|
| Start | +0.055 | ~0.0 |
| K+Q vs K (w wins) | +0.449 | near +1 | ✓ sign correct |
| K+Q vs K (b move) | +0.392 | near -1 | ✗ sign flipped |
| White missing queen | +0.032 | < 0 | ✗ |

Network passed through a "queen present = positive" intermediate representation.
Both K+Q positions scored positive — value head detected material but not perspective.

**Value head regression — game 260 (1,450 steps):**
- K+Q vs K (w wins): +0.793 (strong, approaching +1)
- K+Q vs K (b move): +0.479 (still wrong sign, contracting)

**Value head regression — game 300 (1,650 steps):**
- K+Q vs K (w wins): +0.530 (backing off)
- K+Q vs K (b move): +0.109 (contracting toward zero)

**Value head regression — game 560 (2,910 steps) — MILESTONE:**

| Position | Value | Expected |
|----------|-------|----------|
| Start | -0.068 | ~0.0 | ✓ |
| K+Q vs K (w wins) | +0.346 | near +1 | ✓ correct sign |
| K+Q vs K (b move) | **-0.603** | near -1 | ✓ **past ±0.6 milestone** |
| White missing queen | -0.064 | < 0 | ✓ |

All four positions correct sign. b-move completed full cycle: -0.530 (game 100) → sign-flipped positive
(games 200–300, "queen = positive" intermediate phase) → contracted through zero → **-0.603** (game 560).

**Eval vs random — game 560 (first ever wins, 2,910 steps):**
- As white: **1W** / 6L / 18D (4% win rate)
- As black: **1W** / 2L / 22D (4% win rate)
- Overall HAL win rate vs random: **4.0%** — first non-zero win rate in project history
- vs Stockfish depth 1/3/5: 100% losses (expected at this stage)

HAL has won games. Draws (72–88%) are cap draws — value head says "I'm winning" but policy head
not yet converting material advantages into checkmate consistently. Closing technique develops with
more training. All previous runs (1–7): 0% win rate at equivalent stages.

**Value head regression — game 750 (3,900 steps) — buffer transition oscillation:**

| Position | Value | Expected |
|----------|-------|----------|
| Start | +0.020 | ~0.0 |
| K+Q vs K (w wins) | **-0.290** | near +1 | ✗ wrong sign |
| K+Q vs K (b move) | -0.195 | near -1 | ✓ sign correct, magnitude weak |
| White missing queen | +0.011 | < 0 | ✗ near zero |

Regression from game 560 (-0.603 b-move). Both canonical positions scored negative — network
saying "player to move loses" in both cases regardless of who has the queen. Attributed to
buffer transition (~game 670): seed data fully displaced, pure self-play now dominant. Recent
self-play had white winning 29W/18B (window 750) — possible white-wins skew in rolling buffer.
Value resigns also dropped sharply (12 → 4 at window 750), corroborating value head weakening.
Assessment: known oscillation at buffer transition; not a collapse. Loss still declining.

**Value head regression — game 1000 (5,110 steps):**

| Position | Value | Expected |
|----------|-------|----------|
| Start | **+0.268** | ~0.0 | ✗ positive bias |
| K+Q vs K (w wins) | +0.182 | near +1 | ✓ sign correct, magnitude weak |
| K+Q vs K (b move) | -0.116 | near -1 | ✓ sign correct, magnitude weak |
| White missing queen | +0.258 | < 0 | ✗ wrong sign |

All four positions clustered +0.12 to +0.27. Value head not discriminating well.
Start position at +0.27 (should be ~0) suggests a systematic positive bias — network
may have learned "current player tends to win" from recent self-play skew rather than
genuine position evaluation. White missing queen scoring +0.26 (should be negative) is
concerning. Value resigns have recovered (8 at windows 850 and 900) so MCTS tactical
sense is intact even as the value head oscillates.

**Eval vs random — game 1000 (5,110 steps):**
- As white: 2W / 1L / 22D (**8% win rate**)
- As black: 4W / 2L / 19D (**16% win rate**)
- Overall HAL win rate vs random: **12.0%** — 3× improvement from game 560
- vs Stockfish depth 1/3/5: 100% losses (expected)

12% win rate despite a weak value head — improvement is policy-head driven. HAL as Black
outperforms as White (16% vs 8%), consistent with self-play patterns where Black has
learned to exploit the Qh4/Qh5 opening traps. Draws (76–88%) remain cap draws; checkmate
conversion still the main gap. 20% target not met; trajectory is right and loss is still
declining. Continue training.

**Value head regression — game ~1420 (7,260 steps):**

| Position | Value | Expected |
|----------|-------|----------|
| Start | +0.017 | ~0.0 | ✓ |
| K+Q vs K (w wins) | **-0.812** | near +1 | ✗ sign flipped again |
| K+Q vs K (b move) | **-0.838** | near -1 | ✓ new Run 8 record |
| White missing queen | +0.024 | < 0 | ✗ near zero |

Both canonical positions scoring ~-0.82. Value head has learned the endgame is decisive (strong
magnitude) but is outputting the same negative sign regardless of whose turn it is — detecting
board structure without separating perspective. Same asymmetry observed in Run 7 (b-move peaked
at -0.950 while w-wins stayed negative). b-move at -0.838 is the strongest reading in Run 8.

**Eval vs random — game ~1500 (7,310 steps):**
- As white: 4W / 3L / 18D (**16% win rate**)
- As black: 5W / 2L / 18D (**20% win rate**)
- Overall HAL win rate vs random: **18.0%** — new project high, target met as Black
- vs Stockfish depth 1 (200 sims, 25 games each colour): **100% losses**, 0 draws
- vs Stockfish depth 3/5: not run — depth 1 result made these redundant at current strength

18% vs random is a consistent improvement (4% → 12% → 18%). HAL as Black hits the 20% target;
as White still 16%. Draws remain 72% — cap draws dominate, conversion still the main gap.

Stockfish depth 1 result (0%, 0 draws in 50 games) confirms HAL is winning via opponent blunders,
not generating winning plans against a non-blundering opponent. Gate for running depth 3/5:
HAL must score at least occasional draws vs depth 1 first.

**Next eval: game 2000. Target: 25%+ win rate vs random, draws vs Stockfish depth 1.**

---

## Current Code State

Key config in `train_chess.py`:
```python
N_SIMULATIONS    = 200
MAX_GAME_MOVES   = 150
RESIGN_THRESHOLD   = -0.95
RESIGN_CONSECUTIVE = 5
RESIGN_MATERIAL    = 7
RUN_NAME    = "run8"
CKPT_LOAD   = None                                # fresh weights
BUFFER_LOAD = "checkpoints/run8_seed_buffer.pt"  # curated from run7 decisive games 800–1200
```

Key additions since Run 1:
- `chessai/agent.py` — `get_value()` single forward pass for resignation check
- `chessai/replay.py` — `save()`/`load()` buffer persistence; tiered buffer (permanent + rolling, 25% canonical)
- `chessai/logger.py` — `games.csv` (full per-game log), `end_reasons.csv`, run-namespaced log dirs
- `curate_buffer.py` — builds seed buffer from prior run data + canonical endgame positions
- `train_chess.py` — RUN_NAME system, CKPT_LOAD/BUFFER_LOAD, cap draw material outcome, buffer priority logic
- `chessai/mcts.py` — **backup sign fix** (flip before update — see critical bug below)

**Critical bug fixed before Run 8 — MCTS backup sign:**
All runs prior to Run 8 had an inverted backup. The value from the neural network
represents how good a position is for the player to move at the leaf. The backup
must flip this to the parent's perspective before storing in `node.W`. The old code
flipped *after* storing — so `node.W` held the value from the *wrong player's*
perspective. MCTS was maximising the opponent's advantage every simulation.

Effect: the policy head learned to prefer the moves MCTS visited most — which were
the worst available moves. The value head was unaffected (trained from game outcomes,
not MCTS Q values). This explains the persistent 0% win rate vs random across all
runs despite the value head clearly developing.

```python
# BROKEN (runs 1–7): flip after storing
for node in reversed(path):
    node.N += 1
    node.W += value
    value = -value       # too late — node.W already has the wrong sign

# FIXED (run 8+): flip before storing
for node in reversed(path):
    value = -value       # convert to parent's perspective first
    node.N += 1
    node.W += value
```

**Resume command:**
```bash
caffeinate -dims venv/bin/python3 train_chess.py
```

---

## What to Watch in Run 8

| Signal | What it means |
|--------|---------------|
| Value resigns per 50-game window | Core health signal. Growing = value head developing. Stable = plateau. Zero = investigate. |
| K+Q vs K regression (every 200 games) | Target ≥ ±0.6 by game 1000. Watch for collapse after draw spikes. |
| Cap draw outcome mode | New: cap draws in material-imbalanced positions assigned ±0.8, not 0.0. Verify draws are balanced positions. |
| Cap draws per window | <10 = healthy. A spike >10 is a warning — check if it triggers value collapse. |
| W/B tally | Should stay near 50/50. Sustained skew = self-play meta issue, not encoder bias. |
| Loss trajectory | Expect same arc: rise as seed→self-play transitions, plateau, then decline. |

---

## Next Milestones

1. ~~Run 8 seed buffer~~ — ✓ done
2. ~~Fix cap draw outcomes~~ — ✓ done
3. ~~Increase permanent partition~~ — ✓ done
4. ~~Start Run 8~~ — ✓ running
5. ~~First wins vs random~~ — ✓ game 560, 4% win rate (first in project history)
6. ~~Eval at game 1000~~ — ✓ done. 12% win rate (3× improvement from game 560). 20% target not met; continuing.
7. ~~Eval at game 1500~~ — ✓ done. 18% win rate vs random (20% as Black). 0% vs Stockfish depth 1. Target met as Black; overall just short at 18%.
8. ~~Eval at game 2000~~ — ✓ done. 12% win rate vs random (regression from 18%). 0% vs Stockfish depth 1. See final summary below.
9. ~~Gemini game analysis~~ — ✓ done. Output: `paper/gemini_run8_assessment.md`.
10. **Stage 2 resign** — once K+Q vs K reads ±0.9 consistently, remove material resign entirely.
11. **Phase 4** — UCI wrapper → Lichess bot account → ELO rating.

---

## Run 8 — Final Summary

**Status: COMPLETE — stopped at game 2010, 10,160 steps (2026-06-07)**

### Results

| Eval | Game 560 | Game 1000 | Game 1500 | Game 2010 |
|------|----------|-----------|-----------|-----------|
| Win rate vs random | 4% | 12% | 18% | 12% |
| vs Stockfish depth 1 | — | 0% | 0% | 0% |
| b-move regression | −0.603 | −0.116 | −0.785 | **−0.921** |
| w-wins regression | — | +0.270 | −0.838 | +0.048 |
| Loss | — | 3.57 | 3.26 | **3.093** |

### Key findings

**What Run 8 proved:**
- MCTS backup sign fix was decisive — first wins in project history at game 560 (4%), 12% by game 1000
- Buffer seeding (25% permanent partition) kept the value head stable through the seed→self-play transition
- Cap draw ±0.8 outcome worked — no value collapse despite occasional cap draw spikes
- ~240 checkmates in 2010 games, including Fool's Mate (game 823), Scholar's Mate (game 830), Na5# (game 1886), Bg7# (game 1509)
- Checkmate pattern evolution confirmed: opening traps (games 1–1000) → earned tactical checkmates (games 1000+)

**The Geometry Trap — diagnosis and revision:**
The w-wins regression oscillated throughout the run (+0.27 → −0.838 → +0.048) while b-move converged strongly (−0.921). Initially diagnosed as a missing canonical encoding in the encoder. On inspection (2026-06-07), the canonical encoding (`board.mirror() if player == chess.BLACK`) has been in the encoder since Run 6 (commit 05b3e53). The phenomenon is real; the cause was wrong.

Revised diagnosis: **training data distribution asymmetry**. RESIGN_MATERIAL ends games before White can move in true K+Q vs K positions. The buffer is dominated by Black-to-move losing positions (generated right to the resign) with sparse White-to-move winning positions. The permanent partition covers only one spatial K+Q vs K configuration — insufficient for generalisation. See `paper/run9_architecture.md` addendum for full analysis.

**Win rate plateau:**
Despite loss reaching a new low (3.093) and b-move reaching a new high (−0.921), win rate regressed from 18% to 12% at game 2010. The 84% draw rate confirms HAL consistently reaches winning positions but cannot convert them. The value head's w-wins blindspot (reading near zero for winning positions) prevents MCTS from pressing material advantages. This is the binding constraint — more training under the same conditions would not resolve it.

**External assessment:**
Gemini analysis of games 1500–1900 (401 games): confirmed play quality jump from "geometric chaotic" (game 1500) to "coherent predatory" (game 1890). Identified f-pawn fixation as a recurring tactical overcorrection and noted HAL compensates for perspective ambiguity by preferring forcing tactical lines. Games 1886 (Na5#) and 1509 (Bg7#) flagged as paper-worthy. Full assessment: `paper/gemini_run8_assessment.md`.

### Decision: start Run 9

Run 8 has reached its structural ceiling. The Geometry Trap (sparse winning-side training signal) cannot be resolved by continued training — it requires changes to the permanent partition and RESIGN_MATERIAL schedule. Architecture unchanged; weights carried forward into Run 9 (encoder identical, strong value head foundation to build on).

**Run 9 changes:**
- Continue from Run 8 checkpoint (not fresh weights — encoder unchanged, value head has strong b-move foundation)
- Seed buffer from Run 8 games 1500–2010 (better quality than early Run 8)
- More diverse permanent partition (multiple K+Q vs K configurations)
- Investigate reducing RESIGN_MATERIAL schedule to expose true endgame positions
- See `paper/run9_architecture.md`

---

## Run 9 — COMPLETE (game 1000, 15,135 steps, 2026-06-07)

**Config:** Continue from Run 8 checkpoint (10,160 steps). RESIGN_MATERIAL lowered 7→3. Diverse permanent partition (256 K+Q vs K spatial configurations). Seed buffer from Run 8 games 1500–2010.

### Final Results

| Metric | Game 410 | Game 1000 |
|--------|----------|-----------|
| w-wins | +0.9990 | +0.9477 |
| b-move | −0.9968 | −0.9951 |
| start | −0.026 | −0.117 |
| white missing queen | −0.014 | −0.108 |
| HAL as White vs random | 0% | 8% |
| HAL as Black vs random | 24% | 12% |
| Overall vs random | 12% | 10% |
| vs Stockfish depth 1 | 0% | 0% |
| Loss range | — | 1.6–1.9 |

### Key findings

**Geometry Trap resolved.** w-wins reached +0.9990 at game 410 — Run 8 never exceeded +0.048 across 2010 games. RESIGN_MATERIAL=3 combined with the diverse permanent partition (256 K+Q vs K configurations) broke the data distribution asymmetry that caused the trap. Both value head signals now near-saturated and stable.

**First White wins vs random.** HAL won 2/25 games as White at game 1000 (was 0/25 at game 410). First confirmed White wins in the project. The w-wins fix translated into play, slowly.

**Black performance regression.** HAL dropped from 24% to 12% wins as Black, and lost 4/25 games to random — something that did not happen at game 410. The policy head is mid-transition: calibrated to the old value landscape (Black-dominant) and not yet converged to the rebalanced one. The start position value drifting to −0.117 (from −0.026) is a related signal — the network is developing a mild Black-favoured prior.

**f2f3 opening bias.** Policy head cycled between 21–70% preference for 1. f2f3 throughout the run (snapshots at games 50–850). Self-play local minimum: f2f3 leads to closed structures that HAL handles better than open tactical play, creating a positive feedback loop. Oscillation rather than monotonic lock-in confirms the policy is still learning, but keeps returning to the same local optimum.

**Loss trajectory.** 1.6–1.9 throughout (well below Run 8's 3.0–3.8), reflecting the continuing from Run 8 weights advantage. Zero cap draws throughout — contamination fix held.

**W/B balance.** 25/25 at the game 800 window. Geometry Trap resolved in self-play balance.

**No Stockfish progress.** 100% losses both colours throughout. First draws vs Stockfish depth 1 remain the next benchmark.

### Decision: close Run 9, start Run 10

Run 9's structural ceiling is the policy head local minimum. Without Dirichlet noise at the MCTS root, the f2f3 cycle will persist and the White/Black rebalancing will not resolve cleanly. The value head is healthy and should be carried forward.

**Run 10 plan:**
- Continue from Run 9 weights
- Dirichlet noise at MCTS root (α=0.3, ε=0.25) — primary change
- Seed from Run 9 games 800–1000 (decisive, both-colour wins)
- RESIGN_MATERIAL=3 retained

**Run 10 fix — Dirichlet noise at MCTS root:**

Standard AlphaZero adds noise to the root node policy during training to force exploration of minority moves:

```
p_noisy = (1 - ε) × p_network + ε × Dirichlet(α)
```

Standard values for chess (AlphaZero paper): **α = 0.3, ε = 0.25**. Applied at root node only, training only (not eval). Change location: `chessai/mcts.py`, in the root node policy initialisation. Breaks opening lock-in by ensuring the agent always samples non-preferred moves during self-play, even when the policy is highly confident.

---

## Working Style Notes

- Rob Kirkland is the project lead, learning ML/Python through this project. Explain before writing — the *why* matters.
- Ellis Ward is the AI research collaborator. All formal documents credited as "Rob Kirkland, Ellis Ward."
- No Claude model names or Anthropic references in committed files.
- Co-Authored-By in commits: `Ellis Ward <ellis.ward@chess-ai>`
- CLAUDE.md is local only — never commit it.
