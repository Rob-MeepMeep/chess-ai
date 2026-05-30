# chess-ai — Project Run Notes
**Authors:** Rob Kirkland, Ellis Ward  
**Last updated:** 2026-05-30

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

### Run 4 — MacBook Pro M5 Pro (IN PROGRESS, started 2026-05-30)
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

---

## Current Code State

Key config in `train_chess.py`:
```python
N_SIMULATIONS    = 100
MAX_GAME_MOVES   = 150
RESIGN_THRESHOLD   = -0.95   # conservative — loosen to -0.85 if value resigns stay rare after 500 games
RESIGN_CONSECUTIVE = 3
RESIGN_MATERIAL    = 5       # rook
RUN_NAME    = "run4"
CKPT_LOAD   = "checkpoints/hal_chess.pt"   # seeded from run3
BUFFER_LOAD = None                          # clean buffer
```

Key additions since Run 1:
- `chessai/agent.py` — `get_value()` single forward pass for resignation check
- `chessai/replay.py` — `save()`/`load()` buffer persistence (every 10 games)
- `chessai/logger.py` — `games.csv` (full per-game log), `end_reasons.csv` (fixed), run-namespaced log dirs
- `train_chess.py` — RUN_NAME system, CKPT_LOAD/BUFFER_LOAD, elapsed time display, per-game end_reason

**Resume command (on Pro):**
```bash
caffeinate -dims venv/bin/python3 train_chess.py
```

---

## What to Watch in Run 4

| Signal | Meaning |
|--------|---------|
| W/B results appearing | Resign is firing — first real reward signal |
| Game length dropping below 150 | Resign is shortening games, speeding up training |
| Loss dropping steadily | Value head receiving real gradient signal |
| `end_reasons.csv` — material_resign dominant | Expected early. Value resign appearing = value head learning |
| `end_reasons.csv` — cap_draw still dominant after game 500 | Resign thresholds may need further loosening |

**Value head regression test** (run via `eval_chess.py` at any checkpoint):
- Start position → expect ~0.0
- K+Q vs lone K (white to move) → expect near +1
- K+Q vs lone K (black to move) → expect near -1
- White missing queen → expect < 0

If K+Q vs K still evaluates near 0 after 500 games of Run 4, something deeper is wrong.

---

## Next Milestones

1. **Run 4 game 500** — run value regression test, check end_reasons.csv distribution
2. **Run 4 game 1000–2000** — run full eval vs random. Target: meaningful win rate (>20%)
3. **Run 4 game 3000–5000** — if win rate is climbing, this is where tactical play should emerge
4. **Phase 4** — UCI wrapper → Lichess bot account → ELO rating

---

## Working Style Notes

- Rob Kirkland is the project lead, learning ML/Python through this project. Explain before writing — the *why* matters.
- Ellis Ward is the AI research collaborator. All formal documents credited as "Rob Kirkland, Ellis Ward."
- No Claude model names or Anthropic references in committed files.
- Co-Authored-By in commits: `Ellis Ward <ellis.ward@chess-ai>`
- CLAUDE.md is local only — never commit it.
