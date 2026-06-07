# Gemini Game Analysis Prompt — Run 8
**For use in Phase 3 paper. Run after Run 8 completes.**

---

## Purpose

Feed this prompt + game data to Gemini (or equivalent web AI) to get an independent
assessment of HAL-4000's play quality across Run 8. The output informs the Phase 3 paper
write-up — particularly the sections on tactical evolution and what the agent actually learned.

This is an external perspective on the data, not a training tool. The results go into
the paper analysis, not the replay buffer.

---

## Data Preparation (do this before running the prompt)

You'll need to convert a selection of games from `logs/run8/games.csv` (UCI move format)
into PGN for readability. Pick games from three stages:

1. **Early Run 8** (games 1–500) — include 3–5 checkmate games and 2–3 cap draws
2. **Mid Run 8** (games 800–1100) — include the window 1051–1100 spike; game 1094 and 1081 specifically
3. **Late Run 8** (games 1400–2000) — include 3–5 checkmates and 2–3 losses vs Stockfish depth 1

A UCI→PGN converter: paste UCI move list into [lichess.org/paste](https://lichess.org/paste)
or use `chess.pgn` in Python:

```python
import chess, chess.pgn, io

def uci_to_pgn(uci_moves: str, game_num: int, white="HAL-4000", black="HAL-4000") -> str:
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = white
    game.headers["Black"] = black
    game.headers["Event"] = f"Run 8 Self-Play, Game {game_num}"
    node = game
    for uci in uci_moves.strip().split():
        move = chess.Move.from_uci(uci)
        node = node.add_variation(move)
        board.push(move)
    game.headers["Result"] = board.result()
    return game.accept(chess.pgn.StringExporter())
```

---

## The Prompt

Paste this into Gemini, followed immediately by the prepared PGN games.

---

> I'm building a chess-playing AI from scratch using AlphaZero-style deep reinforcement
> learning and self-play — no human game data, no opening books, no endgame tables.
> The following games are from training Run 8, at three stages of training:
> **early** (~games 1–500), **mid** (~games 800–1100), and **late** (~games 1400–2000).
> All games are self-play: the AI played both sides against itself.
>
> I'd like you to assess the following for each stage, then give an overall summary:
>
> **1. Play quality**
> Are the moves strategically coherent, or do they look random? Are there signs of
> genuine tactical awareness — piece coordination, king safety, material counting —
> or is it pattern noise? Be specific about what you see.
>
> **2. Tactical patterns**
> Identify any recurring tactical ideas the agent appears to have learned. Note if
> the same pattern appears across multiple games or across colours.
>
> **3. Evolution across training**
> How does the quality of play change from early to mid to late in the run?
> What has the agent demonstrably learned by the late stage that it didn't know early?
>
> **4. Notable moments**
> Flag any individual moves or positions that are genuinely impressive or interesting —
> things a human club player would recognise as good chess. Also flag anything that
> is obviously bad in an instructive way.
>
> **5. Paper-worthy material**
> Which games or positions would you highlight in a research paper on self-play learning?
> What story do they tell about the learning process?
>
> Context for your assessment:
> - The agent started from random weights — it had no prior chess knowledge.
> - It uses Monte Carlo Tree Search (200 simulations per move) plus a neural network
>   policy and value head, trained entirely from self-play outcomes.
> - Run 8 is approximately 2,000 games long. The network is still learning.
> - Known limitation: the encoder is colour-blind (no explicit colour plane), which
>   causes the value head to learn board geometry without cleanly separating perspective.
>   This is being fixed in Run 9.
>
> [PASTE PGN GAMES HERE]

---

## What to do with the output

- Paste Gemini's response into `paper/gemini_run8_assessment.md` (create when ready)
- Extract any specific game annotations or observations into `checkmate_analysis.md`
- Use the "paper-worthy material" section to inform the Phase 3 paper narrative

---

*Created 2026-06-06. Run after game 2000 eval completes and before drafting Phase 3 paper.*
