# Eval Results — Game 1090 / 5,480 Training Steps

**Date:** 2026-05-29  
**Checkpoint:** hal_chess_1090games.pt  
**Loss at milestone:** ~3.35 (down from 8.07 at game 0)

## vs Random (100 games each side)

| Matchup | HAL wins | Losses | Draws |
|---|---|---|---|
| HAL (White) vs Random (Black) | 0 (0%) | 9 (9%) | 91 (91%) |
| Random (White) vs HAL (Black) | 0 (0%) | 15 (15%) | 85 (85%) |
| **Overall HAL win rate** | **0.0%** | | |

Draws are from the 100-move cap — not genuine drawn positions.

## vs Stockfish

| Matchup | HAL wins | Losses | Draws |
|---|---|---|---|
| HAL (White) vs Stockfish depth 1 | 0 (0%) | 50 (100%) | 0 (0%) |

Eval cancelled after this matchup — result already clear.

## Assessment

HAL cannot beat a random opponent at this stage. The network is learning  
move structure and losing less badly over time, but has not yet developed  
winning patterns. Expected at ~1k games — AlphaZero-style training  
typically shows real tactical play from ~3,000–5,000 games onward.

Next eval target: 5,000 games.
