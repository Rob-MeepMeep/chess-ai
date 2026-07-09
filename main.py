"""
main.py — FastAPI service for HAL-4000.

Loads the trained chess model on startup and serves moves via HTTP.
Chess-trainer calls this to get HAL's chosen move for any position.

To run:
  venv/bin/uvicorn main:app --reload --port 8765
"""

import torch
import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from pydantic import BaseModel

from chessai.agent import ChessAgent
from run_config    import CKPT_PATH   # the active run's checkpoint — the old
                                      # hardcoded "hal_chess.pt" matched no run,
                                      # so the API silently served random weights

# ── Load HAL on startup ───────────────────────────────────────────────────────

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
agent = ChessAgent(device, n_simulations=50)

CKPT = CKPT_PATH
try:
    agent.load(CKPT)
    print(f"HAL-4000 loaded from {CKPT} ({agent.steps:,} training steps)")
except FileNotFoundError:
    print(f"WARNING: no checkpoint at {CKPT} — HAL will play randomly via untrained network")

app = FastAPI(title="HAL-4000 Chess Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response ────────────────────────────────────────────────────────

class MoveRequest(BaseModel):
    fen: str                     # current board position (used as a sanity check)
    moves: List[str] = []        # all moves played so far in UCI format — used to
                                 # reconstruct board history for the 4-frame encoder
    n_simulations: int = 50      # MCTS rollouts per move (lower = faster but weaker)

class MoveResponse(BaseModel):
    move: str                    # HAL's chosen move in UCI format (e.g. 'e2e4')
    fen: str                     # board position after the move
    done: bool                   # is the game over?
    result: Optional[str] = None # '1-0', '0-1', '1/2-1/2' or None

# ── Helper ────────────────────────────────────────────────────────────────────

def replay_moves(moves: List[str]) -> tuple:
    """
    Replay a list of UCI moves from the starting position.
    Returns (current_board, history) where history is the last 3 board states.
    History is what the 4-frame encoder expects alongside the current board.
    """
    board   = chess.Board()
    history = []
    for uci in moves:
        history = ([board.copy()] + history)[:3]
        board.push_uci(uci)
    return board, history

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model":  "HAL-4000",
        "steps":  agent.steps,
        "device": str(device),
    }

@app.post("/move", response_model=MoveResponse)
def get_move(request: MoveRequest):
    try:
        board, history = replay_moves(request.moves)

        move_uci, _, _ = agent.choose_move(
            board, history,
            greedy=True,
            n_simulations=request.n_simulations
        )

        board.push_uci(move_uci)
        done   = board.is_game_over()
        result = board.result() if done else None

        return MoveResponse(
            move=move_uci,
            fen=board.fen(),
            done=done,
            result=result,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
