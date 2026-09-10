"""
main.py — FastAPI service for HAL-4000.

Loads the trained chess model on startup and serves moves via HTTP.
Chess-trainer calls this to get HAL's chosen move for any position.

To run:
  venv/bin/uvicorn main:app --reload --port 8765
"""

import os
# WSL2 ROCm GPU detection requirement
os.environ["HSA_ENABLE_DXG_DETECTION"] = "1"

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
checkpoint_loaded = False   # exposed via /health -- a missing checkpoint
                           # used to still report "ok" (10 Sept 2026 assessment)
try:
    agent.load(CKPT)
    checkpoint_loaded = True
    print(f"HAL-4000 loaded from {CKPT} ({agent.steps:,} training steps)")
except FileNotFoundError:
    print(f"WARNING: no checkpoint at {CKPT} — HAL will play randomly via untrained network")

MAX_SIMULATIONS = 1000   # bounds request.n_simulations -- unbounded before this
                        # (10 Sept 2026 assessment), a client could request an
                        # arbitrarily large or non-positive simulation count

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
        "status":            "ok" if checkpoint_loaded else "degraded",
        "model":             "HAL-4000",
        "checkpoint_loaded": checkpoint_loaded,   # False = playing on an untrained network
        "steps":             agent.steps,
        "device":            str(device),
    }

@app.post("/move", response_model=MoveResponse)
def get_move(request: MoveRequest):
    try:
        board, history = replay_moves(request.moves)

        # request.fen is documented as "used as a sanity check" but was
        # never actually checked against anything -- a desync between the
        # client's tracked position and its move list would silently make
        # HAL move on the wrong board (10 Sept 2026 assessment). epd() alone
        # ignores the halfmove clock too, but that's a real network input
        # (encoder.py plane 53, the fifty-move counter) — a client desync
        # there wouldn't just be cosmetic, it'd change what HAL actually
        # sees, so it's checked explicitly rather than folded into epd().
        # fullmove number is the only field genuinely safe to ignore (not
        # an input anywhere). The replayed board is authoritative for all
        # of this once request.moves is trusted — this check exists to
        # catch a client-side desync, not because we use request.fen's
        # own counters for anything.
        expected = chess.Board(request.fen)
        if board.epd() != expected.epd() or board.halfmove_clock != expected.halfmove_clock:
            raise ValueError(
                f"fen does not match replayed moves — client thinks the "
                f"position is {request.fen!r}, but replaying moves gives "
                f"{board.fen()!r}"
            )

        if board.is_game_over():
            raise ValueError(f"Position is already game over: {board.result()}")

        if not (1 <= request.n_simulations <= MAX_SIMULATIONS):
            raise ValueError(
                f"n_simulations must be between 1 and {MAX_SIMULATIONS}, "
                f"got {request.n_simulations}"
            )

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
