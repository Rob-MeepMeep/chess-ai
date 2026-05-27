"""
main.py — The FastAPI service that the Electron chess-trainer talks to.

FastAPI is a Python web framework. It lets us define API endpoints
(URLs the Electron app can send requests to) with almost no boilerplate.

To run this server:
  venv/bin/uvicorn main:app --reload --port 8765

'uvicorn' is the server that runs FastAPI.
'--reload' means it restarts automatically when you edit this file.
'--port 8765' keeps it away from common ports so nothing clashes.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel

from game.chess_env import ChessEnvironment

# Create the FastAPI application
app = FastAPI(title="Chess AI Service", version="0.1.0")

# CORS (Cross-Origin Resource Sharing) — without this, the Electron
# renderer would be blocked from calling our API due to browser security rules.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow requests from anywhere (fine for local dev)
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request / Response models ------------------------------------------------
# Pydantic models define the shape of data coming in and going out.
# FastAPI validates requests automatically against these.

class MoveRequest(BaseModel):
    fen: str                      # the current board position
    move: Optional[str] = None    # optional: a specific move to apply

class MoveResponse(BaseModel):
    move: str                     # the move chosen (UCI format, e.g. 'e2e4')
    fen: str                      # the board position after the move
    done: bool                    # is the game over?
    result: Optional[str]         # '1-0', '0-1', '1/2-1/2', or None


# --- Endpoints ----------------------------------------------------------------

@app.get("/health")
def health():
    """
    Simple health check — the Electron app calls this on startup
    to confirm the Python service is running.
    """
    return {"status": "ok", "message": "Chess AI service is running"}


@app.post("/move", response_model=MoveResponse)
def get_move(request: MoveRequest):
    """
    Given a board position (FEN), return the AI's chosen move.

    Right now the AI picks a random legal move — this is our placeholder
    until we build the neural network in later phases. The important thing
    is that the pipeline works end-to-end.
    """
    try:
        env = ChessEnvironment()

        # If the Electron app sent a FEN, load that position.
        # Otherwise start from the beginning.
        import chess
        env.board = chess.Board(request.fen)

        # Pick the AI's move (random for now)
        chosen_move = env.random_move()

        # Apply it and get the new state
        result = env.step(chosen_move)

        return MoveResponse(
            move=chosen_move,
            fen=result["fen"],
            done=result["done"],
            result=result["result"]
        )

    except Exception as e:
        # If anything goes wrong, send a clear error back to Electron
        raise HTTPException(status_code=400, detail=str(e))
