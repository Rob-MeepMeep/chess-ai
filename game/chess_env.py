"""
chess_env.py — A wrapper around python-chess that gives us a clean
interface for the AI to interact with.

python-chess handles all the rules (legal moves, check, checkmate etc.)
the same way chess.js does on the Electron side. We use it here so the
Python agent never has to think about chess rules — it just asks
"what moves are legal?" and "did the game end?".
"""

import chess
import random


class ChessEnvironment:
    """
    Represents a single game of chess.

    The agent calls:
      - reset()       → start a new game, get the starting board state
      - step(move)    → make a move, get back the new state + whether game is over
      - legal_moves() → list of all moves the current player can make
    """

    def __init__(self):
        # chess.Board() sets up the starting position
        self.board = chess.Board()

    def reset(self) -> str:
        """Start a fresh game and return the starting FEN."""
        self.board = chess.Board()
        return self.board.fen()

    def step(self, move_uci: str) -> dict:
        """
        Apply a move (given in UCI format e.g. 'e2e4') and return
        the result. UCI is the same format Stockfish uses — just
        the from-square and to-square concatenated.

        Returns a dict with:
          - fen:     the new board position as a FEN string
          - done:    True if the game is over
          - result:  '1-0', '0-1', '1/2-1/2', or None if still playing
          - legal:   list of legal moves from the new position
        """
        move = chess.Move.from_uci(move_uci)

        # Reject illegal moves rather than crashing
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move_uci}")

        self.board.push(move)

        done = self.board.is_game_over()
        result = self.board.result() if done else None

        return {
            "fen": self.board.fen(),
            "done": done,
            "result": result,
            "legal": [m.uci() for m in self.board.legal_moves]
        }

    def legal_moves(self) -> list[str]:
        """Return all legal moves for the current player in UCI format."""
        return [m.uci() for m in self.board.legal_moves]

    def random_move(self) -> str:
        """Pick a random legal move. Used as a placeholder until the AI is trained."""
        return random.choice(self.legal_moves())
