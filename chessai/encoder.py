"""
encoder.py — Converts a chess board (+ history) into a (55, 8, 8) tensor.

Planes layout:
  0–47  : piece planes for up to 4 history frames (12 planes each)
           within each frame:
             planes 0–5  = current player's pieces (pawn/knight/bishop/rook/queen/king)
             planes 6–11 = opponent's pieces (same order)
  48    : colour to move in absolute terms (all 1s = white, all 0s = black)
  49    : current player — kingside castling right
  50    : current player — queenside castling right
  51    : opponent — kingside castling right
  52    : opponent — queenside castling right
  53    : en passant target square (1 at that square, 0 elsewhere)
  54    : fifty-move counter, normalised to [0, 1] (divide by 100)

The board is always encoded from the perspective of the player to move.
When it's black's turn the board is mirrored vertically and piece colours
swapped so the network always sees "my pieces at the bottom." This is the
same trick we used in Connect Four's 2-channel encoding — just more planes.
"""

import chess
import numpy as np
import torch

PIECE_TYPES = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK,  chess.QUEEN,  chess.KING,
]

N_HISTORY = 4               # number of board states to include
N_PLANES  = 12 * N_HISTORY + 7  # 55 total


def _board_to_planes(board: chess.Board, player: chess.Color) -> np.ndarray:
    """
    Encode one board state as 12 binary planes from player's perspective.
    Mirror the board first if player is BLACK so 'my pieces' sit at rank 1.
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)

    # mirror() flips ranks and swaps piece colours — black's pieces become
    # 'white' at the bottom, giving a consistent perspective for the network
    b = board.mirror() if player == chess.BLACK else board

    for i, pt in enumerate(PIECE_TYPES):
        for sq in b.pieces(pt, chess.WHITE):   # current player's pieces
            planes[i,     sq // 8, sq % 8] = 1.0
        for sq in b.pieces(pt, chess.BLACK):   # opponent's pieces
            planes[i + 6, sq // 8, sq % 8] = 1.0

    return planes


def encode(boards: list) -> torch.Tensor:
    """
    Encode up to N_HISTORY board positions into a (55, 8, 8) float tensor.

    boards[0] = current position (its .turn determines encoding perspective)
    boards[1..] = earlier positions, most recent first
    Pads with zero planes when fewer than N_HISTORY boards are available
    (e.g. early in a game).
    """
    current = boards[0]
    player  = current.turn
    tensor  = np.zeros((N_PLANES, 8, 8), dtype=np.float32)

    # --- Piece planes for each history frame (planes 0–47) ---
    for t, board in enumerate(boards[:N_HISTORY]):
        start = t * 12
        tensor[start : start + 12] = _board_to_planes(board, player)

    # --- Auxiliary planes (48–54) ---
    # Use the mirrored board so castling rights are always
    # expressed as "current player" / "opponent" regardless of colour
    b = current.mirror() if player == chess.BLACK else current

    tensor[48] = 1.0 if player == chess.WHITE else 0.0  # absolute colour

    tensor[49] = 1.0 if b.has_kingside_castling_rights(chess.WHITE)  else 0.0
    tensor[50] = 1.0 if b.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[51] = 1.0 if b.has_kingside_castling_rights(chess.BLACK)  else 0.0
    tensor[52] = 1.0 if b.has_queenside_castling_rights(chess.BLACK) else 0.0

    if b.ep_square is not None:
        tensor[53, b.ep_square // 8, b.ep_square % 8] = 1.0

    # Fifty-move clock: counts half-moves since last capture or pawn push
    tensor[54] = current.halfmove_clock / 100.0

    return torch.tensor(tensor)
