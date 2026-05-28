"""
moves.py — Translates between chess moves and policy indices 0–4095.

We use a flat 64×64 encoding: from_square * 64 + to_square.
This covers every possible (from, to) square pair — 4,096 combinations.

Promotions always default to queen. Under-promotion (to rook, bishop,
or knight) is a known simplification; it's rare in practice and the
complexity of encoding it isn't worth the overhead at this stage.
"""

import chess
import torch


def move_to_index(move: chess.Move) -> int:
    """Map a chess.Move to a policy index in [0, 4095]."""
    return move.from_square * 64 + move.to_square


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Map a policy index back to a chess.Move.
    Adds queen promotion automatically for pawn moves reaching the back rank.
    Requires the board to check whether the moving piece is a pawn.
    """
    from_sq = index // 64
    to_sq   = index % 64

    piece = board.piece_at(from_sq)
    if (piece and piece.piece_type == chess.PAWN
            and chess.square_rank(to_sq) in (0, 7)):
        return chess.Move(from_sq, to_sq, promotion=chess.QUEEN)

    return chess.Move(from_sq, to_sq)


def uci_to_index(uci: str) -> int:
    """Convenience: UCI string (e.g. 'e2e4') → policy index."""
    return move_to_index(chess.Move.from_uci(uci))


def legal_move_mask(board: chess.Board) -> torch.Tensor:
    """
    Return a boolean tensor of shape (4096,) where True = move is legal.
    Used to mask illegal moves to -inf before argmax, same pattern as the
    full-column mask in the Connect Four agent.
    """
    mask = torch.zeros(4096, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move_to_index(move)] = True
    return mask
