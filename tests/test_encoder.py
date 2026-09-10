"""
test_encoder.py — perspective/mirroring and history reconstruction.

chessai/encoder.py's core invariant: the network always sees "my pieces
at the bottom," regardless of which colour is actually to move. These
tests check that invariant directly rather than trusting the docstring.
"""

import chess
import pytest
import torch

from chessai.encoder import encode, N_HISTORY, N_PLANES


def test_symmetric_position_encodes_identically_for_either_colour():
    """The standard starting position is mirror-symmetric. If the encoder's
    perspective-flip is correct, encoding it as White-to-move and as
    Black-to-move (same piece layout, artificial FEN) must give identical
    piece planes (0-11) -- "my pawns on my second rank" looks the same
    either way."""
    white_to_move = chess.Board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    black_to_move = chess.Board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")

    enc_white = encode([white_to_move])
    enc_black = encode([black_to_move])

    assert torch.equal(enc_white[0:12], enc_black[0:12])


def test_asymmetric_single_piece_mirrors_correctly():
    """A single white knight on a1, White to move, vs. the mirror-image
    single black knight on a8, Black to move -- both should show the
    knight in the exact same plane/square from "my" perspective."""
    white_board = chess.Board("4k3/8/8/8/8/8/8/N3K3 w - - 0 1")
    black_board = chess.Board("n3k3/8/8/8/8/8/8/4K3 b - - 0 1")

    enc_white = encode([white_board])
    enc_black = encode([black_board])

    # plane 1 = current player's knights (PIECE_TYPES[1] == KNIGHT)
    assert torch.equal(enc_white[1], enc_black[1])
    # and it should actually be at a1's array position (rank 0, file 0)
    assert enc_white[1, 0, 0] == 1.0


def test_opponent_pieces_land_in_the_opponent_planes():
    """Current player's pieces go in planes 0-5, opponent's in 6-11 --
    never mixed up, regardless of colour."""
    board = chess.Board("4k3/8/8/8/8/8/8/N3K3 w - - 0 1")   # white knight, white to move
    enc = encode([board])
    assert enc[1].sum() == 1.0     # own knight plane has exactly one entry
    assert enc[7].sum() == 0.0     # opponent knight plane is empty

    board2 = chess.Board("n3k3/8/8/8/8/8/8/4K3 w - - 0 1")  # black knight, WHITE to move
    enc2 = encode([board2])
    assert enc2[1].sum() == 0.0    # white has no knight of its own
    assert enc2[7].sum() == 1.0    # black's knight shows up as the opponent's


def test_history_padding_when_fewer_than_n_history_boards():
    """Early in a game there's no real history yet -- missing frames
    should be all-zero, not reused/duplicated from the current position."""
    board = chess.Board()
    enc = encode([board])   # only one board given, no history

    # frame 0 (planes 0-11) should be populated (starting position)
    assert enc[0:12].sum() > 0
    # frames 1-3 (planes 12-47) should be entirely zero -- no history given
    assert enc[12:48].sum() == 0.0


def test_history_frames_are_most_recent_first():
    """boards[0] is current, boards[1] is one ply back, etc. -- frame 1
    (planes 12-23) should reflect the position actually passed as
    boards[1], not boards[0] again."""
    start = chess.Board()
    after_e4 = chess.Board()
    after_e4.push_uci("e2e4")

    # boards[0]=after_e4 (current), boards[1]=start (one ply back)
    enc = encode([after_e4, start])

    # current frame (0-11): mover is Black (after 1.e4), so plane 0 is
    # Black's pawns -- e4 pawn belongs to White, so it should NOT appear
    # as a "current player" pawn in frame 0.
    # history frame (12-23), mirrored the same way, should show the
    # *starting* position instead -- different from frame 0.
    assert not torch.equal(enc[0:12], enc[12:24])


def test_material_plane_reflects_current_player_perspective():
    """Plane 54 is material balance from the CURRENT PLAYER's perspective,
    scaled by /20 and clipped to [-1, 1] -- not raw White-perspective."""
    # White is missing its queen -- bad for White specifically
    white_down_queen = chess.Board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    enc = encode([white_down_queen])
    # current player is White, White is down a queen (-9) -> negative
    assert enc[54, 0, 0].item() < 0

    black_down_queen = chess.Board(
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    enc2 = encode([black_down_queen])
    # current player is White, Black is down a queen -> positive for White
    assert enc2[54, 0, 0].item() > 0

    # magnitude check: -9/20 = -0.45
    assert enc[54, 0, 0].item() == pytest.approx(-9 / 20)


def test_encoded_shape():
    board = chess.Board()
    enc = encode([board])
    assert enc.shape == (N_PLANES, 8, 8)
    assert N_PLANES == 12 * N_HISTORY + 6 + 1
