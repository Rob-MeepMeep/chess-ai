"""
test_moves.py — move index encoding, promotion defaulting, mirror table.
"""

import chess
import torch

from chessai.moves import (
    move_to_index, index_to_move, uci_to_index,
    get_mirror_indices, get_mirror_indices_np, mirror_policy,
)


def test_move_to_index_roundtrip_for_ordinary_moves():
    board = chess.Board()
    for uci in ["e2e4", "g1f3", "d2d4", "b1c3"]:
        move = chess.Move.from_uci(uci)
        idx = move_to_index(move)
        assert 0 <= idx <= 4095
        recovered = index_to_move(idx, board)
        assert recovered.from_square == move.from_square
        assert recovered.to_square == move.to_square


def test_promotion_always_resolves_to_queen():
    """moves.py's documented, deliberate simplification: whichever
    promotion piece was originally intended, index_to_move always comes
    back as queen. This is a design choice (see moves.py's module
    docstring), not a bug -- this test pins that behaviour down so a
    future change to it is a deliberate edit, not an accidental one."""
    board = chess.Board("8/P7/8/8/4k3/8/8/4K3 w - - 0 1")   # pawn one step from promoting
    idx = move_to_index(chess.Move.from_uci("a7a8q"))

    for promo_uci in ["a7a8q", "a7a8r", "a7a8b", "a7a8n"]:
        assert move_to_index(chess.Move.from_uci(promo_uci)) == idx, (
            "all four promotion pieces must collide on the same index -- "
            "this is exactly the fact diagnose_mcts's prior-sum fix depends on"
        )

    move = index_to_move(idx, board)
    assert move.promotion == chess.QUEEN


def test_non_pawn_move_to_back_rank_is_not_treated_as_promotion():
    """index_to_move only adds a promotion when the piece on from_square
    is actually a pawn -- a rook or king move to the back rank must not
    get a spurious promotion attached."""
    board = chess.Board("8/8/8/8/8/8/8/R3K3 w - - 0 1")
    idx = move_to_index(chess.Move.from_uci("a1a8"))
    move = index_to_move(idx, board)
    assert move.promotion is None


def test_uci_to_index_matches_move_to_index():
    assert uci_to_index("e2e4") == move_to_index(chess.Move.from_uci("e2e4"))


def test_mirror_indices_match_chess_square_mirror():
    """get_mirror_indices_np() is a lookup table built from
    chess.square_mirror() -- confirm it actually matches that function
    for a sample of squares, not just that it's internally consistent."""
    mirror = get_mirror_indices_np()
    for from_sq in [chess.A1, chess.E4, chess.H8, chess.D2]:
        for to_sq in [chess.A1, chess.E4, chess.H8, chess.D2]:
            idx = from_sq * 64 + to_sq
            expected = chess.square_mirror(from_sq) * 64 + chess.square_mirror(to_sq)
            assert mirror[idx] == expected


def test_mirror_policy_is_its_own_inverse():
    """Documented property: mirroring twice returns the original."""
    torch.manual_seed(0)
    policy = torch.rand(4096)
    twice = mirror_policy(mirror_policy(policy))
    assert torch.allclose(policy, twice)


def test_mirror_policy_handles_batched_input():
    torch.manual_seed(0)
    batch = torch.rand(4, 4096)
    mirrored = mirror_policy(batch)
    assert mirrored.shape == batch.shape
    assert torch.allclose(mirror_policy(mirrored), batch)
