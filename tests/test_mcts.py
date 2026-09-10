"""
test_mcts.py — MCTS expansion, specifically the promotion prior-sum fix.

10 Sept 2026 assessment: all four promotion pieces collide on one policy
index (move_to_index only encodes from/to squares), so _expand_from_logits
computed a softmax entry per legal move BEFORE deduplicating -- one index
contributed to the softmax denominator up to four times, and the dict
write then silently discarded three of the four results. Net effect:
child priors at any node with a promotable pawn didn't sum to 1.
Reproduced at the time as 0.677 for a simple promotion position; fixed
by np.unique()-ing the indices before the softmax, not after.
"""

import numpy as np
import chess
import pytest
import torch

from chessai.mcts import MCTS, MCTSNode
from chessai.model import ChessNet


def _priors_sum(fen: str, seed: int) -> tuple:
    board = chess.Board(fen)
    mcts = MCTS(ChessNet(), torch.device("cpu"))
    node = MCTSNode(prior=1.0, parent=None)
    logits = np.random.RandomState(seed).randn(4096).astype(np.float32)
    mcts._expand_from_logits(node, board, logits)
    n_legal = len(list(board.legal_moves))
    return sum(c.P for c in node.children.values()), len(node.children), n_legal


def test_promotion_position_priors_sum_to_one():
    total, n_children, n_legal = _priors_sum(
        "8/P7/8/8/4k3/8/8/4K3 w - - 0 1", seed=0)
    # 9 legal moves (5 king moves + 4 promotion variants) collapse to 6
    # distinct actions -- the dict correctly dedupes on write; the bug was
    # in the softmax normalisation before that point, not the dict itself.
    assert n_legal == 9
    assert n_children == 6
    assert total == pytest.approx(1.0, abs=1e-4)


def test_ordinary_position_unaffected():
    """No promotable pawn -- every legal move should map to a distinct
    index, and this must keep working exactly as before the fix."""
    total, n_children, n_legal = _priors_sum(chess.STARTING_FEN, seed=1)
    assert n_children == n_legal == 20
    assert total == pytest.approx(1.0, abs=1e-4)


def test_black_to_move_promotion_uses_mirror_path_correctly():
    """Same collision, exercised through the mirror-index lookup
    _expand_from_logits uses for Black (the network always sees its own
    board mirrored to a 'White-like' perspective)."""
    total, n_children, n_legal = _priors_sum(
        "4k3/8/8/8/8/8/p7/4K3 b - - 0 1", seed=2)
    assert n_legal == 9
    assert n_children == 6
    assert total == pytest.approx(1.0, abs=1e-4)

