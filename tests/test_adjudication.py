"""
test_adjudication.py — SelfPlayGame.update_streaks() and _finish_game()
from train_chess.py.

10 Sept 2026 assessment: streak counters tracked magnitude only, never
checked the SAME side stayed favoured across consecutive qualifying
plies -- a lead flipping sides mid-streak could satisfy the threshold
even though neither side actually held it long enough. These call the
real, refactored production code (not a re-implementation) -- see
train_chess.py's main()-extraction commit for why that import is safe.
"""

import chess

import train_chess as tc
from chessai.replay import ReplayBuffer


def _drive_material_streak(g, boards, values=None):
    """Feed a sequence of board states through update_streaks(), as if
    each were the position after one more ply."""
    values = values or [0.0] * len(boards)
    for board, v in zip(boards, values):
        g.board = board
        g.moves.append("dummy")   # only length matters for MIN_MOVE gating
        g.update_streaks(v)


def test_material_streak_resets_on_side_flip_not_just_magnitude():
    g = tc.SelfPlayGame()
    g.moves = ["dummy"] * tc.MATERIAL_ADJUDICATE_MIN_MOVE   # already past the gate

    white_ahead = chess.Board("4k3/8/8/8/8/8/8/QRBNK3 w - - 0 1")   # huge white lead
    black_ahead = chess.Board("qrbnk3/8/8/8/8/8/8/4K3 w - - 0 1")   # huge black lead

    _drive_material_streak(g, [white_ahead] * 3)
    assert g.material_streak == 3
    assert g.material_streak_side == chess.WHITE

    _drive_material_streak(g, [black_ahead] * 3)
    # flip resets to a fresh streak of 1 for the new side, not a
    # continuation -- 3 (white) + 3 (black) must NOT reach the threshold
    assert g.material_streak == 3
    assert g.material_streak_side == chess.BLACK
    assert g.material_streak < tc.MATERIAL_ADJUDICATE_STREAK
    assert not g.over


def test_material_streak_reaches_threshold_when_one_side_actually_sustains_it():
    g = tc.SelfPlayGame()
    g.moves = ["dummy"] * tc.MATERIAL_ADJUDICATE_MIN_MOVE

    white_ahead = chess.Board("4k3/8/8/8/8/8/8/QRBNK3 w - - 0 1")
    _drive_material_streak(g, [white_ahead] * tc.MATERIAL_ADJUDICATE_STREAK)

    assert g.material_streak == tc.MATERIAL_ADJUDICATE_STREAK
    assert g.material_streak_side == chess.WHITE
    assert g.over


def test_material_streak_does_not_start_before_min_move():
    g = tc.SelfPlayGame()
    g.moves = ["dummy"] * (tc.MATERIAL_ADJUDICATE_MIN_MOVE - 1)   # not past the gate yet
    white_ahead = chess.Board("4k3/8/8/8/8/8/8/QRBNK3 w - - 0 1")
    g.board = white_ahead
    g.update_streaks(0.0)
    assert g.material_streak == 0


def test_resign_streak_resets_on_side_flip():
    g = tc.SelfPlayGame()
    g.board = chess.Board()

    # White (to move) very confident, three plies
    for _ in range(3):
        g.update_streaks(v=0.99)
    assert g.resign_streak == 3
    assert g.resign_streak_side == chess.WHITE   # g.board.turn is WHITE throughout (board never pushed here)

    # now the opposite side becomes the confident one -- v<0 means the
    # OTHER colour is favoured, which is a flip from White
    for _ in range(3):
        g.update_streaks(v=-0.99)
    assert g.resign_streak == 3   # reset to a fresh streak of 3, not 6
    assert g.resign_streak_side == chess.BLACK
    assert g.resign_streak < tc.RESIGN_CONSECUTIVE


def test_finish_game_checkmate_outranks_streaks():
    """A real board result always wins, even if a streak also qualifies --
    the board is the ground truth, not the value head's opinion."""
    g = tc.SelfPlayGame()
    # Fool's mate: checkmate delivered, board.is_game_over() is True
    g.board = chess.Board()
    for uci in ["g2g4", "e7e5", "f2f3", "d8h4"]:
        g.board.push_uci(uci)
        g.moves.append(uci)
    # also artificially satisfy a streak, to prove checkmate still wins
    g.material_streak = tc.MATERIAL_ADJUDICATE_STREAK
    g.material_streak_side = chess.WHITE

    replay = ReplayBuffer(capacity=10)
    winner, end_reason = tc._finish_game(g, replay)

    assert end_reason == "checkmate"
    assert winner == chess.BLACK   # Black delivered mate


def test_finish_game_material_adjudication_winner_matches_current_board():
    g = tc.SelfPlayGame()
    g.board = chess.Board("4k3/8/8/8/8/8/8/QRBNK3 w - - 0 1")   # white way ahead, not game over
    g.material_streak = tc.MATERIAL_ADJUDICATE_STREAK
    g.material_streak_side = chess.WHITE

    replay = ReplayBuffer(capacity=10)
    winner, end_reason = tc._finish_game(g, replay)

    assert end_reason == "material_adjudication"
    assert winner == chess.WHITE


def test_finish_game_cap_draw_uses_material_when_no_streak_qualifies():
    g = tc.SelfPlayGame()
    g.board = chess.Board("4k3/8/8/8/8/8/8/RBNK4 w - - 0 1")   # white ahead but no streak recorded
    replay = ReplayBuffer(capacity=10)
    winner, end_reason = tc._finish_game(g, replay)

    assert end_reason == "cap_draw"
    assert winner == chess.WHITE   # abs(material) > 3, soft win rather than a draw
