"""
test_tactical_benchmark.py — integrity checks on chessai/tactical_benchmark.json.

Doesn't re-run Stockfish validation (that's generate_tactical_benchmark.py's
job, and it's slow) -- this guards against the JSON silently drifting out
of a valid state if it's ever hand-edited or regenerated with a bug:
every position must still replay legally, every category's correct_moves
must actually be legal at that position, and mate_in_1's claims must
still be real checkmates (this one IS cheap to re-verify, python-chess
only, no engine needed).
"""

import json

import chess

BENCHMARK_PATH = "chessai/tactical_benchmark.json"


def _load():
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


def _replay(move_seq):
    board = chess.Board()
    for uci in move_seq:
        board.push_uci(uci)
    return board


def test_benchmark_file_is_non_empty_and_covers_expected_categories():
    positions = _load()
    assert len(positions) > 0
    categories = {p["category"] for p in positions}
    assert categories <= {"hanging_piece", "mate_in_1", "defend_mate_in_1", "conversion"}


def test_every_position_move_sequence_is_legal():
    for pos in _load():
        # _replay raises if any move is illegal -- that's the check
        board = _replay(pos["moves"])
        assert not board.is_game_over(), (
            f"{pos['name']}: game is already over before HAL gets a move"
        )


def test_choice_based_categories_have_legal_correct_moves():
    for pos in _load():
        if pos["category"] not in ("hanging_piece", "mate_in_1", "defend_mate_in_1"):
            continue
        board = _replay(pos["moves"])
        legal = {m.uci() for m in board.legal_moves}
        assert pos["correct_moves"], f"{pos['name']}: no correct_moves recorded"
        for uci in pos["correct_moves"]:
            assert uci in legal, f"{pos['name']}: {uci} is not legal here"


def test_mate_in_1_moves_are_real_checkmates():
    for pos in _load():
        if pos["category"] != "mate_in_1":
            continue
        board = _replay(pos["moves"])
        for uci in pos["correct_moves"]:
            board.push_uci(uci)
            assert board.is_checkmate(), f"{pos['name']}: {uci} isn't actually mate"
            board.pop()


def test_defend_mate_in_1_moves_actually_avoid_a_mate():
    for pos in _load():
        if pos["category"] != "defend_mate_in_1":
            continue
        board = _replay(pos["moves"])
        for uci in pos["correct_moves"]:
            board.push_uci(uci)
            opponent_has_mate = any(_delivers_mate(board, m) for m in board.legal_moves)
            board.pop()
            assert not opponent_has_mate, (
                f"{pos['name']}: {uci} was marked as a defence but the "
                f"opponent still has a mate-in-1 afterward"
            )


def _delivers_mate(board, move):
    board.push(move)
    result = board.is_checkmate()
    board.pop()
    return result


def test_conversion_positions_have_a_reference_eval():
    for pos in _load():
        if pos["category"] != "conversion":
            continue
        assert "reference_cp" in pos
        assert pos["reference_cp"] >= 300
