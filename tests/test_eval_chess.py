"""
test_eval_chess.py — end-reason correctness, --no-adjudication mode,
the eval_games.csv migration, and Tier 3's simulation-budget fairness.

Uses the eval_chess_module fixture (conftest.py) rather than a plain
import, since eval_chess.py builds a real ChessAgent at import time.
"""

import csv
import os

import chess


def _random_fn(rng):
    def fn(board, history):
        return rng.choice(list(board.legal_moves)).uci()
    return fn


def test_checkmate_is_reported_as_checkmate(eval_chess_module):
    mod = eval_chess_module

    def white_fn(board, history):
        return {0: "g2g4", 1: "f2f3"}[len(board.move_stack) // 2]

    def black_fn(board, history):
        return {0: "e7e5", 1: "d8h4"}[len(board.move_stack) // 2]

    result, end_reason, moves = mod.play_game(white_fn, black_fn)
    assert result == "0-1"
    assert end_reason == "checkmate"
    assert len(moves) == 4


def test_normal_mode_only_produces_known_end_reasons(eval_chess_module):
    mod = eval_chess_module
    import random
    rng = random.Random(7)
    fn = _random_fn(rng)

    seen = set()
    for _ in range(8):
        _, end_reason, _ = mod.play_game(fn, fn)
        seen.add(end_reason)

    assert seen <= {
        "checkmate", "rule_draw", "material_adjudication",
        "material_adjudication_moderate", "move_cap",
    }


def test_no_adjudication_mode_never_adjudicates_and_uses_the_larger_cap(eval_chess_module):
    mod = eval_chess_module
    import random
    rng = random.Random(3)
    fn = _random_fn(rng)

    saw_move_cap = False
    for _ in range(6):
        result, end_reason, moves = mod.play_game(fn, fn, no_adjudication=True)
        assert end_reason not in ("material_adjudication", "material_adjudication_moderate")
        if end_reason == "move_cap":
            saw_move_cap = True
            assert len(moves) == mod.MAX_GAME_MOVES_NO_ADJUDICATION
            assert result == "*"

    # not asserting saw_move_cap is True -- random play is noisy enough
    # that 6 games might all resolve early; this just checks that IF the
    # cap was hit, it was the larger one and correctly reported


def test_eval_log_migrates_an_old_format_file_in_place(eval_chess_module, tmp_path):
    mod = eval_chess_module
    old_path = os.path.join(tmp_path, "eval_games.csv")
    with open(old_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eval_game", "matchup", "result", "n_moves",
                    "hal_steps", "timestamp", "moves"])
        w.writerow(["1", "1. HAL (White) vs Random (Black)", "1-0", "10",
                    "100", "2026-01-01 00:00:00", "e2e4 e7e5"])

    mod._EVAL_LOG_PATH = old_path
    mod._init_eval_log()

    with open(old_path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == mod._EVAL_LOG_HEADER
    assert rows[1][rows[0].index("end_reason")] == "unknown"

    # idempotent: running it again on an already-migrated file is a no-op
    mod._init_eval_log()
    with open(old_path, newline="") as f:
        rows_again = list(csv.reader(f))
    assert rows_again == rows

    # no leftover temp file from the atomic write
    assert not os.path.exists(old_path + ".tmp")


def test_eval_log_appends_new_rows_with_real_end_reason(eval_chess_module, tmp_path):
    mod = eval_chess_module
    path = os.path.join(tmp_path, "eval_games.csv")
    mod._EVAL_LOG_PATH = path
    mod._eval_game_num = 0
    mod._init_eval_log()

    mod._log_eval_game("test matchup", "1-0", "checkmate", ["e2e4", "e7e5"])

    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, row = rows[0], rows[1]
    assert row[header.index("end_reason")] == "checkmate"
    assert row[header.index("result")] == "1-0"


def test_tier3_sim_budget_is_equalised(eval_chess_module):
    """10 Sept 2026 assessment: hal_move (used elsewhere for the cheap
    vs-random tier) never overrides hal.n_simulations, so pairing it
    against hal_prev (built with N_SIMS_PREV) gave the current checkpoint
    half the search budget. hal_move_at() is the fix -- confirm it
    actually sets and restores the simulation count around each call."""
    mod = eval_chess_module
    original = mod.hal.n_simulations
    move_fn = mod.hal_move_at(mod.N_SIMS_PREV)

    seen_during_call = {}

    def spy_choose_move(board, history, greedy=True):
        seen_during_call["n_simulations"] = mod.hal.n_simulations
        return "e2e4", None, 0.0

    mod.hal.choose_move = spy_choose_move
    board = chess.Board()
    move_fn(board, [])

    assert seen_during_call["n_simulations"] == mod.N_SIMS_PREV
    assert mod.hal.n_simulations == original   # restored after the call
