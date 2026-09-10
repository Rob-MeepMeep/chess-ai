"""
conftest.py — shared pytest fixtures for the chess-ai test suite.

eval_chess.py and main.py both build a ChessAgent and try to load the
active run's checkpoint at import time. Neither this repo nor CI has one
(checkpoints/ isn't git-synced, chess training never happens on this
machine) -- both already handle that gracefully (a printed warning, an
untrained network), so tests run against a fresh random-weight network.
That's fine for everything here: none of these tests assert anything
about learned behaviour, only about the mechanics (validation, end-reason
bookkeeping, persistence, prior normalisation) around it.
"""

import sys

import pytest


@pytest.fixture
def eval_chess_module():
    """
    Import eval_chess.py without running its (slow, Stockfish-dependent)
    evaluation suite. --regression-only in sys.argv is required even just
    to import cleanly on a machine with no checkpoint -- see eval_chess.py's
    "Load HAL" section, which exit(1)s otherwise (pre-existing, intentional
    CLI behaviour for a real invocation, not something this fixture works
    around so much as respects).
    """
    old_argv = sys.argv
    sys.argv = ["eval_chess.py", "--regression-only"]
    try:
        if "eval_chess" in sys.modules:
            del sys.modules["eval_chess"]
        import eval_chess
        return eval_chess
    finally:
        sys.argv = old_argv
