"""
test_main_api.py — /move and /health validation.

10 Sept 2026 assessment: request.fen was documented as a sanity check but
never compared to the replayed board; no rejection of terminal positions
or out-of-range n_simulations; /health always reported "ok" regardless
of whether the checkpoint actually loaded. A follow-up review added the
halfmove-clock comparison (a real network input, encoder.py plane 53 --
not just bookkeeping like the fullmove number is).

main.py's module-level code already handles a missing checkpoint
gracefully (prints a warning, keeps an untrained network), so it's
directly importable here.
"""

import pytest
from fastapi import HTTPException

import main
from main import get_move, health, MoveRequest

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _call(**kwargs):
    """Call the route function directly, bypassing the HTTP layer --
    httpx/TestClient isn't installed in this venv, and the route function
    itself is a plain callable FastAPI hasn't done anything magic to."""
    req = MoveRequest(**kwargs)
    return get_move(req)


def test_matching_fen_and_moves_succeeds():
    resp = _call(fen=START_FEN, moves=[], n_simulations=5)
    assert resp.move
    assert not resp.done


def test_mismatched_position_is_rejected():
    with pytest.raises(HTTPException) as exc:
        _call(fen="8/8/8/8/8/8/8/8 w - - 0 1", moves=[], n_simulations=5)
    assert exc.value.status_code == 400
    assert "does not match" in exc.value.detail


def test_mismatched_halfmove_clock_is_rejected():
    """The halfmove clock is a real network input (encoder.py plane 53),
    not cosmetic like the fullmove number -- epd() alone doesn't catch a
    client desync here, so it's checked explicitly."""
    bad_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 5 1"
    with pytest.raises(HTTPException) as exc:
        _call(fen=bad_fen, moves=[], n_simulations=5)
    assert exc.value.status_code == 400


def test_mismatched_fullmove_number_only_is_accepted():
    """Fullmove number genuinely isn't an input anywhere -- a client
    disagreeing only on this field shouldn't be rejected."""
    fen_diff_fullmove = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 99"
    resp = _call(fen=fen_diff_fullmove, moves=[], n_simulations=5)
    assert resp.move


def test_terminal_position_is_rejected():
    checkmate_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    with pytest.raises(HTTPException) as exc:
        _call(fen=checkmate_fen, moves=["f2f3", "e7e5", "g2g4", "d8h4"],
              n_simulations=5)
    assert exc.value.status_code == 400
    assert "game over" in exc.value.detail.lower()


@pytest.mark.parametrize("n_sims", [0, -1, 100000])
def test_out_of_range_n_simulations_is_rejected(n_sims):
    with pytest.raises(HTTPException) as exc:
        _call(fen=START_FEN, moves=[], n_simulations=n_sims)
    assert exc.value.status_code == 400
    assert "n_simulations" in exc.value.detail


def test_health_reflects_checkpoint_loaded_state():
    """Drive main.checkpoint_loaded directly rather than relying on
    whatever this machine's disk happens to have -- checkpoints/ isn't
    git-synced, so whether a real checkpoint exists varies by machine
    (this Mac has none; the desktop that actually trains does), and the
    test shouldn't depend on that to be meaningful either way."""
    original = main.checkpoint_loaded
    try:
        main.checkpoint_loaded = False
        resp = health()
        assert resp["checkpoint_loaded"] is False
        assert resp["status"] == "degraded"

        main.checkpoint_loaded = True
        resp = health()
        assert resp["checkpoint_loaded"] is True
        assert resp["status"] == "ok"
    finally:
        main.checkpoint_loaded = original
