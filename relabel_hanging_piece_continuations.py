"""
relabel_hanging_piece_continuations.py — Build a Stockfish-labelled set of
"just captured material, what now" continuation positions, targeting
exactly the pattern the search-misranking investigation found (paper/
value_head_small_material_options.md, Option 2, decided after steps 1-4
confirmed the pattern generalises to real games at scale, Sec 9-10).

Why this shape, not another position dump: steps 1 and 3 of that
investigation found the value head's own one-ply (and deeper) read of
positions immediately after a hanging-piece capture is frequently wrong
-- not the decision to capture itself, the read of the RESULTING
position. This targets exactly that: for every real knight/bishop-value
hanging-piece opportunity found in HAL's own self-play (reusing
mine_real_hanging_pieces.py's exact detection -- same is_attacked_by
check the tactical benchmark's own new positions were validated with),
generates TWO labelled continuations, not one:
  - the position right after CAPTURING the piece (always generated, even
    when HAL didn't actually play it -- we want the network to see "if
    I'd taken it, here's how good this is" regardless of what actually
    happened in the game)
  - the position right after whatever HAL ACTUALLY played, when that
    was a miss (a real blunder continuation, labelled with Stockfish's
    real, bad evaluation)
Both good and bad continuations, per the reviewer's explicit instruction
(11 Sept 2026) -- not just positions after successful captures, which
would just teach "captures are good" again rather than "read this
resulting position correctly."

Scope: knight and bishop only (value=3), not rook/queen (real-game take
rates there are already 75%+, weakest evidence for a problem) and not
pawns (a materially-free pawn often has a legitimate positional reason
to decline that this can't distinguish from a real blunder -- see
mine_real_hanging_pieces.py's own caveat).

Output format matches relabel_with_stockfish.py's raw (state, policy, z,
sf_value) tuples exactly, so curate_buffer.py's existing blend-at-build-
time design applies unchanged -- just with its own separate alpha
constant (HANGING_PIECE_ALPHA), so it can be retuned independently
without re-running Stockfish, same reasoning as the original split.

Usage:
  venv/bin/python3 relabel_hanging_piece_continuations.py

Requires Stockfish on PATH. One-time offline cost, not part of the
training loop -- same convention as relabel_with_stockfish.py.
"""

import random

import chess
import chess.engine
import torch

from chessai.encoder import encode
from mine_real_hanging_pieces import _hanging_targets, PIECE_VALUES
from relabel_with_stockfish import _load_games, _stockfish_value

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GAMES_CSVS      = ["logs/run19/games.csv", "logs/run20/games.csv", "logs/run21/games.csv"]
OUTPUT_PATH     = "checkpoints/stockfish_relabeled_hanging_pieces_raw.pt"
STOCKFISH_PATH  = "stockfish"
STOCKFISH_DEPTH = 16   # matches relabel_with_stockfish.py's ground-truth depth

TARGET_VALUES        = {3}   # knight + bishop only -- see module docstring
MIN_MOVE_PLY          = 20   # skip pure opening theory, same convention as
                              # relabel_with_stockfish.py
MAX_SAMPLES_PER_GAME  = 4    # cap so no single game dominates the set

TARGET_POSITIONS = 4_000


def main():
    print("Loading self-play games...")
    games = _load_games(GAMES_CSVS)
    random.shuffle(games)
    print(f"  {len(games):,} total games available\n")

    print(f"Starting Stockfish (depth {STOCKFISH_DEPTH})...")
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 1, "Hash": 64})

    positions = []
    counts = {"capture_knight": 0, "capture_bishop": 0,
              "decline_knight": 0, "decline_bishop": 0}

    try:
        for game in games:
            if len(positions) >= TARGET_POSITIONS:
                break

            board = chess.Board()
            history = []
            winner = chess.WHITE if game["outcome"] == "W" else chess.BLACK
            candidates = []   # (kind, board_after, history_after, piece_type)

            for ply, move_uci in enumerate(game["moves"], start=1):
                if ply >= MIN_MOVE_PLY:
                    targets = [(sq, pt, ucis) for sq, pt, ucis in _hanging_targets(board)
                               if PIECE_VALUES[pt] in TARGET_VALUES]
                    if targets:
                        square, ptype, ucis = max(targets, key=lambda t: PIECE_VALUES[t[1]])
                        capture_uci = ucis[0]
                        took_it = move_uci in ucis
                        hist_before = ([board.copy()] + history)[:3]

                        b_capture = board.copy()
                        b_capture.push_uci(capture_uci)
                        if not b_capture.is_game_over():
                            candidates.append(("capture", b_capture, hist_before, ptype))

                        if not took_it:
                            b_actual = board.copy()
                            b_actual.push_uci(move_uci)
                            if not b_actual.is_game_over():
                                candidates.append(("decline", b_actual, hist_before, ptype))

                history = ([board.copy()] + history)[:3]
                try:
                    board.push_uci(move_uci)
                except Exception:
                    break

            random.shuffle(candidates)
            for kind, pos_board, pos_history, ptype in candidates[:MAX_SAMPLES_PER_GAME]:
                state = encode([pos_board] + pos_history)
                policy = torch.zeros(4096)

                z = 1.0 if pos_board.turn == winner else -1.0
                sf_value = _stockfish_value(engine, pos_board, STOCKFISH_DEPTH)

                positions.append((state, policy, z, sf_value))
                counts[f"{kind}_{chess.piece_name(ptype)}"] += 1
                if len(positions) % 200 == 0:
                    print(f"  {len(positions):,} positions evaluated...")

                if len(positions) >= TARGET_POSITIONS:
                    break
    finally:
        engine.quit()

    print(f"\nFinal set: {len(positions):,} positions")
    for key, n in counts.items():
        print(f"  {key}: {n:,}")

    torch.save(positions, OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")
    print("Run curate_buffer.py next (once wired to load this file) to fold "
          "these into the next run's seed buffer.")


if __name__ == "__main__":
    main()
