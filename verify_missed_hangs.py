"""
verify_missed_hangs.py — Spot-check mine_real_hanging_pieces.py's "missed"
classifications with Stockfish, for knight-value-and-above pieces only.

mine_real_hanging_pieces.py flags a piece as "hanging" using a purely
material/defense rule (undefended, legally capturable) -- it can't tell
a genuine blunder from a case where NOT capturing was actually correct
(a real declined_capture-style trap, or some other tactical reason it
can't see). This takes a random sample of its "missed" bishop-value-
and-above instances and asks Stockfish directly: was the move HAL
actually played meaningfully worse than taking the free piece would
have been?

Usage:
  venv/bin/python3 verify_missed_hangs.py [games_csv ...] [--n 20] [--min-value 3]
"""

import argparse
import csv
import random

import chess
import chess.engine

from mine_real_hanging_pieces import scan, DEFAULT_GAMES_CSVS

ENGINE_PATH = "stockfish"
VALIDATE_DEPTH = 14   # shallower than the benchmark's ground-truth depth (16) --
                       # this is a bulk spot-check across many samples, not
                       # establishing any one position's canonical eval
BLUNDER_CP = 150       # "actually a real blunder", not just marginally worse


def _replay_to_ply(moves: list, ply: int) -> chess.Board:
    board = chess.Board()
    for uci in moves[:ply - 1]:
        board.push_uci(uci)
    return board


def _eval_after(engine, board: chess.Board, move_uci: str, mover: bool):
    b2 = board.copy()
    b2.push_uci(move_uci)
    if b2.is_game_over():
        return 10000 if b2.is_checkmate() else 0
    info = engine.analyse(b2, chess.engine.Limit(depth=VALIDATE_DEPTH))
    return info["score"].pov(mover).score(mate_score=10000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("games_csvs", nargs="*", default=DEFAULT_GAMES_CSVS)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--min-value", type=int, default=3)
    parser.add_argument("--min-ply", type=int, default=0,
                         help="only sample misses at this ply or later "
                              "(see mine_real_hanging_pieces.py's TEMP_MOVES note)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_opps = []
    moves_by_game = {}
    for path in args.games_csvs:
        try:
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        except FileNotFoundError:
            print(f"(skipping {path} -- not found)")
            continue
        for game_idx, row in enumerate(rows):
            moves_by_game[(path, game_idx)] = row["moves"].split()
        opps = scan(path)
        for o in opps:
            o["path"] = path
        all_opps.extend(opps)

    missed = [o for o in all_opps if not o["took_it"] and o["value"] >= args.min_value
              and o["ply"] >= args.min_ply]
    random.seed(args.seed)
    random.shuffle(missed)
    sample = missed[:args.n]

    print(f"Spot-checking {len(sample)} of {len(missed)} missed hangs "
          f"(value >= {args.min_value}) with Stockfish depth {VALIDATE_DEPTH}...\n")

    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    swings = []
    try:
        for o in sample:
            moves = moves_by_game[(o["path"], o["game"])]
            board = _replay_to_ply(moves, o["ply"])
            mover = board.turn

            cp_capture = _eval_after(engine, board, o["capture_moves"][0], mover)
            cp_played = _eval_after(engine, board, o["played"], mover)
            if cp_capture is None or cp_played is None:
                continue

            swing = cp_capture - cp_played   # positive = capturing would have been better
            swings.append(swing)
            tag = ("BLUNDER" if swing >= BLUNDER_CP else
                   "mild" if swing > 0 else "fine to decline")
            print(f"  {o['path']} game {o['game']} ply {o['ply']} [{o['piece']}]: "
                  f"capture={cp_capture:+d}cp played={cp_played:+d}cp "
                  f"swing={swing:+d}cp  {tag}")
    finally:
        engine.quit()

    if not swings:
        print("\nNo comparable samples (engine returned no usable eval for any of them).")
        return

    blunders = sum(1 for s in swings if s >= BLUNDER_CP)
    swings.sort()
    print(f"\n{blunders}/{len(swings)} sampled misses are confirmed real blunders "
          f"(capturing would have been >= {BLUNDER_CP}cp better than what was "
          f"played); median swing {swings[len(swings)//2]:+d}cp")


if __name__ == "__main__":
    main()
