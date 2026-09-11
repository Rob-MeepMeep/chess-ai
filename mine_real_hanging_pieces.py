"""
mine_real_hanging_pieces.py — Step 4 (Option 4) of the search-misranking
investigation (paper/value_head_small_material_options.md, Sec 6/8): does
search actually leave real pieces hanging in HAL's own self-play games,
or is this only a hand-authored-benchmark phenomenon?

Scans a games.csv (the same self-play move log relabel_with_stockfish.py
and generate_material_probe_positions.py already read -- one row per
game, a space-separated "moves" column of UCI moves) ply by ply. At every
position, finds every enemy piece the side to move has a legal capture
against where that piece's own square isn't defended by its own side
(the same "genuinely free, not just any capture" check
generate_tactical_benchmark.py's new candidates were validated with).
Records whether the move HAL actually played that ply took the most
valuable one available.

Self-play in this project is one network playing both colours, so "did
HAL take it" is meaningful regardless of which side the opportunity
belonged to -- there's no second player to blame it on.

Caveats this doesn't resolve: this checks the MOST valuable hanging
piece only, so a real capture of a second, less valuable hanging piece
in the same position still counts as "missed" here -- a conservative
simplification, not a bug, but worth knowing when reading the numbers.
It also can't say WHY a miss happened (this is exactly what step 1-3's
value-head/search-dynamics diagnosis was for) -- it only answers whether
the tactical-benchmark pattern shows up in real play at all.

Usage:
  venv/bin/python3 mine_real_hanging_pieces.py [games_csv ...]
    defaults to logs/run19/games.csv logs/run20/games.csv
"""

import argparse
import csv

import chess

TEMP_MOVES = 30   # chessai/agent.py's own constant: self-play SAMPLES from the
                   # visit distribution (plus root Dirichlet noise) for the
                   # first TEMP_MOVES plies of every game, then switches to
                   # greedy argmax. A "miss" before that point may reflect
                   # intentional training-time exploration, not search's best
                   # judgement -- see the --min-ply option below.

DEFAULT_GAMES_CSVS = ["logs/run19/games.csv", "logs/run20/games.csv"]

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9,
}


def _hanging_targets(board: chess.Board) -> list:
    """Every square holding an enemy piece (from the mover's perspective)
    that the mover has a legal capture against and that isn't defended by
    its own side -- a genuinely free capture, not just any capture.
    Returns (square, piece_type, [capturing move ucis]) tuples."""
    opponent = not board.turn
    capturing_by_square: dict = {}
    for move in board.legal_moves:
        victim = board.piece_at(move.to_square)
        if victim is not None and victim.color == opponent:
            capturing_by_square.setdefault(move.to_square, []).append(move.uci())

    targets = []
    for square, ucis in capturing_by_square.items():
        if board.is_attacked_by(opponent, square):
            continue   # defended -- not free
        piece = board.piece_at(square)
        targets.append((square, piece.piece_type, ucis))
    return targets


def scan(games_csv: str) -> list:
    opportunities = []
    with open(games_csv, newline="") as f:
        reader = csv.DictReader(f)
        for game_idx, row in enumerate(reader):
            moves = row.get("moves", "").split()
            board = chess.Board()
            for ply, move_uci in enumerate(moves, start=1):
                targets = _hanging_targets(board)
                if targets:
                    square, ptype, ucis = max(targets, key=lambda t: PIECE_VALUES[t[1]])
                    opportunities.append({
                        "game": game_idx,
                        "ply": ply,
                        "piece": chess.piece_name(ptype),
                        "value": PIECE_VALUES[ptype],
                        "took_it": move_uci in ucis,
                        "played": move_uci,
                        "capture_moves": ucis,
                    })
                try:
                    board.push_uci(move_uci)
                except Exception:
                    break
    return opportunities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("games_csvs", nargs="*", default=DEFAULT_GAMES_CSVS)
    parser.add_argument("--min-ply", type=int, default=0,
                         help=f"only count opportunities at this ply or later "
                              f"(pass {TEMP_MOVES + 1} to look only at the "
                              f"greedy phase of self-play, past TEMP_MOVES)")
    args = parser.parse_args()

    all_opps = []
    for path in args.games_csvs:
        try:
            opps = scan(path)
        except FileNotFoundError:
            print(f"(skipping {path} -- not found)")
            continue
        if args.min_ply:
            opps = [o for o in opps if o["ply"] >= args.min_ply]
        print(f"{path}: {len(opps)} hanging-piece opportunities found"
              + (f" (ply >= {args.min_ply})" if args.min_ply else ""))
        all_opps.extend(opps)

    if not all_opps:
        print("\nNo hanging-piece opportunities found in the given "
              "games.csv file(s) (or none of them exist here).")
        return

    print(f"\n{len(all_opps)} total opportunities across all games.\n")

    # Grouped by piece, not by value -- knight and bishop share value=3, and
    # grouping by value alone silently merged them under whichever piece name
    # happened to appear first, hiding that they're tactically distinct
    # pieces with (it turns out) noticeably different take rates.
    by_piece: dict = {}
    for opp in all_opps:
        by_piece.setdefault(opp["piece"], []).append(opp)

    print(f"{'value':>6} {'piece':>8} {'n':>6} {'took it':>9} {'missed':>8} {'take rate':>10}")
    for piece_name in sorted(by_piece, key=lambda p: -by_piece[p][0]["value"]):
        group = by_piece[piece_name]
        value = group[0]["value"]
        n = len(group)
        took = sum(o["took_it"] for o in group)
        print(f"{value:>6} {piece_name:>8} {n:>6} {took:>9} {n - took:>8} {100*took/n:>9.1f}%")

    total_took = sum(o["took_it"] for o in all_opps)
    print(f"\nOverall: {total_took}/{len(all_opps)} ({100*total_took/len(all_opps):.1f}%) "
          f"of hanging-piece opportunities were captured when available.")
    print("Read the pawn row with real skepticism: 'undefended and capturable' "
          "doesn't mean 'good to take' the way it does for the hand-authored "
          "benchmark -- a materially free pawn in a real game can come with a "
          "legitimate positional reason to decline (king safety, development, "
          "structure) that this purely material/defense-based check can't see. "
          "Knight-value and above are much harder to have an innocent reason "
          "to leave alone.")

    print("\nMissed examples worth spot-checking against the tactical "
          "benchmark's pattern (knight value or above):")
    missed = [o for o in all_opps if not o["took_it"] and o["value"] >= 3]
    for o in missed[:15]:
        print(f"  game {o['game']} ply {o['ply']}: {o['piece']} was free "
              f"({'/'.join(o['capture_moves'])}), HAL played {o['played']} instead")
    if not missed:
        print("  (none)")


if __name__ == "__main__":
    main()
