"""
diagnose_knight_e4_anomaly.py — Why does search prefer c4e6 over d3e4 at
the "undefended_knight_on_e4" tactical benchmark position?

Established so far (10-11 Sept 2026):
  - The raw policy prior (no search) correctly picks d3e4 (captures the
    undefended knight).
  - MCTS search at both 200 and 600 simulations instead converges on
    c4e6, a non-capture bishop move to an empty square.
  - Stockfish confirms this isn't a matter of taste: d3e4 is +400cp for
    White, c4e6 is -297cp -- a ~700cp swing. c4e6 is a genuine blunder,
    not a defensible alternative plan.

This inspects the actual search tree at the root to see WHERE the
disagreement comes from: if c4e6's Q (the value estimate the search
built up for it, from evaluating positions in its subtree) is higher
than d3e4's despite the true outcome being much worse, that's a value-
head miscalibration specific to whatever continuation follows c4e6 --
different from a prior/exploration problem, which would show up as
c4e6 getting far more visits without a correspondingly higher Q.

Usage:
  venv/bin/python3 diagnose_knight_e4_anomaly.py [--sims 600]
"""

import argparse

import chess
import torch

from chessai.agent import ChessAgent
from chessai.moves import move_to_index, index_to_move
from run_config import CKPT_PATH

MOVES = ["e2e4", "e7e5", "g1f3", "f8c5", "f1c4", "g8f6", "d2d3", "f6e4"]
CORRECT_MOVE = "d3e4"
SEARCH_CHOSEN_MOVE = "c4e6"   # what the tactical benchmark reported at both 200 and 600 sims

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=None)
parser.add_argument("--sims", type=int, default=600)
args = parser.parse_args()


def main():
    ckpt_path = args.ckpt or CKPT_PATH
    device = torch.device("cpu")
    agent = ChessAgent(device, n_simulations=args.sims)
    try:
        agent.load(ckpt_path)
        print(f"Loaded {ckpt_path} ({agent.steps:,} steps)\n")
    except FileNotFoundError:
        print(f"No checkpoint at {ckpt_path} -- running against a fresh, "
              f"untrained network (fine for smoke-testing this script; "
              f"not meaningful for the actual question it's asking).\n")

    board = chess.Board()
    history = []
    for uci in MOVES:
        history = ([board.copy()] + history)[:3]
        board.push_uci(uci)

    print(f"Position after {' '.join(MOVES)}")
    print(f"To move: {'White' if board.turn else 'Black'}\n")

    # add_noise=False -- matches agent.choose_move(greedy=True)'s default
    # exactly, so this is the same tree shape the benchmark actually saw,
    # not a noisier exploratory version of it.
    policy, root = agent.mcts.search(board, history, args.sims, add_noise=False)

    rows = []
    for idx, child in root.children.items():
        move = index_to_move(idx, board)
        rows.append((move.uci(), child.N, child.Q, child.P))
    rows.sort(key=lambda r: -r[1])   # by visit count, most-visited first

    print(f"{'move':>8} {'N':>6} {'Q':>8} {'P':>8}")
    print("-" * 34)
    for uci, N, Q, P in rows[:10]:
        marker = ""
        if uci == CORRECT_MOVE:
            marker = "  <- objectively correct (Stockfish: +400cp)"
        elif uci == SEARCH_CHOSEN_MOVE:
            marker = "  <- what search chose (Stockfish: -297cp)"
        print(f"{uci:>8} {N:>6} {Q:>+8.4f} {P:>8.4f}{marker}")

    print(f"\n(showing top {min(10, len(rows))} of {len(rows)} legal moves by visit count)")

    correct_row = next((r for r in rows if r[0] == CORRECT_MOVE), None)
    chosen_row = next((r for r in rows if r[0] == SEARCH_CHOSEN_MOVE), None)
    if correct_row and chosen_row:
        print(f"\n{CORRECT_MOVE}: N={correct_row[1]} Q={correct_row[2]:+.4f} P={correct_row[3]:.4f}")
        print(f"{SEARCH_CHOSEN_MOVE}: N={chosen_row[1]} Q={chosen_row[2]:+.4f} P={chosen_row[3]:.4f}")
        if chosen_row[2] > correct_row[2]:
            print(f"\n-> {SEARCH_CHOSEN_MOVE}'s own Q is HIGHER than {CORRECT_MOVE}'s despite "
                  f"being objectively much worse -- this points at the VALUE HEAD "
                  f"misjudging positions reached after {SEARCH_CHOSEN_MOVE}, not an "
                  f"exploration/prior issue.")
        else:
            print(f"\n-> {CORRECT_MOVE}'s own Q is actually higher or equal -- if it still "
                  f"lost the visit-count contest, that points at exploration/PUCT "
                  f"dynamics (or too few total sims to converge) rather than the "
                  f"value head being wrong about either move directly.")


if __name__ == "__main__":
    main()
