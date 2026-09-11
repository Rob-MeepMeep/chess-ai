"""
inspect_two_search_failures.py — Step 1 of the search-misranking
investigation (paper/value_head_small_material_options.md, §6): a closer
look at the two policy-correct/search-wrong hanging-piece positions
(undefended_knight_on_e4, undefended_knight_on_d5_black_captures).

For both the correct move and search's chosen move at each position,
records:
  - the root child's N, Q (search's accumulated estimate) and P (prior)
  - the raw network value one ply deeper, with NO search at all -- what
    the value head says about the resulting position by itself, flipped
    to the same "good for the side that just moved" perspective the root
    Q values use, so it's directly comparable to them
  - a representative continuation: the highest-N line search actually
    built under that child, a few plies deep, so we can see what search
    "believes" happens next rather than just the summary Q number

Why this comparison: if the shallow, no-search network value already
disagrees with Stockfish the same way search's Q does, that points at
the value head itself, already visible one ply down. If the shallow
value agrees with Stockfish but search's Q doesn't, the discrepancy is
coming from something in search -- exploration, visit allocation, depth,
batching -- rather than a bad leaf evaluation at this depth. This script
doesn't decide between those on its own; it produces the readout the
comparison needs.

Usage:
  venv/bin/python3 inspect_two_search_failures.py [--sims 600] [--depth 5]
"""

import argparse

import chess
import torch

from chessai.agent import ChessAgent
from chessai.encoder import encode
from chessai.moves import move_to_index, index_to_move, get_mirror_indices_np
from run_config import CKPT_PATH

CASES = [
    {
        "name": "undefended_knight_on_e4",
        "moves": ["e2e4", "e7e5", "g1f3", "f8c5", "f1c4", "g8f6", "d2d3", "f6e4"],
        "correct_move": "d3e4",
    },
    {
        "name": "undefended_knight_on_d5_black_captures",
        "moves": ["b1c3", "g8f6", "c3d5"],
        "correct_move": "f6d5",
    },
]

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=None)
parser.add_argument("--sims", type=int, default=600)
parser.add_argument("--depth", type=int, default=5,
                     help="plies of continuation to print under each move")
args = parser.parse_args()


def _replay_with_history(move_seq):
    board = chess.Board()
    history = []
    for uci in move_seq:
        history = ([board.copy()] + history)[:3]
        board.push_uci(uci)
    return board, history


def _root_rows(root, board):
    rows = {}
    for idx, child in root.children.items():
        move = index_to_move(idx, board)
        rows[move.uci()] = child
    return rows


def _network_value(agent, board, history):
    """Raw value-head output for this exact position, no search -- from
    the perspective of the side to move IN this position (network
    convention), not yet flipped to any other perspective."""
    state = encode([board] + history).unsqueeze(0)
    with torch.inference_mode():
        _, value = agent.network(state)
    return float(value.squeeze(0).item())


def _continuation(node, board, history, depth):
    """Follow the highest-N child at each step -- the line search actually
    built the most evidence for under this node -- up to `depth` plies or
    until an unexpanded/terminal node is reached. Returns a list of
    (uci, N, Q, P) for each ply taken."""
    line = []
    b = board.copy()
    h = list(history)
    n = node
    for _ in range(depth):
        if not n.is_expanded or not n.children or b.is_game_over():
            break
        best_idx = max(n.children, key=lambda i: n.children[i].N)
        child = n.children[best_idx]
        move = index_to_move(best_idx, b)
        line.append((move.uci(), child.N, child.Q, child.P))
        h = ([b.copy()] + h)[:3]
        b.push(move)
        n = child
    return line


def _print_case(agent, name, moves, correct_uci, sims, depth):
    board, history = _replay_with_history(moves)
    mover = "White" if board.turn == chess.WHITE else "Black"
    print(f"=== {name} ===")
    print(f"To move: {mover}  (root Q below is from {mover}'s perspective)\n")

    policy, root = agent.mcts.search(board, history, sims, add_noise=False)
    rows = _root_rows(root, board)
    search_uci = max(root.children, key=lambda i: root.children[i].N)
    search_uci = index_to_move(search_uci, board).uci()

    for label, uci in [("correct", correct_uci), ("search chose", search_uci)]:
        if uci not in rows:
            print(f"{label:>13}: {uci} -- not a root child (unexpected)")
            continue
        child = rows[uci]

        b2 = board.copy()
        h2 = ([board.copy()] + history)[:3]
        b2.push_uci(uci)
        if b2.is_game_over():
            raw_value_note = "(game over -- no value-head read)"
            flipped = None
        else:
            raw = _network_value(agent, b2, h2)
            flipped = -raw   # network's own-perspective value -> mover's perspective
            raw_value_note = f"raw(own-persp)={raw:+.4f}"

        print(f"{label:>13}: {uci}   N={child.N:<5} Q={child.Q:+.4f} P={child.P:.4f}"
              + (f"   1-ply net value (mover's persp)={flipped:+.4f}  {raw_value_note}"
                 if flipped is not None else f"   {raw_value_note}"))

        line = _continuation(child, b2, h2, depth - 1)
        if line:
            line_str = " ".join(f"{u}(N={n},Q={q:+.3f})" for u, n, q, p in line)
            print(f"              continuation search favours: {uci} {line_str}")
        else:
            print(f"              continuation search favours: {uci} (no further expansion)")
        print()

    print()


def main():
    ckpt_path = args.ckpt or CKPT_PATH
    device = torch.device("cpu")
    agent = ChessAgent(device, n_simulations=args.sims)
    try:
        agent.load(ckpt_path)
        print(f"Loaded {ckpt_path} ({agent.steps:,} steps)\n")
    except FileNotFoundError:
        print(f"No checkpoint at {ckpt_path} -- running against a fresh, "
              f"untrained network (not meaningful, just exercises the script).\n")

    for case in CASES:
        _print_case(agent, case["name"], case["moves"], case["correct_move"],
                    args.sims, args.depth)


if __name__ == "__main__":
    main()
