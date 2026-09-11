"""
check_leaves_against_stockfish.py — Step 3 of the search-misranking
investigation (paper/value_head_small_material_options.md, Sec 6/7).

Step 1 (inspect_two_search_failures.py) found that for
undefended_knight_on_e4, the network's raw value one ply deep already
disagrees with Stockfish as badly as search's final Q does. This script
checks whether that miscalibration is a single bad read at that one leaf,
or persists through the specific continuation search actually built its
232 (or 79) visits' worth of evidence on.

For every node in the tree under the correct move and under search's
chosen move -- down to --depth plies, restricted to nodes search visited
at least --min-visits times, so we're only looking at positions search's
evidence actually rests on, not the long tail of barely-explored moves --
this prints three numbers for that exact position, all in the same
convention (value for the side that just moved into it, matching how
node.Q is defined in chessai/mcts.py's _backup):

  Q          search's own accumulated estimate for that node
  net        the raw network value AT THAT EXACT POSITION, no further
             search -- what step 1 checked one ply deep, now checked at
             every visited depth, with the correct 3-board history at
             each node (not just the depth-0 approximation)
  stockfish  Stockfish's static evaluation of that exact position, same
             perspective, at VALIDATE_DEPTH (matching the depth
             generate_tactical_benchmark.py used to establish this
             benchmark's own ground truth)

If `net` and `stockfish` disagree at the same nodes where `Q` disagrees
with `stockfish`, the value head's miscalibration runs through the whole
line, not just the first ply. If `net` matches `stockfish` fine at deeper
nodes but not at the first ply, that would point at something specific to
the immediate post-capture position rather than a general problem along
the line.

Usage:
  venv/bin/python3 check_leaves_against_stockfish.py [--sims 600] [--depth 4] [--min-visits 5]
"""

import argparse

import chess
import chess.engine
import torch

from chessai.agent import ChessAgent
from chessai.encoder import encode
from chessai.moves import index_to_move
from run_config import CKPT_PATH

ENGINE_PATH = "stockfish"
VALIDATE_DEPTH = 16   # matches generate_tactical_benchmark.py's own ground-truth depth
MATE_STANDIN = 10.0   # arbitrary large finite value/-value used in place of a mate score

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
parser.add_argument("--depth", type=int, default=4,
                     help="plies below the candidate move to inspect")
parser.add_argument("--min-visits", type=int, default=5,
                     help="only inspect nodes search visited at least this often")
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


def _last_movers_perspective_net_value(agent, board, history):
    """Raw value-head output at this exact position, flipped from the
    network's own to-move convention to the perspective of whoever just
    moved -- the same convention node.Q uses (see module docstring)."""
    state = encode([board] + history).unsqueeze(0)
    with torch.inference_mode():
        _, value = agent.network(state)
    return -float(value.squeeze(0).item())


def _last_movers_perspective_stockfish(engine, board):
    if board.is_game_over():
        return None
    info = engine.analyse(board, chess.engine.Limit(depth=VALIDATE_DEPTH))
    score = info["score"].pov(not board.turn)   # not board.turn == whoever just moved
    if score.is_mate():
        return MATE_STANDIN if score.mate() > 0 else -MATE_STANDIN
    cp = score.score()
    return cp / 100.0 if cp is not None else None


def _collect_nodes(start_node, start_board, start_history, min_visits, max_depth):
    """BFS from the candidate move's own node (depth 0 = the position right
    after that move), collecting every node visited at least min_visits
    times, up to max_depth plies further -- each with the exact 3-board
    history encode() needs, carried along the path rather than
    approximated."""
    results = [(0, [], start_board, start_history, start_node)]
    frontier = [(0, [], start_board, start_history, start_node)]
    while frontier:
        depth, path, board, history, node = frontier.pop(0)
        if depth >= max_depth or not node.is_expanded or board.is_game_over():
            continue
        for idx, child in node.children.items():
            if child.N < min_visits:
                continue
            move = index_to_move(idx, board)
            new_history = ([board.copy()] + history)[:3]
            new_board = board.copy()
            new_board.push(move)
            new_path = path + [move.uci()]
            results.append((depth + 1, new_path, new_board, new_history, child))
            frontier.append((depth + 1, new_path, new_board, new_history, child))
    return results


def _print_case(agent, engine, name, moves, correct_uci, sims, depth, min_visits):
    board, history = _replay_with_history(moves)
    print(f"=== {name} ===\n")

    policy, root = agent.mcts.search(board, history, sims, add_noise=False)
    rows = _root_rows(root, board)
    search_uci = max(root.children, key=lambda i: root.children[i].N)
    search_uci = index_to_move(search_uci, board).uci()

    for label, uci in [("correct", correct_uci), ("search chose", search_uci)]:
        if uci not in rows:
            continue
        node = rows[uci]
        b2 = board.copy()
        h2 = ([board.copy()] + history)[:3]
        b2.push_uci(uci)

        print(f"--- under {label} move {uci} ---")
        print(f"{'path':>28} {'N':>6} {'Q':>8} {'net':>8} {'stockfish':>10}")

        nodes = _collect_nodes(node, b2, h2, min_visits, depth)
        for d, path, b, h, n in nodes:
            net_val = _last_movers_perspective_net_value(agent, b, h)
            sf_val = _last_movers_perspective_stockfish(engine, b)
            path_str = uci if not path else f"{uci} {' '.join(path)}"
            sf_str = f"{sf_val:+.2f}" if sf_val is not None else "n/a"
            print(f"{path_str:>28} {n.N:>6} {n.Q:>+8.4f} {net_val:>+8.3f} {sf_str:>10}")
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

    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    try:
        for case in CASES:
            _print_case(agent, engine, case["name"], case["moves"],
                        case["correct_move"], args.sims, args.depth, args.min_visits)
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
