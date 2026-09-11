"""
diagnose_search_disagreements.py — For every tactical-benchmark position
where search's choice differs from the correct answer, show WHY: the
root's per-move N/Q/P, same analysis diagnose_knight_e4_anomaly.py did
by hand for one specific position, generalised to run across the whole
benchmark automatically.

Built to answer a specific question raised by that one earlier result:
is "policy correctly favours the right move, but search at both 200 and
600 sims converges on a worse one because its own Q ends up backwards"
a pattern, or was undefended_knight_on_e4 an isolated quirk? Doesn't
answer that by itself -- prints the evidence for every disagreement
found so it can be read across positions, same as that one was read by
hand.

Usage:
  venv/bin/python3 diagnose_search_disagreements.py [--sims 600]
"""

import argparse
import json

import chess
import chess.engine
import numpy as np
import torch

from chessai.agent import ChessAgent
from chessai.encoder import encode
from chessai.moves import move_to_index, index_to_move, get_mirror_indices_np
from run_config import CKPT_PATH

BENCHMARK_PATH = "chessai/tactical_benchmark.json"
ENGINE_PATH = "stockfish"
CONVERSION_DEPTH = 14
CONVERSION_STILL_WINNING_CP = 100

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=None)
parser.add_argument("--sims", type=int, default=600)
args = parser.parse_args()


def _replay_with_history(move_seq):
    board = chess.Board()
    history = []
    for uci in move_seq:
        history = ([board.copy()] + history)[:3]
        board.push_uci(uci)
    return board, history


def _policy_only_move(agent, board, history):
    state = encode([board] + history).unsqueeze(0)
    with torch.inference_mode():
        logits, _ = agent.network(state)
    logits = logits.squeeze(0).numpy()
    legal = list(board.legal_moves)
    idxs = np.fromiter((move_to_index(m) for m in legal), dtype=np.int64, count=len(legal))
    take = get_mirror_indices_np()[idxs] if board.turn == chess.BLACK else idxs
    best_i = int(np.argmax(logits[take]))
    return legal[best_i].uci()


def _root_rows(root, board):
    rows = []
    for idx, child in root.children.items():
        move = index_to_move(idx, board)
        rows.append((move.uci(), child.N, child.Q, child.P))
    rows.sort(key=lambda r: -r[1])
    return rows


def _print_comparison(rows, search_uci, correct_ucis):
    by_move = {r[0]: r for r in rows}
    print(f"  {'move':>8} {'N':>6} {'Q':>8} {'P':>8}")
    shown = set()
    for uci in [search_uci] + correct_ucis:
        if uci in shown or uci not in by_move:
            continue
        shown.add(uci)
        _, N, Q, P = by_move[uci]
        tag = []
        if uci == search_uci:
            tag.append("search chose this")
        if uci in correct_ucis:
            tag.append("correct")
        print(f"  {uci:>8} {N:>6} {Q:>+8.4f} {P:>8.4f}   ({', '.join(tag)})")

    search_row = by_move.get(search_uci)
    correct_rows = [by_move[u] for u in correct_ucis if u in by_move]
    if search_row and correct_rows:
        best_correct_q = max(r[2] for r in correct_rows)
        if search_row[2] > best_correct_q:
            print("  -> search's own Q is HIGHER than the correct move's -- "
                  "value-head miscalibration in that move's subtree, not "
                  "an exploration/prior problem.")
        else:
            print("  -> the correct move's own Q reads at least as good -- "
                  "if it still lost the visit contest, that points at "
                  "PUCT/exploration dynamics or an insufficient budget "
                  "instead.")


def _is_ok(category, board, chosen_uci, correct_moves, engine):
    """Same scoring rule run_tactical_benchmark.py uses, factored out so
    both the policy-only and search choices are checked identically."""
    if category != "conversion":
        return chosen_uci in correct_moves
    b2 = board.copy()
    b2.push_uci(chosen_uci)
    if b2.is_game_over():
        return b2.is_checkmate()
    info = engine.analyse(b2, chess.engine.Limit(depth=CONVERSION_DEPTH))
    score = info["score"].pov(not b2.turn)
    if score.is_mate():
        return score.mate() > 0
    cp = score.score()
    return cp is not None and cp >= CONVERSION_STILL_WINNING_CP


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

    with open(BENCHMARK_PATH) as f:
        positions = json.load(f)

    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    disagreements = 0
    try:
        for pos in positions:
            board, history = _replay_with_history(pos["moves"])
            policy_uci = _policy_only_move(agent, board, history)
            policy, root = agent.mcts.search(board, history, args.sims, add_noise=False)
            rows = _root_rows(root, board)
            search_uci = rows[0][0]   # most-visited = what greedy selection returns

            category = pos["category"]
            correct_moves = pos.get("correct_moves", [])   # empty for conversion
            policy_ok = _is_ok(category, board, policy_uci, correct_moves, engine)
            search_ok = _is_ok(category, board, search_uci, correct_moves, engine)

            print(f"{pos['name']} [{category}]: "
                  f"policy={'OK' if policy_ok else policy_uci}, "
                  f"search={'OK' if search_ok else search_uci}")

            # Only choice-based categories have a fixed correct_moves list
            # to compare the search's tree stats against -- conversion's
            # "correct answer" is "didn't blunder," not one specific move,
            # so there's nothing to diff its root stats against here.
            if not search_ok and correct_moves:
                disagreements += 1
                _print_comparison(rows, search_uci, correct_moves)
            print()
    finally:
        engine.quit()

    print(f"{disagreements} disagreement(s) found across {len(positions)} positions.")


if __name__ == "__main__":
    main()
