"""
compare_batched_vs_serial_search.py — Step 2 of the search-misranking
investigation (paper/value_head_small_material_options.md, §6): does the
in-tree batching (BATCH_SIMS leaves expanded per network call, held
together with virtual loss) change which move search settles on, versus
fully serial search (one leaf, one network call, one backup at a time)?

Same network weights, same simulation budget, same positions -- the only
thing that differs between the two MCTS instances built here is
batch_sims. If the wrong-move-wins outcome is identical under both, that
rules batching out as the explanation for these specific failures (not
for MCTS in general). If serial search gets it right where batched search
doesn't, batching -- specifically the virtual-loss approximation it
relies on -- is implicated and worth investigating further on its own.

This does not by itself distinguish "value head is wrong" from "search
exploration/prior effects are wrong" -- see inspect_two_search_failures.py
(step 1) and the Stockfish-on-sampled-leaves check (step 3) for that.

Usage:
  venv/bin/python3 compare_batched_vs_serial_search.py [--sims 600]
"""

import argparse
import json

import chess
import torch

from chessai.agent import ChessAgent
from chessai.mcts import MCTS, BATCH_SIMS
from chessai.moves import index_to_move
from run_config import CKPT_PATH

BENCHMARK_PATH = "chessai/tactical_benchmark.json"

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


def _search_choice(mcts, board, history, sims):
    policy, root = mcts.search(board, history, sims, add_noise=False)
    best_idx = max(root.children, key=lambda i: root.children[i].N)
    child = root.children[best_idx]
    uci = index_to_move(best_idx, board).uci()
    return uci, child.N, child.Q


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

    mcts_batched = MCTS(agent.network, device, batch_sims=BATCH_SIMS)
    mcts_serial  = MCTS(agent.network, device, batch_sims=1)

    with open(BENCHMARK_PATH) as f:
        positions = [p for p in json.load(f) if p["category"] == "hanging_piece"]

    print(f"{'position':>44} {'correct':>8}  {'batched('+str(BATCH_SIMS)+')':>16}  {'serial(1)':>16}")
    print("-" * 100)
    agreements = 0
    for pos in positions:
        board, history = _replay_with_history(pos["moves"])
        correct = pos["correct_moves"][0]

        b_uci, b_N, b_Q = _search_choice(mcts_batched, board, history, args.sims)
        s_uci, s_N, s_Q = _search_choice(mcts_serial, board, history, args.sims)

        b_mark = "OK" if b_uci == correct else b_uci
        s_mark = "OK" if s_uci == correct else s_uci
        print(f"{pos['name']:>44} {correct:>8}  {b_mark+f' (Q={b_Q:+.3f})':>16}  "
              f"{s_mark+f' (Q={s_Q:+.3f})':>16}")

        if (b_uci == correct) == (s_uci == correct):
            agreements += 1

    print(f"\nBatched and serial search agree on pass/fail for "
          f"{agreements}/{len(positions)} positions.")
    print("(A mismatch here would mean batching itself changes the outcome "
          "for that position -- worth a closer look with step 1's tooling.)")


if __name__ == "__main__":
    main()
