"""
run_tactical_benchmark.py — Score a checkpoint against
chessai/tactical_benchmark.json, separately at the raw policy prior and
at a fixed search budget.

Why two scores, not one (10 Sept 2026 assessment, forward plan item 2):
"Evaluate policy-only selection and search at equal fixed budgets. This
directly tests the weaknesses visible in actual games" -- a position the
raw prior gets wrong but a short search finds anyway is a search-budget
question, not a policy-training one, and conflating the two hides which
lever would actually help.

Each category is scored differently, matching how generate_tactical_
benchmark.py validated it:
  hanging_piece, mate_in_1, defend_mate_in_1 — chosen move must be in
      the position's pre-validated correct_moves list
  conversion — chosen move must not blunder the win: after playing it,
      Stockfish must still rate the position as winning for the side
      that was ahead (checked live here, not pre-computed, since the
      "correct" answer depends on what HAL actually played)

Usage:
  venv/bin/python3 run_tactical_benchmark.py [--ckpt PATH] [--sims N]
"""

import argparse
import json

import chess
import chess.engine
import numpy as np
import torch

from chessai.agent import ChessAgent
from chessai.encoder import encode
from chessai.moves import move_to_index, get_mirror_indices_np
from run_config import CKPT_PATH

BENCHMARK_PATH = "chessai/tactical_benchmark.json"
ENGINE_PATH    = "stockfish"   # assumes stockfish is on PATH -- same convention as eval_chess.py
CONVERSION_DEPTH = 14
CONVERSION_STILL_WINNING_CP = 100   # generous margin -- this checks "didn't
                                    # throw the win away", not "played the
                                    # engine's top choice"

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=None, help="Checkpoint to evaluate")
parser.add_argument("--sims", type=int, default=200, help="Search budget for the search-based score")
args = parser.parse_args()


def _replay_with_history(move_seq: list):
    """Returns (board, history) exactly like the encoder expects --
    history is the last 3 boards, most recent first."""
    board = chess.Board()
    history = []
    for uci in move_seq:
        history = ([board.copy()] + history)[:3]
        board.push_uci(uci)
    return board, history


def policy_only_move(agent: ChessAgent, board: chess.Board, history: list) -> str:
    """The raw network prior's argmax over legal moves, no search at all --
    mirrors chessai/mcts.py's _expand_from_logits masking/softmax exactly,
    including the promotion-collision fix (dedupe indices before softmax)."""
    state = encode([board] + history).unsqueeze(0)
    with torch.inference_mode():
        logits, _ = agent.network(state)
    logits = logits.squeeze(0).numpy()

    legal = list(board.legal_moves)
    idxs = np.fromiter((move_to_index(m) for m in legal), dtype=np.int64, count=len(legal))
    take = get_mirror_indices_np()[idxs] if board.turn == chess.BLACK else idxs

    best_i = int(np.argmax(logits[take]))
    return legal[best_i].uci()


def search_move(agent: ChessAgent, board: chess.Board, history: list, n_sims: int) -> str:
    move_uci, _, _ = agent.choose_move(board, history, greedy=True, n_simulations=n_sims)
    return move_uci


def score_choice_move(position: dict, chosen_uci: str) -> bool:
    return chosen_uci in position["correct_moves"]


def score_conversion(position: dict, board: chess.Board, chosen_uci: str, engine) -> bool:
    board = board.copy()
    board.push_uci(chosen_uci)
    if board.is_game_over():
        # delivering mate or stalemating -- mate is obviously fine,
        # stalemate is the one real way to "blunder" a totally won
        # position into a draw, so it must fail the check
        return board.is_checkmate()
    info = engine.analyse(board, chess.engine.Limit(depth=CONVERSION_DEPTH))
    score = info["score"].pov(not board.turn)   # from the mover's own POV
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
        print(f"Loaded checkpoint: {ckpt_path} ({agent.steps:,} steps)\n")
    except FileNotFoundError:
        print(f"No checkpoint at {ckpt_path} -- scoring a fresh, untrained "
              f"network (expect close to chance).\n")

    with open(BENCHMARK_PATH) as f:
        positions = json.load(f)

    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    results = []
    try:
        for pos in positions:
            board, history = _replay_with_history(pos["moves"])
            policy_uci = policy_only_move(agent, board, history)
            search_uci = search_move(agent, board, history, args.sims)

            if pos["category"] == "conversion":
                policy_ok = score_conversion(pos, board, policy_uci, engine)
                search_ok = score_conversion(pos, board, search_uci, engine)
            else:
                policy_ok = score_choice_move(pos, policy_uci)
                search_ok = score_choice_move(pos, search_uci)

            results.append({
                "name": pos["name"], "category": pos["category"],
                "policy_move": policy_uci, "policy_ok": policy_ok,
                "search_move": search_uci, "search_ok": search_ok,
            })
    finally:
        engine.quit()

    print(f"{'name':>28} {'category':>18}   {'policy':>8}  {'search(' + str(args.sims) + ')':>12}")
    print("-" * 76)
    for r in results:
        p = "PASS" if r["policy_ok"] else f"fail ({r['policy_move']})"
        s = "PASS" if r["search_ok"] else f"fail ({r['search_move']})"
        print(f"{r['name']:>28} {r['category']:>18}   {p:>8}  {s:>12}")

    n = len(results)
    policy_pass = sum(r["policy_ok"] for r in results)
    search_pass = sum(r["search_ok"] for r in results)
    print(f"\nPolicy-only: {policy_pass}/{n}   Search({args.sims} sims): {search_pass}/{n}")


if __name__ == "__main__":
    main()
