"""
logger.py — Rich logging for chess training.

Four data streams that answer questions Phase 2 logging couldn't:

  training.csv     — win rate, avg loss, game length histogram — every 100 games
  openings.csv     — first 12 moves of every game (detects repertoire emergence)
  snapshots.csv    — MCTS visit distributions at 5 canonical positions — every 500 games
                     (shows when a preference became confident, not just what it was)

The key signal in openings.csv: if the same 12-move sequence starts appearing
in an increasing share of games, HAL has found a line it believes in.
When it diversifies again, that line is being countered.

The key signal in snapshots.csv: a narrow visit distribution (80% on one move)
means confident exploitation. A flat one means still exploring.
"""

import os
import csv
import chess
import torch
from collections import defaultdict
from chessai.moves import move_to_index

SNAPSHOT_SIMS = 100   # simulations per canonical position — lightweight, for logging only

CANONICAL_POSITIONS = {
    "start":         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "after_e4":      "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "after_d4":      "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
    "italian_game":  "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4",
    "queens_gambit": "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 3",
}

LENGTH_BUCKETS = [(0, 20), (21, 40), (41, 60), (61, 80), (81, float("inf"))]


class Logger:

    def __init__(self, log_dir: str = "logs/chess",
                 perf_interval: int = 100,
                 snapshot_interval: int = 500):
        self.log_dir           = log_dir
        self.perf_interval     = perf_interval
        self.snapshot_interval = snapshot_interval

        os.makedirs(log_dir, exist_ok=True)

        self._perf_path     = os.path.join(log_dir, "training.csv")
        self._opening_path  = os.path.join(log_dir, "openings.csv")
        self._snapshot_path = os.path.join(log_dir, "snapshots.csv")

        self._end_reason_path = os.path.join(log_dir, "end_reasons.csv")

        self._init_csv(self._perf_path, [
            "game", "white_wins", "black_wins", "draws",
            "avg_loss", "avg_game_length",
            "len_0_20", "len_21_40", "len_41_60", "len_61_80", "len_81plus",
        ])
        self._init_csv(self._end_reason_path, [
            "game", "checkmates", "material_resigns", "value_resigns",
            "cap_draws", "rule_draws",
        ])
        self._init_csv(self._opening_path, ["game", "moves"])
        self._init_csv(self._snapshot_path, [
            "game", "position", "move1", "pct1", "move2", "pct2",
            "move3", "pct3", "move4", "pct4", "move5", "pct5",
        ])

        # Rolling accumulators for the current 100-game window
        self._reset_window()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def record_game(self, game_num: int, winner,
                    moves: list, loss: float,
                    end_reason: str = "unknown") -> None:
        """
        Call after every completed game.

        winner     : chess.WHITE, chess.BLACK, or None for draw
        moves      : list of UCI strings played in the game
        loss       : training loss for this game's batch (0.0 if no training step yet)
        end_reason : one of "checkmate", "material_resign", "value_resign",
                     "cap_draw", "rule_draw"
        """
        # Opening sequence — write immediately, useful at full resolution
        opening = " ".join(moves[:12]) if moves else ""
        self._append(self._opening_path, [game_num, opening])

        # Accumulate performance stats
        if winner == chess.WHITE:
            self._window["white_wins"] += 1
        elif winner == chess.BLACK:
            self._window["black_wins"] += 1
        else:
            self._window["draws"] += 1

        # Track game-end distribution
        key = end_reason.replace("-", "_")
        if key in self._window:
            self._window[key] += 1

        self._window["losses"].append(loss)
        game_length = len(moves)
        self._window["lengths"].append(game_length)

        # Flush every perf_interval games
        if game_num % self.perf_interval == 0 and game_num > 0:
            self._flush_performance(game_num)

    def record_snapshot(self, game_num: int, agent) -> None:
        """
        Call every snapshot_interval games.
        Runs lightweight MCTS on each canonical position and logs the
        top-5 move distribution — shows confidence, not just preference.
        """
        agent.network.eval()

        for name, fen in CANONICAL_POSITIONS.items():
            board = chess.Board(fen)
            _, policy = agent.choose_move(
                board, history=[],
                greedy=True,
                n_simulations=SNAPSHOT_SIMS,
            )

            # Top 5 moves by visit share
            top = policy.topk(5)
            row = [game_num, name]
            for prob, idx in zip(top.values.tolist(), top.indices.tolist()):
                from_sq = idx // 64
                to_sq   = idx % 64
                uci     = chess.Move(from_sq, to_sq).uci()
                row += [uci, f"{prob*100:.1f}%"]
            self._append(self._snapshot_path, row)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reset_window(self) -> None:
        self._window = {
            "white_wins": 0, "black_wins": 0, "draws": 0,
            "losses": [], "lengths": [],
            "checkmates": 0, "material_resigns": 0,
            "value_resigns": 0, "cap_draws": 0, "rule_draws": 0,
        }

    def _flush_performance(self, game_num: int) -> None:
        w = self._window
        n = len(w["losses"]) or 1
        avg_loss   = sum(w["losses"])   / n
        avg_length = sum(w["lengths"])  / n

        buckets = defaultdict(int)
        for length in w["lengths"]:
            for lo, hi in LENGTH_BUCKETS:
                if lo <= length <= hi:
                    buckets[f"{lo}_{int(hi) if hi != float('inf') else '81plus'}"] += 1
                    break

        self._append(self._perf_path, [
            game_num,
            w["white_wins"], w["black_wins"], w["draws"],
            f"{avg_loss:.6f}", f"{avg_length:.1f}",
            buckets.get("0_20", 0),
            buckets.get("21_40", 0),
            buckets.get("41_60", 0),
            buckets.get("61_80", 0),
            buckets.get("81_81plus", 0),
        ])
        self._append(self._end_reason_path, [
            game_num,
            w["checkmates"], w["material_resigns"], w["value_resigns"],
            w["cap_draws"], w["rule_draws"],
        ])
        self._reset_window()

    def _init_csv(self, path: str, headers: list) -> None:
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(headers)

    def _append(self, path: str, row: list) -> None:
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)
