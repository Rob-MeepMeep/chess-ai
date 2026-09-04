"""
logger.py — Rich logging for chess training.

Four data streams that answer questions Phase 2 logging couldn't:

  games.csv        — one row per game: outcome, end reason, move list — every game
  training.csv     — win rate, avg loss, game length histogram — every perf_interval games
  openings.csv     — first 12 moves of every game (detects repertoire emergence)
  snapshots.csv    — MCTS visit distributions at 5 canonical positions — every snapshot_interval games
                     (shows when a preference became confident, not just what it was)

The key signal in openings.csv: if the same 12-move sequence starts appearing
in an increasing share of games, HAL has found a line it believes in.
When it diversifies again, that line is being countered.

The key signal in snapshots.csv: a narrow visit distribution (80% on one move)
means confident exploitation. A flat one means still exploring.
"""

import os
import csv
import json
import time
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

# Diverse material-imbalance probe (run16+, generalisation-gap Option A —
# see paper/generalization_gap_options.md). Originally seven single FENs:
# the starting position with one piece deleted, zero moves played, empty
# history. That construction cannot occur naturally — removing a piece
# requires a capture, which requires history — and the run16-19 arc
# (rung 1b, run17's Stockfish relabelling, run18/run19's alpha tuning) was
# chasing what turned out to be a probe-construction artifact, not a real
# training gap: diagnose_network_output.py and diagnose_probe_construction.py
# (3 Sept, run19 game ~1190) showed the checkpoint tracks Stockfish's
# material judgement well (corr 0.90-0.96) on real in-context positions at
# every magnitude, including bishop/knight/two-pawns — it just couldn't
# read the synthetic zero-history FENs, which it had never seen paired
# with a material deficit in any form.
#
# Fixed by replacing each single FEN with 30 real move-sequences per
# category, sampled once from run19's own games.csv (see
# generate_material_probe_positions.py) — positions where exactly one
# piece type differs by the category's amount and every other piece count
# matches, "White to move" enforced the same way the original FENs were.
# record_material_probe() replays each sequence to reconstruct real board
# history (matching relabel_with_stockfish.py's convention exactly) and
# averages the 30 predictions, instead of reading one FEN. Averaging also
# damps the higher variance real positions carry (std ~0.6-0.9, vs a
# single point estimate) into a stable per-checkpoint reading.
#
# NOTE for anyone reading material_probe.csv across this change: rows
# before this fix measured the old synthetic construction; rows after
# measure real in-context positions. The two are not directly comparable
# — expect a visible discontinuity at the game number this landed on.
with open(os.path.join(os.path.dirname(__file__),
                        "material_probe_positions.json")) as _f:
    MATERIAL_PROBE_POSITIONS = json.load(_f)


class Logger:

    def __init__(self, log_dir: str = "logs/chess",
                 perf_interval: int = 50,
                 snapshot_interval: int = 500):
        self.log_dir           = log_dir
        self.perf_interval     = perf_interval
        self.snapshot_interval = snapshot_interval

        os.makedirs(log_dir, exist_ok=True)

        self._perf_path     = os.path.join(log_dir, "training.csv")
        self._opening_path  = os.path.join(log_dir, "openings.csv")
        self._snapshot_path = os.path.join(log_dir, "snapshots.csv")
        self._games_path    = os.path.join(log_dir, "games.csv")

        self._end_reason_path = os.path.join(log_dir, "end_reasons.csv")

        # NOTE: steps and timestamp columns were added mid-Run 8 (game ~1000).
        # Rows written before that point do not have these columns — the header
        # in existing run8 CSVs will not match. Acceptable for a training log;
        # use the column name rather than position when reading in pandas.
        self._init_csv(self._perf_path, [
            "game", "white_wins", "black_wins", "draws",
            "avg_loss", "avg_game_length",
            "len_0_20", "len_21_40", "len_41_60", "len_61_80", "len_81plus",
            "steps", "timestamp",
        ])
        self._init_csv(self._end_reason_path, [
            "game", "checkmates", "material_resigns", "value_resigns",
            "cap_draws", "rule_draws",
        ])
        self._init_csv(self._games_path, [
            "game", "outcome", "end_reason", "n_moves", "loss", "steps", "timestamp", "moves",
        ])
        self._init_csv(self._opening_path, ["game", "moves"])
        self._init_csv(self._snapshot_path, [
            "game", "position", "move1", "pct1", "move2", "pct2",
            "move3", "pct3", "move4", "pct4", "move5", "pct5",
        ])

        self._regression_path = os.path.join(log_dir, "regression.csv")
        self._init_csv(self._regression_path, [
            "game", "start", "w_wins", "b_move", "missing_queen", "timestamp",
        ])

        self._material_probe_path = os.path.join(log_dir, "material_probe.csv")
        self._init_csv(self._material_probe_path,
                        ["game"] + list(MATERIAL_PROBE_POSITIONS.keys()) + ["timestamp"])

        # avg_loss (training.csv) is policy_loss + value_loss summed before
        # logging (agent.py's train()), which can't distinguish "value head
        # not converging" from "policy head getting more confident/lower-
        # entropy as play improves" -- both look like a climbing number.
        # Split out separately here so a future climb is diagnosable. New
        # file rather than a new training.csv column: _init_csv only writes
        # a header if the file doesn't exist yet, so an in-progress run's
        # existing training.csv keeps its old header on resume, same issue
        # as material_probe.csv's fix (see material_probe_correction.md).
        self._loss_breakdown_path = os.path.join(log_dir, "loss_breakdown.csv")
        self._init_csv(self._loss_breakdown_path, [
            "game", "avg_policy_loss", "avg_value_loss", "steps", "timestamp",
        ])

        # Rolling accumulators for the current 100-game window
        self._reset_window()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def record_game(self, game_num: int, winner,
                    moves: list, loss: float,
                    end_reason: str = "unknown",
                    steps: int = 0,
                    policy_loss: float = 0.0, value_loss: float = 0.0) -> None:
        """
        Call after every completed game.

        winner      : chess.WHITE, chess.BLACK, or None for draw
        moves       : list of UCI strings played in the game
        loss        : training loss for this game's batch (0.0 if no training step yet)
        end_reason  : one of "checkmate", "material_resign", "value_resign",
                      "cap_draw", "rule_draw"
        steps       : agent.steps at game completion — enables elapsed-time accounting
        policy_loss, value_loss : loss's two components (0.0 if no training
                      step yet) — accumulated into loss_breakdown.csv,
                      separately from games.csv/training.csv's combined loss
        """
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        outcome_str = "W" if winner == chess.WHITE else "B" if winner == chess.BLACK else "D"
        self._append(self._games_path, [
            game_num, outcome_str, end_reason,
            len(moves), f"{loss:.6f}", steps, ts, " ".join(moves),
        ])
        # Keep the latest step count so _flush_performance can record it
        self._window["steps"] = steps

        # Opening sequence — first 12 moves for pattern analysis
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
        self._window["policy_losses"].append(policy_loss)
        self._window["value_losses"].append(value_loss)
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
            _, policy, _ = agent.choose_move(
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

    def record_regression(self, game_num: int, agent) -> None:
        """
        Run value head regression and append one row to regression.csv.
        Called every 200 games — gives a continuous curve instead of manual
        spot-checks between evals.

        Four positions match eval_chess.py's REGRESSION_POSITIONS exactly so
        the logged values are directly comparable to manual eval output.
        """
        # K+Q FENs replaced 2026-07-09: the old w_wins position was illegal
        # (black in check, white to move) and the old b_move position was
        # already checkmate — a terminal state the value head never trains on.
        # regression.csv values before this date are not comparable.
        # Keep in sync with eval_chess.REGRESSION_POSITIONS.
        fens = {
            "start":         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "w_wins":        "k7/8/8/8/4K3/8/3Q4/8 w - - 0 1",
            "b_move":        "k7/8/8/8/4K3/8/3Q4/8 b - - 0 1",
            "missing_queen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
        }
        agent.network.eval()
        values = {}
        for key, fen in fens.items():
            board = chess.Board(fen)
            values[key] = agent.get_value(board, [])

        self._append(self._regression_path, [
            game_num,
            f"{values['start']:+.4f}",
            f"{values['w_wins']:+.4f}",
            f"{values['b_move']:+.4f}",
            f"{values['missing_queen']:+.4f}",
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ])

    def record_material_probe(self, game_num: int, agent) -> None:
        """
        Diagnostic for generalisation-gap Option A (run16+): does the value
        head respond sensibly to material imbalance across DIFFERENT missing
        pieces and BOTH sides, not just the single missing_queen FEN
        regression.csv tracks? A network that has genuinely learned
        "material matters" as a concept — rather than having memorised one
        FEN's answer — should read negative for every white-down category
        and positive for every black-down one, roughly scaled by how much is
        missing (queen > rook > bishop/knight > two pawns). Several
        independent positions moving together is a much stronger signal
        than one number moving.

        Each category is 30 real move-sequences (see the comment above
        MATERIAL_PROBE_POSITIONS) rather than one synthetic FEN — replay
        each to reconstruct real board history, then average the 30
        predictions per category.

        Called at the same cadence as record_regression — cheap (plain
        forward passes, no MCTS), so no reason to wait for eval_watcher's
        much slower 1,500-game cycle.
        """
        agent.network.eval()
        values = {}
        for key, sequences in MATERIAL_PROBE_POSITIONS.items():
            preds = []
            for seq in sequences:
                board = chess.Board()
                history = []
                for move_uci in seq:
                    # prepend BEFORE pushing — matches
                    # relabel_with_stockfish.py's history convention exactly
                    history = ([board.copy()] + history)[:3]
                    board.push_uci(move_uci)
                preds.append(agent.get_value(board, history))
            values[key] = sum(preds) / len(preds)

        row = ([game_num]
               + [f"{values[k]:+.4f}" for k in MATERIAL_PROBE_POSITIONS]
               + [time.strftime("%Y-%m-%d %H:%M:%S")])
        self._append(self._material_probe_path, row)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reset_window(self) -> None:
        self._window = {
            "white_wins": 0, "black_wins": 0, "draws": 0,
            "losses": [], "lengths": [],
            "policy_losses": [], "value_losses": [],
            # Keys match end_reason strings from train_chess.py exactly
            "checkmate": 0, "material_resign": 0,
            "value_resign": 0, "cap_draw": 0, "rule_draw": 0,
            "steps": 0,
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
            w["steps"],
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ])
        self._append(self._end_reason_path, [
            game_num,
            w["checkmate"], w["material_resign"], w["value_resign"],
            w["cap_draw"], w["rule_draw"],
        ])

        avg_policy_loss = sum(w["policy_losses"]) / n
        avg_value_loss  = sum(w["value_losses"])  / n
        self._append(self._loss_breakdown_path, [
            game_num,
            f"{avg_policy_loss:.6f}", f"{avg_value_loss:.6f}",
            w["steps"], time.strftime("%Y-%m-%d %H:%M:%S"),
        ])
        self._reset_window()

    def _init_csv(self, path: str, headers: list) -> None:
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(headers)

    def _append(self, path: str, row: list) -> None:
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)
