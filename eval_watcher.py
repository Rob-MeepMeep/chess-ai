#!/usr/bin/env python3
"""
eval_watcher.py — Auto-trigger eval_chess.py every EVAL_INTERVAL games.

Run in a second terminal alongside train_chess.py:
  venv/bin/python3 eval_watcher.py

Watches logs/run11/games.csv and fires a full eval whenever a new
EVAL_INTERVAL boundary is crossed. Eval output prints to this terminal.
Stops cleanly with Ctrl+C.
"""

import csv
import os
import subprocess
import sys
import time

RUN_NAME      = "run11"
EVAL_INTERVAL = 1500
GAMES_CSV     = f"logs/{RUN_NAME}/games.csv"
POLL_SECONDS  = 30


def last_game_num(path):
    """Return the highest valid game number in games.csv, or 0 if not yet written."""
    if not os.path.exists(path):
        return 0
    try:
        with open(path) as f:
            rows = list(csv.reader(f))
        for row in reversed(rows[1:]):   # skip header, scan from end for first valid row
            try:
                return int(row[0])
            except (ValueError, IndexError):
                continue
    except Exception:
        pass
    return 0


def main():
    # Initialise last_eval_at to the last completed interval already in the log,
    # so starting mid-run doesn't immediately trigger a catch-up eval.
    initial_game = last_game_num(GAMES_CSV)
    last_eval_at = (initial_game // EVAL_INTERVAL) * EVAL_INTERVAL

    next_eval = last_eval_at + EVAL_INTERVAL
    print(f"Eval watcher started — watching {GAMES_CSV}")
    print(f"Eval interval: every {EVAL_INTERVAL} games | poll: every {POLL_SECONDS}s")
    print(f"Current game: {initial_game} | next eval at game {next_eval}\n")

    try:
        while True:
            game_num = last_game_num(GAMES_CSV)

            if game_num >= next_eval:
                print(f"\n[{time.strftime('%H:%M:%S')}] Game {game_num} — running eval (trigger: {next_eval})...")
                print("─" * 60)
                subprocess.run([sys.executable, "eval_chess.py"], check=False)
                print("─" * 60)
                last_eval_at = next_eval
                next_eval    = last_eval_at + EVAL_INTERVAL
                print(f"[{time.strftime('%H:%M:%S')}] Eval complete. Next at game {next_eval}.\n")
            else:
                print(
                    f"  [{time.strftime('%H:%M:%S')}] game {game_num} / {next_eval}  "
                    f"({next_eval - game_num} games until eval)",
                    end="\r",
                )

            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        print(f"\n\nWatcher stopped at game {last_game_num(GAMES_CSV)}.")


if __name__ == "__main__":
    main()
