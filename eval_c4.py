"""
eval_c4.py — Evaluate HAL-3000's Connect Four performance.

Runs four matchups to give a clear picture of what the trained network
actually learned:

    1. HAL-1 vs Random   — HAL playing first, opponent plays randomly
    2. Random vs HAL-2   — random plays first, HAL playing second
    3. HAL-1 vs HAL-2    — pure exploitation self-play (ε=0 both sides)
    4. Random vs Random  — baseline: what does chance look like?

Matchups 1 and 2 are the most meaningful — they show absolute performance
against a known-bad opponent. If HAL is winning 80%+ from both sides, it
has learned genuine strategy, not just how to beat its training partner.
"""

import random
import torch

from connect4.env import ConnectFourEnv
from connect4.agent import DQNAgent

N_GAMES     = 1_000
CKPT_HAL1   = "checkpoints/hal1_c4.pt"
CKPT_HAL2   = "checkpoints/hal2_c4.pt"

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ---------------------------------------------------------------------------
# Load trained agents
# ---------------------------------------------------------------------------

hal1 = DQNAgent(player=1, device=device)
hal1.load(CKPT_HAL1)
hal1.epsilon = 0.0  # pure exploitation — no random moves

hal2 = DQNAgent(player=2, device=device)
hal2.load(CKPT_HAL2)
hal2.epsilon = 0.0

print(f"Loaded HAL-1 (trained steps: {hal1.steps:,})")
print(f"Loaded HAL-2 (trained steps: {hal2.steps:,})\n")

# ---------------------------------------------------------------------------
# Helper: play one game between two move functions
# ---------------------------------------------------------------------------

def play_game(env, move_fn_1, move_fn_2):
    """
    Play one complete game.

    move_fn_1: function(state, legal_actions) -> action for player 1
    move_fn_2: function(state, legal_actions) -> action for player 2

    Returns: winner (1, 2, or 0 for draw)
    """
    state = env.reset()
    done  = False

    while not done:
        legal = env.legal_actions()
        action = move_fn_1(state, legal)
        state, _, done, winner = env.step(action)
        if done:
            return winner

        legal = env.legal_actions()
        action = move_fn_2(state, legal)
        state, _, done, winner = env.step(action)
        if done:
            return winner

    return 0

# ---------------------------------------------------------------------------
# Move functions
# ---------------------------------------------------------------------------

def random_move(state, legal_actions):
    return random.choice(legal_actions)

def hal1_move(state, legal_actions):
    return hal1.choose_action(state, legal_actions)

def hal2_move(state, legal_actions):
    return hal2.choose_action(state, legal_actions)

# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate(label, move_fn_1, move_fn_2, n=N_GAMES):
    env = ConnectFourEnv()
    p1_wins = 0
    p2_wins = 0
    draws   = 0

    for _ in range(n):
        winner = play_game(env, move_fn_1, move_fn_2)
        if winner == 1:
            p1_wins += 1
        elif winner == 2:
            p2_wins += 1
        else:
            draws += 1

    print(f"{label}")
    print(f"  Player 1 wins: {p1_wins:>4} ({p1_wins/n*100:5.1f}%)")
    print(f"  Player 2 wins: {p2_wins:>4} ({p2_wins/n*100:5.1f}%)")
    print(f"  Draws:         {draws:>4} ({draws/n*100:5.1f}%)")
    print()

    return p1_wins, p2_wins, draws

# ---------------------------------------------------------------------------
# Run all four matchups
# ---------------------------------------------------------------------------

print(f"Evaluating over {N_GAMES:,} games per matchup...\n")
print("=" * 50)
print()

evaluate("1. HAL-1 (P1) vs Random (P2)", hal1_move, random_move)
evaluate("2. Random (P1) vs HAL-2 (P2)", random_move, hal2_move)
evaluate("3. HAL-1 (P1) vs HAL-2 (P2) — ε=0 self-play", hal1_move, hal2_move)
evaluate("4. Random (P1) vs Random (P2) — baseline", random_move, random_move)

print("=" * 50)
print("Done.")
