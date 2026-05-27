"""
eval_ttt.py — Evaluate HAL's true performance with no randomness.

Loads the trained checkpoints and plays 1,000 games with epsilon=0 (pure
exploitation). This removes the noise from exploration and shows what HAL
actually learned.

Run from the chess-ai directory:
    venv/bin/python eval_ttt.py
"""

from tictactoe.env import TicTacToeEnv
from tictactoe.agent import QLearningAgent

EVAL_EPISODES = 1_000


def evaluate():
    env = TicTacToeEnv()
    hal_x = QLearningAgent('HAL-X')
    hal_o = QLearningAgent('HAL-O')

    hal_x.load('checkpoints/hal_x.pkl')
    hal_o.load('checkpoints/hal_o.pkl')

    # Pure exploitation — no random moves
    hal_x.epsilon = 0.0
    hal_o.epsilon = 0.0

    x_wins = o_wins = draws = 0

    for _ in range(EVAL_EPISODES):
        state = env.reset()
        done = False
        winner = 0

        while not done:
            agent = hal_x if env.current_player == 1 else hal_o
            action = agent.choose_action(state, env.legal_actions())
            state, _, done, winner = env.step(action)

        if winner == 1:
            x_wins += 1
        elif winner == 2:
            o_wins += 1
        else:
            draws += 1

    print(f"\nEvaluation over {EVAL_EPISODES:,} games (epsilon = 0 — pure exploitation)\n")
    print(f"  X wins : {x_wins:>5}  ({x_wins/EVAL_EPISODES*100:.1f}%)")
    print(f"  O wins : {o_wins:>5}  ({o_wins/EVAL_EPISODES*100:.1f}%)")
    print(f"  Draws  : {draws:>5}  ({draws/EVAL_EPISODES*100:.1f}%)")
    print()

    if draws == EVAL_EPISODES:
        print("HAL plays perfectly — every game is a draw.")
    elif draws / EVAL_EPISODES >= 0.95:
        print("HAL plays near-perfectly — draws in 95%+ of games.")
    elif draws / EVAL_EPISODES >= 0.80:
        print("HAL plays well — still some suboptimal lines to resolve.")
    else:
        print("HAL has learned but hasn't fully converged to optimal play.")


if __name__ == '__main__':
    evaluate()
