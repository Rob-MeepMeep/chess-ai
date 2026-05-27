"""
train_ttt.py — Train HAL to play Tic-Tac-Toe via self-play.

Two agents (HAL-X and HAL-O) play each other for NUM_EPISODES games.
Both start knowing nothing and learn purely from wins, losses, and draws.

Run from the chess-ai directory:
    venv/bin/python train_ttt.py

Outputs:
    logs/training_log.csv     — win/draw rates every 1,000 games
    logs/learning_curve.png   — chart of HAL's progress over time
    checkpoints/hal_x.pkl     — HAL-X's learned Q-table
    checkpoints/hal_o.pkl     — HAL-O's learned Q-table
"""

import os
import csv
import time
import matplotlib.pyplot as plt

from tictactoe.env import TicTacToeEnv
from tictactoe.agent import QLearningAgent

NUM_EPISODES  = 200_000
LOG_INTERVAL  = 1_000   # print stats and write to CSV every N games
DEMO_INTERVAL = 5_000   # play a visible demo game every N episodes


def demo_game(hal_x, hal_o, episode):
    """
    Play one fully visible game between HAL-X and HAL-O.
    HAL plays greedily (epsilon=0) so we see what he's learned, not random moves.
    Called every DEMO_INTERVAL episodes so you can watch play improve over time.
    """
    env = TicTacToeEnv()
    state = env.reset()
    done = False
    move_num = 0

    print(f"\n{'='*52}")
    print(f"  DEMO GAME — Episode {episode:,}")
    print(f"  ε={hal_x.epsilon:.4f}  |  States known: {len(hal_x.q_table):,}")
    print(f"{'='*52}")

    # Show HAL-X's opening opinion before any moves are made
    hal_x.explain(state)
    time.sleep(0.8)

    while not done:
        current_player = env.current_player
        agent = hal_x if current_player == 1 else hal_o
        player_name = 'HAL-X' if current_player == 1 else 'HAL-O'

        legal = env.legal_actions()

        # Temporarily disable exploration so the demo shows HAL's best play
        saved_epsilon = agent.epsilon
        agent.epsilon = 0.0
        action = agent.choose_action(state, legal)
        agent.epsilon = saved_epsilon

        move_num += 1
        print(f"Move {move_num}: {player_name} plays cell {action}")

        state, _, done, winner = env.step(action)
        env.render()
        time.sleep(0.6)

    outcomes = {1: "HAL-X wins!", 2: "HAL-O wins!", 0: "Draw!"}
    print(f"Result: {outcomes[winner]}")
    print(f"{'='*52}\n")
    time.sleep(1.0)


def train():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    env = TicTacToeEnv()
    hal_x = QLearningAgent('HAL-X', epsilon_min=0.02)
    hal_o = QLearningAgent('HAL-O', epsilon_min=0.02)

    results = []   # 'x', 'o', or 'd' for each episode
    log_rows = []  # one row per LOG_INTERVAL for the CSV

    print(f"Training HAL for {NUM_EPISODES:,} episodes...\n")
    print(f"{'Episode':>10} | {'X wins':>8} | {'O wins':>8} | {'Draws':>8} | {'Epsilon':>8} | {'States':>10}")
    print("-" * 68)

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        done = False
        winner = 0

        # Track the last (state, action) each player made so we can
        # issue the -1 penalty to the loser when the game ends.
        last_move = {1: None, 2: None}

        while not done:
            current_player = env.current_player
            agent = hal_x if current_player == 1 else hal_o

            legal = env.legal_actions()
            action = agent.choose_action(state, legal)
            next_state, reward, done, winner = env.step(action)
            next_legal = env.legal_actions() if not done else []

            # Update the agent that just moved
            agent.update(state, action, reward, next_state, next_legal, done)
            last_move[current_player] = (state, action)

            # If someone just won, go back and penalise the loser's last move
            if done and winner != 0:
                loser_player = 3 - current_player
                loser_agent = hal_o if current_player == 1 else hal_x
                if last_move[loser_player] is not None:
                    ls, la = last_move[loser_player]
                    loser_agent.update(ls, la, -1, next_state, [], True)

            state = next_state

        # Record the outcome
        if winner == 1:
            results.append('x')
        elif winner == 2:
            results.append('o')
        else:
            results.append('d')

        # Both HALs explore a little less next game
        hal_x.decay_epsilon()
        hal_o.decay_epsilon()

        # Every DEMO_INTERVAL games: play one visible game so you can watch HAL improve
        if episode % DEMO_INTERVAL == 0:
            demo_game(hal_x, hal_o, episode)

        # Every LOG_INTERVAL games: print progress and write to CSV
        if episode % LOG_INTERVAL == 0:
            recent = results[-LOG_INTERVAL:]
            x_wins = recent.count('x')
            o_wins = recent.count('o')
            draws  = recent.count('d')

            print(
                f"{episode:>10,} | "
                f"{x_wins:>5} {x_wins/LOG_INTERVAL*100:>4.1f}% | "
                f"{o_wins:>5} {o_wins/LOG_INTERVAL*100:>4.1f}% | "
                f"{draws:>5} {draws/LOG_INTERVAL*100:>4.1f}% | "
                f"{hal_x.epsilon:>8.4f} | "
                f"{len(hal_x.q_table):>10,}"
            )

            log_rows.append({
                'episode':     episode,
                'x_wins':      x_wins,
                'o_wins':      o_wins,
                'draws':       draws,
                'x_win_rate':  round(x_wins / LOG_INTERVAL, 4),
                'o_win_rate':  round(o_wins / LOG_INTERVAL, 4),
                'draw_rate':   round(draws  / LOG_INTERVAL, 4),
                'epsilon':     round(hal_x.epsilon, 6),
                'hal_x_states': len(hal_x.q_table),
                'hal_o_states': len(hal_o.q_table),
            })

    # Save Q-tables to disk
    print()
    hal_x.save('checkpoints/hal_x.pkl')
    hal_o.save('checkpoints/hal_o.pkl')

    # Write CSV log
    csv_path = 'logs/training_log.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Training log saved to {csv_path}")

    # Generate and save the learning curve chart
    _plot_learning_curve(log_rows)

    return hal_x, hal_o


def _plot_learning_curve(log_rows):
    """Save a chart showing win/draw rates across all episodes."""
    episodes   = [r['episode']    for r in log_rows]
    x_rates    = [r['x_win_rate'] * 100 for r in log_rows]
    o_rates    = [r['o_win_rate'] * 100 for r in log_rows]
    draw_rates = [r['draw_rate']  * 100 for r in log_rows]

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, x_rates,    label='X wins (%)', color='royalblue',  linewidth=1.5)
    plt.plot(episodes, o_rates,    label='O wins (%)', color='tomato',     linewidth=1.5)
    plt.plot(episodes, draw_rates, label='Draws (%)',  color='seagreen',   linewidth=1.5)

    plt.xlabel('Episode')
    plt.ylabel(f'Rate over last {LOG_INTERVAL:,} games (%)')
    plt.title("HAL's Learning Curve — Tic-Tac-Toe Self-Play")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    chart_path = 'logs/learning_curve.png'
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Learning curve saved to {chart_path}")


if __name__ == '__main__':
    hal_x, hal_o = train()

    # After training, show HAL-X's opinion of the empty board
    print("\nHAL-X's view of the starting position after training:")
    env = TicTacToeEnv()
    hal_x.explain(env.reset())

    print("HAL-O's view of the starting position after training:")
    hal_o.explain(env.reset())
