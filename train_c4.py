"""
train_c4.py — Training loop for HAL-3000 (Connect Four DQN).

Two agents play each other in self-play:
    HAL-1 plays as player 1 (X)
    HAL-2 plays as player 2 (O)

Transition storage:
    HAL-1's transitions are pushed AFTER HAL-2 responds, so the stored
    next_state reflects the board HAL-1 will actually see on its next turn
    (after both players have moved), not an intermediate half-step state.

Rewards:
    +1  issued by the environment to the winner
    -1  issued by this training loop to the loser
     0  draw — no penalty, no reward

Storage safety:
    Training stops gracefully if free disk space drops below 10GB.
    Checkpoints are saved before stopping.
"""

import os
import csv
import time
import shutil
import torch

from connect4.env import ConnectFourEnv
from connect4.replay import ReplayBuffer
from connect4.agent import DQNAgent
from connect4.visualizer import Visualizer

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

NUM_EPISODES       = 50_000
BATCH_SIZE         = 64
REPLAY_CAPACITY    = 50_000
LOG_INTERVAL       = 500    # print stats and write CSV every N episodes
SAVE_INTERVAL      = 5_000  # save checkpoints every N episodes
DEMO_INTERVAL      = 5_000  # show a visual demo game every N episodes
MIN_FREE_GB        = 10     # stop training if disk drops below this

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

os.makedirs("logs",              exist_ok=True)
os.makedirs("logs/screenshots",  exist_ok=True)
os.makedirs("checkpoints",       exist_ok=True)

LOG_PATH   = "logs/training_c4.csv"
CHART_PATH = "logs/learning_curve_c4.png"
CKPT_HAL1  = "checkpoints/hal1_c4.pt"
CKPT_HAL2  = "checkpoints/hal2_c4.pt"

# ---------------------------------------------------------------------------
# Device selection — MPS (Apple GPU) if available, otherwise CPU
# ---------------------------------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple GPU)")
else:
    device = torch.device("cpu")
    print("MPS not available — using CPU")

# ---------------------------------------------------------------------------
# Initialise environment, replay buffers, agents
# ---------------------------------------------------------------------------

env = ConnectFourEnv()

buffer1 = ReplayBuffer(capacity=REPLAY_CAPACITY)
buffer2 = ReplayBuffer(capacity=REPLAY_CAPACITY)

hal1 = DQNAgent(player=1, device=device)
hal2 = DQNAgent(player=2, device=device)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def free_gb():
    """Return free disk space in GB for the current directory."""
    return shutil.disk_usage(".").free / (1024 ** 3)

with open(LOG_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "hal1_wins", "hal2_wins", "draws",
                     "hal1_epsilon", "hal2_epsilon", "avg_loss", "avg_game_length"])

# ---------------------------------------------------------------------------
# Demo game function
# ---------------------------------------------------------------------------

def run_demo(episode):
    """
    Play one game between HAL-1 and HAL-2 in a pygame window.
    Both agents play with epsilon=0 (pure exploitation — best known moves).
    Training is paused for the duration of the demo.
    """
    vis = Visualizer()

    saved_e1, saved_e2 = hal1.epsilon, hal2.epsilon
    hal1.epsilon = 0.0
    hal2.epsilon = 0.0

    state  = env.reset()
    done   = False
    winner = 0

    vis.draw(state, episode=episode, label="Demo game", delay=0.8)

    while not done:
        legal  = env.legal_actions()
        action = hal1.choose_action(state, legal)
        state, _, done, winner = env.step(action)
        vis.draw(state, episode=episode, label=f"HAL-1 plays column {action}", delay=0.5)
        if done:
            break

        legal  = env.legal_actions()
        action = hal2.choose_action(state, legal)
        state, _, done, winner = env.step(action)
        vis.draw(state, episode=episode, label=f"HAL-2 plays column {action}", delay=0.5)

    screenshot_path = f"logs/screenshots/demo_ep{episode:06d}.png"
    vis.show_result(winner, screenshot_path=screenshot_path)
    vis.close()
    print(f"  Screenshot saved: {screenshot_path}")

    hal1.epsilon = saved_e1
    hal2.epsilon = saved_e2

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

hal1_wins    = 0
hal2_wins    = 0
draws        = 0
loss_log     = []
game_lengths = []   # moves per episode — watch for early convergence collapse
start        = time.time()

print(f"\nTraining HAL-3000 for {NUM_EPISODES:,} episodes...\n")

for episode in range(1, NUM_EPISODES + 1):

    # --- Storage safety check ---
    if episode % LOG_INTERVAL == 0:
        free = free_gb()
        if free < MIN_FREE_GB:
            print(f"\nStorage warning: only {free:.1f}GB free. Stopping early.")
            hal1.save(CKPT_HAL1)
            hal2.save(CKPT_HAL2)
            break

    state = env.reset()
    done  = False
    move_count = 0

    # HAL-1's transition is held until HAL-2 responds, so the stored next_state
    # reflects what HAL-1 will actually see on its next turn (after both have moved).
    pending1  = None   # (state, action, reward) waiting for HAL-2's response
    last_hal2 = None   # (state, action, next_state) for HAL-2 loss penalty

    while not done:

        # --- HAL-1's turn ---
        legal   = env.legal_actions()
        action1 = hal1.choose_action(state, legal)
        next1, r1, done, winner = env.step(action1)
        move_count += 1

        if done:
            # Game ended on HAL-1's move — push immediately (no HAL-2 response coming)
            buffer1.push(state, action1, r1, next1, True)
            loss = hal1.train(buffer1, BATCH_SIZE)
            if loss is not None:
                loss_log.append(loss)
            if winner == 1:
                hal1_wins += 1
                if last_hal2 is not None:
                    s, a, ns = last_hal2
                    buffer2.push(s, a, -1, ns, True)  # penalise HAL-2
            else:
                draws += 1
            break

        # Hold HAL-1's transition — push after HAL-2 responds
        pending1 = (state, action1, r1)
        loss = hal1.train(buffer1, BATCH_SIZE)  # train on existing buffer entries
        if loss is not None:
            loss_log.append(loss)

        # --- HAL-2's turn ---
        legal   = env.legal_actions()
        action2 = hal2.choose_action(next1, legal)
        next2, r2, done, winner = env.step(action2)
        move_count += 1

        # Push HAL-1's deferred transition with the real next_state (after HAL-2 moved).
        # If HAL-2 just won, flip HAL-1's reward to -1 here rather than pushing a
        # separate penalty transition.
        hal1_r = -1 if (done and winner == 2) else pending1[2]
        buffer1.push(pending1[0], pending1[1], hal1_r, next2, done)
        pending1 = None

        # Push HAL-2's transition immediately
        buffer2.push(next1, action2, r2, next2, done)
        last_hal2 = (next1, action2, next2)

        loss = hal2.train(buffer2, BATCH_SIZE)
        if loss is not None:
            loss_log.append(loss)

        if done:
            if winner == 2:
                hal2_wins += 1
            else:
                draws += 1
            break

        state = next2

    game_lengths.append(move_count)
    hal1.decay_epsilon()
    hal2.decay_epsilon()

    # --- Logging ---
    if episode % LOG_INTERVAL == 0:
        total    = hal1_wins + hal2_wins + draws
        avg_loss = sum(loss_log)     / len(loss_log)     if loss_log     else 0.0
        avg_len  = sum(game_lengths) / len(game_lengths) if game_lengths else 0.0
        elapsed  = time.time() - start

        print(
            f"Ep {episode:>6,} | "
            f"HAL-1 {hal1_wins/total*100:4.1f}%  "
            f"HAL-2 {hal2_wins/total*100:4.1f}%  "
            f"Draw {draws/total*100:4.1f}% | "
            f"ε {hal1.epsilon:.3f} | "
            f"loss {avg_loss:.4f} | "
            f"len {avg_len:.1f} | "
            f"{elapsed:.0f}s"
        )

        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                hal1_wins, hal2_wins, draws,
                round(hal1.epsilon, 4),
                round(hal2.epsilon, 4),
                round(avg_loss, 6),
                round(avg_len, 1),
            ])

        hal1_wins    = 0
        hal2_wins    = 0
        draws        = 0
        loss_log     = []
        game_lengths = []

    # --- Periodic checkpoint and demo ---
    if episode % SAVE_INTERVAL == 0:
        hal1.save(CKPT_HAL1)
        hal2.save(CKPT_HAL2)
        print(f"  Checkpoints saved at episode {episode:,}")

    if episode % DEMO_INTERVAL == 0:
        print(f"  Opening demo window for episode {episode:,}...")
        run_demo(episode)
        print(f"  Demo done. Resuming training.")

# ---------------------------------------------------------------------------
# Final save and summary chart
# ---------------------------------------------------------------------------

hal1.save(CKPT_HAL1)
hal2.save(CKPT_HAL2)
print(f"\nTraining complete. Checkpoints saved.")

try:
    import csv as _csv
    import matplotlib.pyplot as plt

    episodes_log, h1, h2, dr, lengths = [], [], [], [], []
    with open(LOG_PATH) as f:
        reader = _csv.DictReader(f)
        for row in reader:
            total = int(row["hal1_wins"]) + int(row["hal2_wins"]) + int(row["draws"])
            if total == 0:
                continue
            episodes_log.append(int(row["episode"]))
            h1.append(int(row["hal1_wins"]) / total * 100)
            h2.append(int(row["hal2_wins"]) / total * 100)
            dr.append(int(row["draws"])     / total * 100)
            lengths.append(float(row["avg_game_length"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(episodes_log, h1, label="HAL-1 wins %", color="royalblue")
    ax1.plot(episodes_log, h2, label="HAL-2 wins %", color="tomato")
    ax1.plot(episodes_log, dr, label="Draw %",       color="grey")
    ax1.set_ylabel("% of last 500 games")
    ax1.set_title("HAL-3000 — Connect Four Training")
    ax1.legend()

    ax2.plot(episodes_log, lengths, color="mediumpurple", label="Avg game length (moves)")
    ax2.set_ylabel("Moves per game")
    ax2.set_xlabel("Episode")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(CHART_PATH)
    print(f"Learning curve saved to {CHART_PATH}")
except Exception as e:
    print(f"Chart generation failed: {e}")
