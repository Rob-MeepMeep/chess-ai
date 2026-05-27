"""
agent.py — HAL's brain.

HAL is a Q-learning agent. He has no knowledge of the rules — he only sees
board states, chooses actions, and receives rewards. Everything he knows
about Tic-Tac-Toe he learned by playing tens of thousands of games.

Key concepts:
  Q-table   : a dictionary mapping board states to an array of 9 values,
              one per cell. Higher value = HAL thinks playing here is better.
  Epsilon   : how often HAL explores (random move) vs. exploits (best move).
              Starts at 1.0 (fully random) and decays toward 0.05.
  Alpha     : learning rate — how much new evidence overwrites old belief.
  Gamma     : discount — how much HAL values future rewards vs. immediate ones.
"""

import random
import pickle
import numpy as np


class QLearningAgent:

    def __init__(self, name, alpha=0.1, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.9999):
        self.name = name
        self.alpha = alpha                  # learning rate
        self.gamma = gamma                  # discount factor
        self.epsilon = epsilon              # current exploration rate
        self.epsilon_min = epsilon_min      # floor — never goes fully greedy
        self.epsilon_decay = epsilon_decay  # multiplied each episode

        # The Q-table: state (tuple) → numpy array of 9 Q-values
        self.q_table = {}

    # ------------------------------------------------------------------
    # Core Q-learning methods
    # ------------------------------------------------------------------

    def _get_q_values(self, state):
        """
        Look up Q-values for a state. If HAL has never seen this board
        before, initialise all 9 values to 0.0.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        return self.q_table[state]

    def choose_action(self, state, legal_actions):
        """
        Epsilon-greedy action selection.

        With probability epsilon: pick a random legal move (explore).
        Otherwise: pick the legal move with the highest Q-value (exploit).

        Early in training epsilon is high so HAL mostly explores.
        Later it's low so HAL mostly plays his best known move.
        """
        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        q_values = self._get_q_values(state)
        return max(legal_actions, key=lambda a: q_values[a])

    def update(self, state, action, reward, next_state, next_legal_actions, done):
        """
        The Bellman update — HAL's learning step.

        After every move, HAL asks: 'Given what just happened, was my
        Q-value for that move accurate?' Then nudges it toward reality.

        If the game is over (done=True):
            target = reward                          (no future moves)
        Otherwise:
            target = reward + gamma * best_future_Q  (reward + discounted future)

        Then: Q(state, action) += alpha * (target - Q(state, action))
              ↑ old value stays mostly, nudged slightly toward target
        """
        q_values = self._get_q_values(state)

        if done:
            target = reward
        else:
            next_q = self._get_q_values(next_state)
            best_future = max(next_q[a] for a in next_legal_actions)
            target = reward + self.gamma * best_future

        q_values[action] += self.alpha * (target - q_values[action])

    def decay_epsilon(self):
        """
        Called once per episode. Slowly shifts HAL from explorer to exploiter.
        With decay=0.9999 over 50,000 games, epsilon reaches its minimum
        around game 30,000.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Introspection — see the world through HAL's eyes
    # ------------------------------------------------------------------

    def explain(self, state):
        """
        Print HAL's Q-values for every empty cell in the given position.
        Occupied cells show the piece instead. Lets you see what HAL
        thinks of any board position at any point in training.
        """
        symbols = {1: '  X  ', 2: '  O  '}
        q = self._get_q_values(state)

        print(f"\n{self.name}'s read on this position "
              f"(epsilon={self.epsilon:.3f}, states seen={len(self.q_table):,}):")
        for row in range(3):
            cells = []
            for col in range(3):
                idx = row * 3 + col
                if state[idx] != 0:
                    cells.append(symbols[state[idx]])
                else:
                    cells.append(f'{q[idx]:+.3f}')
            print(' | '.join(cells))
            if row < 2:
                print('-' * 23)
        print()

    # ------------------------------------------------------------------
    # Persistence — save and load HAL's knowledge
    # ------------------------------------------------------------------

    def save(self, path):
        """Save HAL's Q-table and current epsilon to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'name': self.name,
            }, f)
        print(f"{self.name} saved to {path} ({len(self.q_table):,} states learned)")

    def load(self, path):
        """Load a previously saved Q-table — HAL picks up where he left off."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table = data['q_table']
        self.epsilon = data['epsilon']
        print(f"{self.name} loaded from {path} ({len(self.q_table):,} states remembered)")
