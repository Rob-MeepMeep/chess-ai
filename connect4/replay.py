"""
replay.py — Experience replay buffer for HAL-3000.

Every move made during training gets stored here as a transition:
    (state, action, reward, next_state, done)

During training, random batches are sampled from this buffer rather than
training on consecutive moves. This breaks the correlation between
sequential experiences and stabilises neural network training.

The buffer has a fixed maximum size. Once full, the oldest experiences
are overwritten — recent experience is more relevant than ancient history.
"""

import random
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity=50_000):
        # deque with maxlen automatically discards the oldest entry when full
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store one transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions.
        Returns five separate lists — one per field — ready to convert
        to tensors in the agent.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Allows: if len(buffer) >= batch_size — before we start training."""
        return len(self.buffer)
