"""
agent.py — HAL-3000: DQN agent for Connect Four.

DQN (Deep Q-Network) is the neural-network version of the Q-learning
HAL used in Phase 1. The core idea is the same:
    - choose actions using epsilon-greedy (explore vs exploit)
    - after every move, update Q-values toward the Bellman target

What's different from Phase 1:
    - Q-values come from a neural network, not a lookup table
    - training happens in batches sampled from the replay buffer
    - a second frozen network (target_net) provides stable training targets

Two networks, one agent:
    policy_net  — trained every step; HAL's current best guess
    target_net  — frozen copy of policy_net; only updated every N steps

Why two networks? If you train toward targets produced by the same network
you're updating, the targets shift with every weight change — like chasing
a moving shadow. target_net holds still long enough for training to converge,
then gets updated to match policy_net, then holds still again.
"""

import random
import torch
import torch.nn as nn

from connect4.model import ConnectFourNet, encode_state


class DQNAgent:

    def __init__(
        self,
        player,          # 1 or 2 — which player this agent is
        device,          # torch.device (MPS or CPU)
        lr=1e-3,         # learning rate for Adam optimiser
        gamma=0.99,      # discount factor — how much future rewards matter
        epsilon=1.0,     # starting exploration rate (100% random)
        epsilon_min=0.02,
        epsilon_decay=0.9999,
        target_update_freq=500,  # copy policy_net → target_net every N training steps
    ):
        self.player = player
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.steps = 0  # counts training steps (not episodes) for target update

        # Build both networks and move them to the right device
        self.policy_net = ConnectFourNet().to(device)
        self.target_net  = ConnectFourNet().to(device)

        # Initialise target_net as an exact copy of policy_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # target_net is never trained directly — always in eval mode
        self.target_net.eval()

        # Adam: adaptive learning rate optimiser — adjusts each weight's
        # update individually based on recent gradients. More stable than
        # plain gradient descent for deep networks.
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state, legal_actions):
        """
        Epsilon-greedy action selection — same logic as Phase 1.

        With probability epsilon: pick a random legal column (explore).
        Otherwise: ask the network for Q-values, pick the best legal one (exploit).

        Tie-breaking: if multiple columns share the best Q-value, choose
        randomly among them — same fix as the Phase 1 bug.
        """
        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        # Get Q-values from the policy network (no gradient tracking needed)
        q_values, _ = self.policy_net.predict(state, self.player, self.device)

        # Only consider legal columns; apply Phase 1 tie-breaking fix
        best_value = max(q_values[a] for a in legal_actions)
        best_actions = [a for a in legal_actions if q_values[a] == best_value]
        return random.choice(best_actions)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, replay_buffer, batch_size=64):
        """
        Sample a random batch from the replay buffer and update policy_net.

        Returns the loss value (float) for logging, or None if the buffer
        doesn't have enough transitions yet.
        """
        if len(replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # --- Build tensors ---

        # Encode each state from HAL's perspective: (2, 6, 7) per state
        # torch.stack turns a list of tensors into a batched tensor: (batch, 2, 6, 7)
        state_t      = torch.stack([encode_state(s, self.player) for s in states]).to(self.device)
        next_state_t = torch.stack([encode_state(s, self.player) for s in next_states]).to(self.device)

        action_t = torch.tensor(actions, dtype=torch.long).to(self.device)    # (batch,)
        reward_t = torch.tensor(rewards, dtype=torch.float32).to(self.device) # (batch,)
        done_t   = torch.tensor(dones,   dtype=torch.float32).to(self.device) # (batch,)

        # --- Predicted Q-values (from policy_net) ---

        # Forward pass: (batch, 7) Q-values for every column
        q_all, _ = self.policy_net(state_t)

        # gather() picks the Q-value for the action that was actually taken.
        # unsqueeze(1) reshapes action_t from (batch,) to (batch, 1) so gather
        # can index along the column dimension, then squeeze(1) removes it again.
        # Result: (batch,) — one Q-value per transition.
        q_predicted = q_all.gather(1, action_t.unsqueeze(1)).squeeze(1)

        # --- Action mask for next states ---

        # A column is full if the top row (indices 0–6 of the flat 42-tuple) is non-zero.
        # We must ignore full columns when computing the best future action — otherwise
        # the network can assign high Q-values to illegal moves and corrupt the target.
        next_states_raw = torch.tensor(next_states, dtype=torch.float32)  # (batch, 42)
        full_col_mask = (next_states_raw[:, :7] != 0).to(self.device)     # (batch, 7)

        # --- Target Q-values (Double DQN) ---

        with torch.no_grad():
            # Double DQN: decouple action *selection* from action *evaluation*.
            # Vanilla DQN uses target_net for both — it tends to overestimate Q-values
            # because the same network that picks the best action also scores it.
            # Double DQN fix: policy_net picks the best legal action; target_net scores it.

            # Step 1 — policy_net picks the best legal action for each next state
            next_q_policy, _ = self.policy_net(next_state_t)
            next_q_policy = next_q_policy.masked_fill(full_col_mask, float('-inf'))
            best_next_actions = next_q_policy.argmax(dim=1)  # (batch,)

            # Step 2 — target_net evaluates those specific actions (no argmax bias)
            next_q_target, _ = self.target_net(next_state_t)
            q_next = next_q_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

            # Bellman target:
            #   if done: target = reward only (no future — game is over)
            #   if not:  target = reward + gamma × best future Q-value
            # (1 - done_t) zeroes out the future term when the episode ended.
            q_target = reward_t + self.gamma * q_next * (1 - done_t)

        # --- Loss and backpropagation ---

        # MSE loss: how far off were our Q-value predictions?
        loss = nn.functional.mse_loss(q_predicted, q_target)

        self.optimizer.zero_grad()  # clear gradients from last step
        loss.backward()             # compute new gradients via backpropagation
        self.optimizer.step()       # nudge weights in the direction that reduces loss

        # --- Periodically update the target network ---

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        """Reduce exploration rate by one step. Call once per episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path):
        """Save everything needed to resume training or run evaluation."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net':  self.target_net.state_dict(),
            'optimizer':   self.optimizer.state_dict(),
            'epsilon':     self.epsilon,
            'steps':       self.steps,
        }, path)

    def load(self, path):
        """Load a checkpoint. map_location handles CPU↔MPS differences."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps   = checkpoint['steps']
