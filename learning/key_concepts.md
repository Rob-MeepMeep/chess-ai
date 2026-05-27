# Key Learning — chess-ai Project
## Maintained by Ellis Ward. Updated throughout the project.

This document builds up as we go. Each concept is explained at the level
we understood it when we first encountered it — not a textbook, a record
of genuine learning.

---

## Reinforcement Learning Fundamentals

### The Environment
The rulebook. Knows what moves are legal, what the board looks like, when
the game is over, and what reward to give. Has no strategy — it just
enforces rules and responds to actions.

**Key interface:** `reset()` → initial state, `step(action)` → (new state, reward, done)

### The Agent
The learner. Sees states, chooses actions, receives rewards. Knows nothing
about the rules — learns purely from experience.

### State
A snapshot of the world at a given moment. In our games, the board
represented as a tuple of integers. Must be immutable (tuple not list) so
it can be used as a dictionary key.

### Action
A choice the agent makes. In Tic-Tac-Toe: which cell (0–8). In Connect
Four: which column (0–6).

### Reward
The signal that tells the agent if what it did was good or bad.
+1 = win, -1 = loss, 0 = draw or game ongoing.
The agent's entire goal is to maximise total reward over time.

### Episode
One complete game from start to finish.

---

## Q-Learning (Phase 1 — Tic-Tac-Toe)

### The Q-Table
A dictionary: `{board state: [value for each action]}`. HAL's notebook.
Higher value = HAL thinks playing there is better. Starts all zeros.
Updated after every move using the Bellman equation.

### The Bellman Equation
```
Q(s, a) += alpha × (reward + gamma × max(Q(s', a')) − Q(s, a))
```
After every move, nudge the Q-value for what was just done toward reality.
Target = immediate reward + discounted best future value.

### Alpha (Learning Rate)
How much to trust new information vs old belief. 0.1 = cautious updates.
Too high = unstable. Too low = learns too slowly.

### Gamma (Discount Factor)
How much future rewards are worth compared to immediate ones. 0.9 = future
matters but slightly less than now. Causes reward signal to weaken over
long sequences — the opening move only "feels" 0.9^6 ≈ 0.53 as important
as the final move.

### Epsilon-Greedy Exploration
With probability epsilon: explore (random move).
With probability 1−epsilon: exploit (best known move).
Epsilon starts at 1.0 (fully random) and decays toward a minimum floor.
Ensures HAL tries new things early, commits to what works later.

### Self-Play
Two agents play each other. Both learn simultaneously. The risk: they can
converge to a stable but suboptimal strategy pair — one learns to beat the
other's specific weaknesses rather than discovering universally good play.

---

## Phase 1 Lessons Learned

### L1 — Tie-breaking matters
When Q-values are equal, Python's `max()` always returns the first element.
HAL always played cell 0, HAL-O always played cell 1. Same game repeated
50,000 times. Fix: randomly choose among all actions sharing the max value.

### L2 — Reward propagation is slow
With gamma=0.9 over 6 moves, the opening Q-value is only ~53% as strong
as the terminal reward. Even after 200,000 games, opening values stayed
weak. More training didn't help once the full state space was explored.

### L3 — More training has diminishing returns
HAL explored all 4,520 reachable states by episode ~15,000. The remaining
185,000 episodes produced no measurable improvement. Training time is only
valuable while new experience is being gained.

### L4 — Q-learning memorises, it doesn't understand
The Q-table stores a value for every specific position seen. Two positions
that are strategically identical but look different are treated as
completely unrelated. HAL cannot generalise — cannot reason "this looks
like a situation where I should block."

### L5 — The ceiling of Q-learning
For small state spaces (Tic-Tac-Toe: 4,520 states) Q-learning works but
plateaus. For large state spaces (Connect Four: millions of states, Chess:
~10^44 positions) a Q-table is not feasible — too many states to memorise.
This is why neural networks exist.

---

## Neural Networks (Phase 2 — Connect Four)

### What a Neural Network Is
A function approximator. Instead of storing a value for every state in a
table, it learns *weights* that can compute a value for any state — including
ones it has never seen. It generalises from experience rather than memorising.

### Layers
Data flows through the network in stages. Each layer transforms the data,
learning increasingly abstract patterns. Layer 1 might notice raw features
("three pieces in a row here"), deeper layers understand strategy ("that's
a fork threat").

### Neurons
Individual processing units within a layer. Each neuron receives inputs,
multiplies them by its weights, adds a bias, and passes the result through
an activation function.

### Weights and Biases
The numbers the network learns. Initialised randomly. Adjusted during
training via backpropagation. The entire "knowledge" of the network is
stored in these values.

### Forward Pass
Data flows in at the input layer, is transformed at each hidden layer,
and produces an output. This replaces the Q-table lookup.

### Loss Function
Measures how wrong the network's prediction was. Lower = better.
Training tries to minimise this value.

### Backpropagation
After computing the loss, the error is traced backwards through every
layer. Each weight is nudged slightly in the direction that reduces the
loss. This is how the network learns.

### Optimizer (Adam)
The algorithm that performs the weight updates. Adam is adaptive —
it adjusts the learning rate for each weight individually based on
recent gradients. More stable than basic gradient descent.

### Experience Replay
Store every move ever made in a buffer. During training, sample random
batches from the buffer rather than training on consecutive moves.
Breaks correlation between consecutive samples — makes training stable.
Without it, the network oscillates.

### PyTorch
The library we use to build and train neural networks. We describe the
shape of the network; PyTorch handles the maths of forward passes,
backpropagation, and weight updates automatically.

### MPS (Metal Performance Shaders)
Apple's GPU acceleration backend for PyTorch on M-series Macs.
Activated with: `torch.device("mps")`. Moves computation from CPU to
the GPU cores in the M3 chip — significantly faster for neural network
training.

---

## Hardware Notes (MacBook Air M3, 2024)

- 16GB unified memory — shared between CPU and GPU
- ~10-12GB realistically available for ML training
- No fan — throttles under sustained load; train overnight or in chunks
- MPS backend available for PyTorch GPU acceleration
- Storage threshold: stop training if less than 10GB free

---

## Phase 2 Lessons Learned

### L6 — Q-value divergence is the central DQN stability problem
Without safeguards, Q-values spiral upward in a feedback loop: the network
starts winning consistently, reinforces those winning moves heavily, the
Q-values grow large, which become training targets, which push them larger.
The loss explodes exponentially and the weights become meaningless.
Run 1 reached a loss of 321,000 by episode 10,500 without any safeguards.

### L7 — Gradient clipping delays but does not prevent divergence
Capping the gradient norm (max_norm=10.0) prevents any single weight update
from being catastrophically large. In Run 2 this delayed the divergence by
~10,000 episodes and allowed the network to self-correct once (loss dropped
from 15.5 at ep8500 back to 1.6 at ep9000). But with a learning rate of
1e-3, the updates were still large enough to diverge eventually — reaching
a loss of 2,608 and game lengths of 9.7 moves by episode 18,500.

### L8 — Learning rate is the root cause, not just the symptom
A learning rate of 1e-3 is too aggressive for DQN on Connect Four. Each
weight update is large enough that Q-values drift faster than the target
network can stabilise them. Standard DQN uses 1e-4 or lower. Combining
a lower learning rate (1e-4) with tighter gradient clipping (max_norm=1.0)
is the correct fix — both levers together constrain how fast Q-values move.

### L9 — Game length is the clearest collapse signal
Average game length dropping sharply is a more reliable early warning than
win rate alone. In both runs, game length started falling before the loss
exploded or win rates became extreme. A game averaging 10 moves means one
player is winning in roughly 5 moves each — near the physical minimum for
Connect Four and a definitive sign of co-evolutionary collapse.

### L10 — The exploit/counter cycle is real and measurable
In Run 2, between episodes 8,000–12,500, a clear cycle was visible:
HAL-1 found an exploit (win rate rose to 65–70%), the loss spiked, gradient
clipping helped the network recover (loss self-corrected to 0.83), and
HAL-2 adapted (win rates returned to 54/46 by ep12500). This is the
competitive dynamic self-play is designed to produce. It was only visible
because the divergence was delayed long enough by gradient clipping to
observe it.

### L11 — Action masking must apply in training, not just inference
When computing the Bellman target, the max Q-value over next-state actions
must exclude full columns. If illegal columns can have high Q-values, they
corrupt the training target silently — the loss looks normal but the network
is learning from bad targets. This was caught before any training run
(credit: external review by Dave).

### L12 — Perspective-correct transitions require deferred push
In alternating self-play, if HAL-1's transition is pushed immediately after
HAL-1 moves, the stored next_state is the board before HAL-2 responds. The
Q-value computed for that state is "best move from here" — but HAL-1 won't
see that state on its next turn. Fix: hold HAL-1's transition until HAL-2
has moved, then push with the real next_state.

---

*This document is updated as new concepts are introduced. Last updated: Phase 2 Run 2 complete.*
