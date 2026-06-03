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
the GPU cores in the M-series chip — significantly faster for neural
network training.

---

## Hardware Notes (MacBook Pro M5 Pro, 2025)

- 24GB unified memory — shared between CPU and GPU
- Active cooling (fan) — sustained training runs are fine
- MPS backend available for PyTorch GPU acceleration
- At 200 simulations per move: ~90–115 seconds per game
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

---

## Phase 3 — AlphaZero-Style Chess (HAL-4000)

Phase 3 is a step change in complexity. We move from a network that outputs
a single Q-value for a single action, to a network that plays the whole game
from a position — evaluating how good the position is AND suggesting which
moves to explore. The training algorithm (AlphaZero) is fundamentally
different from DQN.

### Board Encoding
Before the network can think about a chess position, it needs to see it as
numbers. We convert the board into a 3D tensor: 54 planes × 8 rows × 8
columns. The first 48 planes encode piece locations across the current and
last 3 positions (12 piece types × 4 time steps). The remaining 6 planes
encode castling rights, en passant squares, and the 50-move clock.

Each plane is a binary 8×8 grid — 1 means "there is a white bishop on this
square", 0 means there isn't. The network sees the whole board at once,
not one square at a time.

**Why history planes?** Chess is not fully described by the current board
alone — you need to know if castling rights have been lost, whether a pawn
can take en passant, and whether a position is being repeated. The history
planes give the network a short memory.

### Residual Network (ResNet)
A deep neural network where each block has a skip connection — the input
is added directly to the output of two conv layers:

```
output = relu(x + conv_block(x))
```

Without skip connections, gradients shrink as they travel backwards through
many layers ("vanishing gradient"), making deep networks hard to train.
The skip connection gives gradients a direct route back to early layers.
This is what makes it practical to stack 10, 20, or more blocks.

We use 10 residual blocks and 160 channels — about 3× smaller than the
original AlphaZero's 20 blocks and 256 channels, scaled for our hardware.

### Policy Head and Value Head
The network has two outputs — two "heads" — branching off the same
learned representation:

**Policy head:** outputs 4096 numbers — one for every possible
(from-square, to-square) move pair on a chessboard, whether legal or not.
Illegal moves are masked to −∞ before use. After training, the policy head
tells MCTS which moves are worth exploring.

**Value head:** outputs a single number in [−1, +1]. +1 means "the current
player is winning", −1 means "the current player is losing". This is what
MCTS uses to evaluate positions without playing them to the end.

The two heads are trained simultaneously on the same network. The shared
lower layers learn what makes a position good or bad; the heads specialise
on top of that shared understanding.

### MCTS (Monte Carlo Tree Search)
The search algorithm that decides which move to play. Where DQN picked the
move with the highest Q-value (greedy), MCTS builds a search tree by
repeatedly simulating "what would happen if I played here?"

Each simulation:
1. Traverse the tree using UCB selection — prefer nodes that are both
   promising (high policy probability) and underexplored (low visit count)
2. When reaching a leaf (unexplored position), ask the network for a value
3. Propagate that value back up the tree

After N simulations, pick the move with the most visits — not the highest
single value, but the one the search consistently returned to.

**Why visits rather than value?** A move can look great in isolation but
be refutable with one strong reply. MCTS finds that refutation by exploring
deeply. High visit count means the move held up under scrutiny.

### UCB Selection (Upper Confidence Bound)
The formula that balances exploration vs exploitation inside MCTS:

```
score = Q(s,a) + c × P(s,a) × sqrt(N(s)) / (1 + n(s,a))
```

Q = mean value seen so far, P = network's prior probability for this move,
N = total visits to this position, n = visits to this specific move, c = exploration constant.

Moves the network thinks are good (high P) get a head start. Moves that
haven't been explored much (low n) get a bonus. As a move is visited more,
its bonus shrinks and its true value Q dominates.

### AlphaZero Self-Play
Unlike DQN, there is no separate target network and no Bellman equation.
Instead, one network plays both sides of every game. After each game, every
position gets labelled with the actual game outcome (+1 or −1 from the
current player's perspective) and the MCTS visit distribution as the policy
target. These (position, policy, outcome) triples go into the replay buffer.

Training then asks the network to match its own best search: predict the
outcome and predict which moves MCTS visited most. Over thousands of games,
the network gets better at predicting what MCTS will do — and since MCTS
uses the network to guide search, the whole system lifts together.

### Perspective-Relative Outcomes
Every position in the buffer stores the outcome from the point of view of
whoever was to move at that position. If white won and it was white's turn
at position X, the label is +1. If white won and it was black's turn at
position X, the label is −1. The network always sees the world from "my
perspective" rather than an absolute white/black frame.

---

## Phase 3 Lessons Learned

### L13 — Draw collapse: when the value head learns nothing
The most dangerous failure mode in AlphaZero-style training. If games
end in draws (50-move rule, repetition, stalemate), every position gets
labelled 0. The value head learns to predict 0 everywhere — it's never
wrong by much. Once this happens, MCTS has no signal to distinguish good
positions from bad ones. All moves look equal. The network plays randomly.
More training makes it worse, not better.

The fix is resign logic: detect losing positions by material imbalance or
value head confidence, and end the game early with a decisive result.
Draws from resigned games should be rare.

### L14 — The encoder can silently teach the network to be racist
In Run 4, HAL developed a persistent black-wins bias (~55–60%). The root
cause: the original encoder had a plane 48 that was set to 1.0 when it was
white's turn and 0.0 when it was black's turn. Over 2,300 games, the network
learned to correlate that plane with losing — because in self-play, positions
where the network had committed mistakes tended to be white-to-move positions
(white plays first, makes the first errors).

This is a colour indicator masquerading as a positional feature. The fix
was to remove the plane entirely, making the encoder colour-blind. The
network now sees every position from the current player's perspective with
no absolute colour information — identical to how a strong player thinks.

### L15 — The bootstrapping problem: early self-play is noise
When the network is initialised with random weights, both players play
randomly. The game outcomes are random. The policy targets (MCTS visit
distributions) reflect random search. The value labels are meaningless.
Training on this data teaches the network to predict random outcomes.

This is not a bug — it's an inherent property of learning from self-play
from scratch. The network cannot generate good training data until it's
already somewhat trained, but it can't get trained without good data. The
first ~500 games are largely noise. The value head is effectively blind
during this period.

### L16 — Buffer seeding can skip the bootstrapping phase
If the replay buffer is pre-loaded with positions from a previous (better)
training run, gradient step 1 has real signal. The value head starts
training on positions with known, meaningful outcomes rather than random
noise. In Run 7, training loss was non-zero from game 1 (confirming the
buffer loaded correctly) and value resigns appeared by game 100 instead
of game 500.

The tradeoff: seed positions have zero policy labels (we don't store MCTS
visit counts in the game log). The policy head gets no benefit from seed
data — only the value head does. This is acceptable because the value head
is the harder and more important thing to bootstrap.

### L17 — Canonical endgame positions give the value head ground truth
Positions like K+Q vs K are trivially decisive — white wins, always. The
correct value label is +1.0 (from white's perspective) or −1.0 (from
black's). These are positions the network should learn to rate confidently.

But they almost never appear in self-play. Resign fires long before kings
are left alone. Without specific training examples, the value head has no
direct signal for pure endgames.

Adding canonical positions directly to the replay buffer gives the value
head ground truth it would otherwise never encounter. A network that
correctly rates K+Q vs K near ±1 has learned something real about material
and king safety.

### L18 — The tiered replay buffer: some knowledge is permanent
A standard circular buffer evicts the oldest positions as new ones arrive.
Once the buffer fills, seed positions (including canonical endgames) are
lost. The value head loses its ground truth signal.

A tiered buffer separates positions into two partitions: a rolling buffer
that behaves normally (oldest evicted first), and a permanent buffer that
is never evicted and always sampled at a fixed rate. Ground-truth endgame
positions belong in permanent; self-play positions belong in rolling.

At ~12.5% of each training batch, the permanent partition keeps a constant
thread of canonical signal running throughout the entire training run.

### L19 — Rising loss is not always a sign of failure
In Run 7, average training loss rose from ~1.1 at game 50 to ~2.9 at game
300. This looks alarming. It isn't.

The seed buffer was built with zero policy labels. Early training loss was
artificially low — the policy head had nothing hard to predict. As self-play
positions (with real, peaked MCTS distributions) replaced seed positions in
the buffer, the policy head faced harder targets. Loss rose to reflect the
harder problem being solved.

The diagnostic question is not "is loss rising?" but "are value resigns
increasing?" Value resigns growing (0 → 1 → 2 → 3 → 6 per 50-game window
in Run 7) while loss rises means the network is becoming more decisive, not
less. That is health, not collapse.

### L20 — The resign mechanism is both a fix and a constraint
Resign solves draw collapse (L13) by ensuring games end decisively. But it
also sets a ceiling on what the network can learn: if resign fires whenever
material imbalance exceeds a threshold, the network never sees positions
where a material deficit is overcome by tactics. It learns to give up rather
than fight back.

The parameters — RESIGN_MATERIAL (resign when down this many points) and
RESIGN_CONSECUTIVE (require this many consecutive hopeless evaluations) —
are a deliberate tradeoff. Set them too loose and draw collapse returns.
Set them too tight and the network never learns closing technique.

### L21 — The loss metric reflects what the buffer contains
Training loss measures the average prediction error across recent batches.
But the buffer changes composition over time — from seed data (zero policy,
simple value) to self-play data (peaked policy, complex value). A drop in
loss can mean the network got better, or it can mean the data got easier.
A rise in loss can mean the network got worse, or the data got harder.

Loss must always be read alongside the end-reason distribution and tally
balance. Loss alone is not a reliable health signal.

---

*This document is updated as new concepts are introduced. Last updated: Phase 3 Run 7 in progress.*
