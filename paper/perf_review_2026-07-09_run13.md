# chess-ai Performance Review — Run 13 Bottleneck Autopsy (main @ 38cb26d)

**Date:** 9 July 2026
**Authors:** Rob Kirkland, Ellis Ward
**Target hardware:** Ryzen 5 9600X (6C/12T) + Radeon RX 9070 XT, ROCm on WSL2 (Ubuntu 24.04)
**Symptom:** GPU at ~71% utilization / ~72W — idling between inference calls while the
CPU-side Python MCTS feeds it. Companion document to `code_review_2026-07-09_run13.md`
(correctness); this one is throughput only.

Caveat up front: this analysis is from code reading, not profiling on the target box.
The ranking below is what the code structure predicts; a 60-second profile should
confirm it before the larger refactors (see "Measure first" at the end).

---

## 1. Architecture map (data pipeline)

```
train_chess.py game loop (single process, single game at a time)
  └─ per ply:
       encode(board+history) ──────────────── CPU, python loops (encoder.py)
       agent.choose_move → MCTS.search ────── 600 sims in waves of BATCH_SIMS=8
       │    per wave: select 8 leaves (python-chess board.copy per tree edge)
       │              1 GPU inference call, batch ≤ 8
       │              expand + backup (pure Python)
       agent.get_value(board) ──────────────── 1 extra GPU call, batch 1 (resign check)
       GameBuffer.push (state, policy, turn)
  └─ per game:
       GameBuffer.commit → ReplayBuffer (deque, dense 4096-float policies)
       5 × agent.train(batch=512) ──────────── GPU, healthy batch size
       every 10 games: torch.save(~GB-scale buffer) — synchronous, blocks the loop
```

The training step is fine — batch 512 keeps the GPU busy. All the starvation is in
self-play, which is >95% of wall-clock: **~77 GPU round-trips per ply at batch ≤ 8**
(75 waves + root expansion + resign-check value call), with pure-Python tree work
between every one of them. A 24.2M-parameter net at batch 8 finishes in low
single-digit milliseconds; the GPU then waits for Python. That is the 72W.

---

## 2. Bottleneck autopsy (ranked)

### B1. Inference batch is 8 — `BATCH_SIMS` (chessai/mcts.py:35)
600 sims/move ÷ 8 = 75 sequential GPU calls per move. The RX 9070 XT would serve
batch 64–256 at nearly the same latency as batch 8. Virtual loss is already
implemented (mcts.py:201-209), so raising the wave size is a constant change — the
mechanism designed to make large waves work is already there and being wasted at 8.
Trade-off: very large waves on a single tree degrade search quality (virtual loss
forces diversity onto one 600-sim tree). 32–64 is the safe single-tree range; beyond
that, batch **across games** instead (see §3).

### B2. Per-child GPU→CPU syncs during expansion (chessai/mcts.py:130-136)
`priors[idx].item()` runs once per legal move per expanded leaf, and each `.item()`
on a device tensor is a host-device synchronization. Per wave: up to 8 leaves × ~30
legal moves ≈ 240 syncs; per move: 75 waves × 240 ≈ **18,000 tiny device round-trips**.
At even 10–20µs each this is hundreds of milliseconds per move — comparable to the
inference itself. Same pattern at the root (mcts.py:198) and `value_batch[i].item()`
per leaf (mcts.py:136). Fix: one `.cpu()` for the whole priors/values batch per wave,
then index numpy arrays.

### B3. Mirror-index tensor re-uploaded on every call (chessai/moves.py:67-68)
`get_mirror_indices(device)` caches the CPU tensor but runs `.to(device)` on **every
call** — a fresh 4096-int64 host→device copy per black-to-move leaf expansion and per
black training-target mirror. Cache per device (one-line fix).

### B4. python-chess overhead in the hot path
- `sim_board.copy()` **plus** a history-list rebuild per tree edge traversed
  (mcts.py:173-174), and a root `board.copy()` per simulation (mcts.py:87-89).
  600 sims × average depth ⇒ thousands of full board copies per move.
- `board.mirror()` per history frame for black positions (encoder.py:47) — mirror()
  constructs a whole new board; with 4-frame history that's up to 4 per encode, and
  there is one encode per leaf.
- `legal_moves` is generated twice per expansion — once in `legal_move_mask`
  (moves.py:50) and again in the children loop (mcts.py:130). Legal-move generation
  is the most expensive python-chess operation; generate once, reuse for both.
- `encode()` itself fills 54 planes with Python loops over `b.pieces()` (encoder.py:49-53).
  Bitboard-to-numpy tricks (python-chess exposes piece bitboards as ints;
  `np.unpackbits` on the 8-byte view fills a plane without a Python loop) cut this
  several-fold.

### B5. Redundant GPU calls per ply
- `agent.get_value()` for the resign check (train_chess.py:173) is a batch-1 GPU
  round-trip per ply — but MCTS just computed a far better estimate of the same
  quantity: the root Q / child values. Reuse the search result; delete the call.
- `_expand_node` for the root (mcts.py:184-199) is a separate synchronous batch-1
  call; fold the root into the first wave.

### B6. Replay buffer mechanics
- `random.sample` over a 200k-entry **deque** (replay.py:63-67) — deque indexing is
  O(n), so each 512-batch pays ~50M pointer hops, ×5 batches/game. Switch to a
  list or preallocated numpy/torch ring buffer.
- Dense 4096-float32 policy per position ⇒ ~6GB buffer, `torch.save`d synchronously
  every 10 games (train_chess.py CHECKPOINT_EVERY=10). Blocks self-play for the
  duration of a multi-GB write. Store policies sparsely (legal-move indices + probs,
  typically ~30 entries instead of 4096: ~99% smaller), and/or save the buffer far
  less often than the weights.

### B7. Single core doing everything
One Python process runs tree traversal, encoding, python-chess, and GPU dispatch.
The other 5 cores contribute nothing to self-play. This is the ceiling that remains
after B1–B6, and it's what §3 addresses.

### Free wins on the inference side
- `torch.inference_mode()` instead of `no_grad` in MCTS/agent.
- bf16 autocast for self-play inference (RDNA4 handles bf16 well; training can stay
  fp32) — roughly halves inference time and memory traffic.
- Fixed input shapes mean MIOpen kernel find cost is paid once; keep shapes constant
  (they already are).

---

## 3. Parallelization strategy (staged, WSL2-safe)

**Stage A — single-process quick wins (B1–B5).** No architectural change:
`BATCH_SIMS` 8→32-64, batch the `.cpu()` transfers, cache mirror indices per device,
single legal-move generation per expansion, reuse root value for resign, fold root
into wave 1, inference_mode + bf16. Expected effect: GPU calls per move drop ~5-8×,
Python overhead per call drops substantially. Low risk; each item independently
revertible. Verify with a 20-game timing run + an eval vs the previous checkpoint
(wave-size changes can shift search quality; 32-64 with VL=1.0 is conservative).

**Stage B — multi-game lockstep batching (one process).** Restructure `MCTS.search`
into a step-driven form so G concurrent games each contribute a wave of leaves to a
single inference call: G=16 games × wave 8 = **batch 128** without increasing
per-tree virtual-loss distortion at all. No IPC, no multiprocessing, so zero WSL2
deadlock surface. This is the best quality-per-effort step and the recommended core
refactor. Self-play games are independent, so game logic needs no changes — only the
search loop and the train_chess game loop (drive G games round-robin, commit each as
it finishes).

**Stage C — worker processes (only if still CPU-bound after B).** 4–5 self-play
worker processes (CPU: tree ops, python-chess, encoding) + one GPU owner process
(inference server) fed via `torch.multiprocessing` queues. WSL2 rules: `spawn` start
method (never fork after ROCm init — classic deadlock), exactly one process touches
the GPU, keep the repo/checkpoints/buffer on the ext4 filesystem (not /mnt/c — 9p
I/O would erase the gains), size /dev/shm in .wslconfig if sharing tensors.
This uses all 6 cores; combined with Stage B batching it is the full
actor/inference-server pattern LC0-style setups use.

**Not recommended:** C++ rewrite of the board logic. The staged plan above should
move games/hour by an order of magnitude before language choice is the constraint,
and it keeps the codebase learnable — which is a design goal of this project.

---

## 4. Concrete refactor list (file / change / risk)

| # | File | Change | Risk |
|---|------|--------|------|
| 1 | chessai/mcts.py:35 | BATCH_SIMS 8 → 32-64 (make it a constructor arg) | low — eval-check |
| 2 | chessai/mcts.py:118-136,184-199 | one `.cpu()` per wave for priors+values; index numpy; same at root | none — pure mechanics |
| 3 | chessai/moves.py:56-69 | per-device cache dict for mirror indices | none |
| 4 | chessai/mcts.py:126-131 | generate `legal_moves` list once per expansion; build mask from it | none |
| 5 | train_chess.py:173 | drop `get_value` per ply; return root value from `choose_move` and use it for the resign check | low — resign signal now search-based (better) |
| 6 | chessai/mcts.py | fold root expansion into first wave; `torch.inference_mode`; optional bf16 autocast flag | low |
| 7 | chessai/encoder.py:38-55 | bitboard → numpy plane fill (np.unpackbits path) | low — assert-equal test vs old encoder |
| 8 | chessai/replay.py | list/ring instead of deque for sampling; sparse policy storage; decouple buffer-save interval from weight-save | low-medium (buffer file format changes — keep legacy loader) |
| 9 | chessai/mcts.py + train_chess.py | Stage B: step-driven MCTS + G-game lockstep loop | medium — the real refactor |
| 10 | new selfplay worker module | Stage C actors + inference server (only if needed after 9) | medium-high |

Items 1–8 are an afternoon of changes and are individually testable. Item 9 is the
structural one. Item 10 only if profiling after 9 still shows CPU saturation.

## Measure first (on the desktop)

Before Stage B, confirm the ranking with a one-minute sample during self-play:

```
py-spy top --pid <train_chess pid>          # live: where CPU time actually goes
py-spy record -o profile.svg --pid <pid> -d 60   # flamegraph for the run notes
```

Expect `copy`/`legal_moves`/`mirror` (python-chess), `encode`, and `.item()` sync
time to dominate. If they don't, the plan above gets re-ranked before any refactor.

Success metric: games/hour and GPU utilization/power, tracked in run notes per stage —
not loss curves, which this work should not affect (except via more games).

---

# Addendum (17 July 2026) — Stage A/B results and revised Stage C design

**Authors:** Rob Kirkland, Ellis Ward

## What Stage A + B actually delivered (measured, run14)

| Metric | Before (run13_retune) | After (run14, lockstep ×16) |
|---|---|---|
| GPU utilisation / board power | ~71% / ~72W | 100% / ~150W |
| CPU | one core flatlined | ~55% package (one Python thread still the wall) |
| Games/hour | 29–33 | **26–28** |

Verdict: the GPU-starvation diagnosis was correct and is fixed — and it
**didn't matter for throughput**. The binding constraint was never inference;
it is leaf *production*: one Python thread walking trees (a `board.copy()`
per edge), generating legal moves in pure-Python python-chess, and running
the plane-loop `encode()` per leaf, for all 16 games in rotation. Larger
inference batches cannot speed up a loop that is CPU-bound between batches.
Lockstep currently spends ~2× the GPU power to go the same speed; with
`N_PARALLEL_GAMES = 1` the loop would likely run at similar games/h with
milder GPU draw and 16× lower per-game latency (worth a one-hour benchmark
before the next run).

The useful thing lockstep proved: the GPU has enormous headroom. That is
what makes Lever 1 below safe.

## Revised plan: two independent, multiplying levers

### Lever 1 — process-level parallelism: player/learner split (revised Stage C)

Credit where due: this is Rob's "just run a second instance" idea, with one
correction — two full training instances would be two diverging models, each
at half compute. The split that makes extra instances feed **one** model:

- **Learner** (one process — essentially today's train_chess.py):
  plays, trains, owns the checkpoint, the buffer, and all logging/game
  numbering. Additionally polls a `shards/` directory each game-completion
  and ingests any finished-game files it finds (then deletes them).
- **Players** (1–3 processes): load the learner's latest checkpoint,
  self-play WITHOUT training, and write each finished game as a shard file —
  the committed (state, sparse_policy, outcome) list via `torch.save` to a
  temp name, then `os.replace` (atomic, same trick as the checkpoint saves).
  Reload the newest checkpoint every ~10 minutes. No logging of their own.

Why file-based handoff: no queues, no shared memory, no `multiprocessing` —
zero WSL2 deadlock surface, trivially debuggable (a shard is a file you can
open), and crash-isolated (a dead player loses one game). Self-play on
minutes-stale weights is standard AlphaZero practice and mildly good for
diversity.

Realistic ceiling on the 9600X (6C/12T): learner + 2–3 players ≈ **2.5–3×**
before memory bandwidth and thermals bite. Note the CPU already runs ~82°C
with one core loaded — case airflow becomes a real constraint for week-long
multi-player runs. GPU serves all processes comfortably (headroom proven
above).

### Lever 2 — cheaper simulations (less Python per sim; unchanged structure)

1. **Encoder bitboard vectorisation** — python-chess exposes each piece
   type's placement as a 64-bit int; `np.unpackbits` over the 8-byte view
   fills a plane without a Python loop. Runs once per leaf, every leaf —
   the largest single constant-factor win left (~1.5–2× overall). Must be
   verified bit-identical against the current encoder before deployment.
2. **Push/pop selection walk** — replace the per-edge `board.copy()` with
   one board per search that pushes moves during descent and pops after
   backup (keeping a small ring of recent positions for the 4-frame
   encoder). Cuts thousands of full board copies per move to ~zero.
3. Beyond these lies the language boundary: python-chess itself. C/Rust
   move generation is how LC0-class engines live, and it is deliberately
   **out of scope** — a codebase readable end-to-end is a goal of this
   project, not a nice-to-have.

### Combined expectation

Levers multiply: (2.5–3×) × (1.5–2×) ≈ **4–6×** → roughly 100–150 games/h,
a 10k-game run in ~2 days instead of ~2 weeks. All in Python.

## Sequencing (subordinate to run14_decision_protocol.md)

Nothing changes while run14 is between gates — game-generation speed and
provenance are part of the experiment. Build and desktop-verify the player
script and shard ingest during the Gate 2 → Gate 3 stretch; deploy only at
a gate boundary or on the next run. Compute is not currently the binding
constraint on HAL — training-signal quality is (run13_retune bought nothing
with 5,000 steps). This work pays off the moment the signal is proven fixed.
