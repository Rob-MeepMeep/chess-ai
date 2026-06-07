# Run 9 Architecture — Canonical Encoding
**Rob Kirkland, Ellis Ward**
*Draft — 2026-06-07*

---

## Overview

Run 9 is a single-fix run. Every infrastructure decision made for Run 8 — the tiered
replay buffer, the 25% permanent partition, the ±0.8 cap draw outcomes, the corrected
MCTS backup sign — was validated and holds. The one structural flaw that Run 8 exposed
but could not self-correct is the Geometry Trap: a mathematical consequence of the
colour-blind encoder that prevents the value head from cleanly separating perspective.

This document defines the problem, the fix, the implementation, and the expected outcome.

---

## 1. The Geometry Trap — Why It Happened

### Background

The colour indicator (plane 48) was removed in Run 6 after it was identified as the
cause of persistent Black-wins bias. It worked: the bias disappeared. But removing it
created a subtler structural problem that Run 8 made visible.

### The encoder's inputs

The 54-plane encoder represents each board state as a fixed-orientation 8×8 tensor.
Planes 0–5 always encode White pieces (pawn, knight, bishop, rook, queen, king).
Planes 6–11 always encode Black pieces. The board is always shown from White's
physical orientation — rank 1 at the bottom, rank 8 at the top.

There is no plane that tells the network whose turn it is.

### The mathematical consequence

Consider the canonical endgame position used in regression testing:

```
K+Q vs K — White has the queen, position is otherwise identical
```

When it is **White's turn**, the value head should output near **+1** (White winning).
When it is **Black's turn** in the same position, the value head should output near **−1**
(Black to move, facing certain loss).

These two inputs to the network are nearly identical: same piece positions, same plane
assignments. The only difference is an implicit one — who is about to move — which the
network must infer from context rather than read directly.

The gradient signals these two positions generate are contradictory:
- The White-to-move version pushes the network toward positive output
- The Black-to-move version pushes the network toward negative output

Both positions look almost the same to the network. It cannot resolve the conflict
cleanly. What actually happens in practice is that one signal dominates:

**The losing-side (b-move) signal is more common in training data.** When White has a
decisive material advantage, Black is almost always the one being forced to move in the
endgame — there are simply more positions in the buffer where the side with less material
is to move. The b-move gradient dominates, and b-move converges reliably (reaching −0.921
by game 2010). The w-wins gradient fights against it on similar inputs, causing the
oscillation observed across every regression test in Run 8.

This is the Geometry Trap: the network learns that a particular board geometry (king
in corner, queen nearby) signals a decisive endgame, but cannot reliably determine
**which player is winning** without a perspective anchor. It learns the shape of the
position without cleanly learning the perspective.

### What the regression data shows

| Steps | w wins | b move | Interpretation |
|-------|--------|--------|---------------|
| 7,260 | −0.838 | −0.785 | Both negative — network sees geometry, not perspective |
| 9,360 | −0.314 | −0.795 | b-move strengthening; w-wins oscillating toward zero |
| 10,160 | +0.048 | −0.921 | b-move near target; w-wins near zero — Geometry Trap confirmed |

The w-wins reading oscillates between −0.84 and +0.05 across the run. It is not learning;
it is being pulled in two directions by contradictory gradient signals on similar inputs.

---

## 2. The Canonical Fix

### The principle

AlphaZero solves this by always evaluating positions from the current player's perspective.
Before encoding, if it is Black's turn, the board is flipped 180° — so the network
always sees its own pieces on ranks 1–2, regardless of actual colour.

The network no longer has a concept of White or Black. It exclusively evaluates:
- **My pieces** — always on ranks 1–2, always encoded as White-piece planes
- **Opponent's pieces** — always on ranks 7–8, always encoded as Black-piece planes

The position "K+Q vs K, my turn" always produces the same tensor regardless of whether
"me" is White or Black. The gradient signals are no longer contradictory — they are the
same signal. The value head converges cleanly.

### The operation

`chess.Board.mirror()` in python-chess performs the canonical flip:
- Flips the board vertically (rank 1 ↔ rank 8, rank 2 ↔ rank 7, etc.)
- Swaps piece colours (White pieces become Black pieces and vice versa)
- Swaps the side to move (Black to move → White to move after mirror)
- Handles castling rights and en passant squares correctly

Applied consistently to the full position history before encoding, this produces a
canonical representation where the current player always occupies the White perspective.

---

## 3. Implementation

### chessai/encoder.py

Two lines of new logic at the top of `encode()`, before the existing 54-plane extraction:

```python
is_black_turn = (history[0].turn == chess.BLACK)

canonical_history = []
for board in history:
    canonical_history.append(board.mirror() if is_black_turn else board.copy())
```

The rest of the function operates on `canonical_history` instead of `history`.
No changes to the plane structure — all 54 planes remain as-is.

### chessai/agent.py or mcts.py — the decode side

This is the critical pairing. The network outputs a policy distribution over moves —
but those moves are now in canonical (always-White) coordinate space. If the board was
mirrored for encoding, the selected move must be mirrored back before it is played:

```python
if is_black_turn:
    chosen_move = chosen_move.mirror()
```

`chess.Move.mirror()` flips the source and destination squares vertically. Without
this step, HAL would play the canonical move on the real board — moving pieces to the
wrong squares.

### Seed buffer for Run 9

The Run 8 replay buffer cannot be reused directly. Every position in it was encoded with
the old (non-canonical) encoder. Reusing it would feed the new network contradictory
inputs during early training.

The correct approach: run `curate_buffer.py` against Run 8's `games.csv` with the new
encoder in place. This re-encodes the seed positions from Run 8's late decisive games
(games 1500–2010) using the canonical encoder, producing a clean seed buffer for Run 9.

Canonical endgame positions in the permanent partition also need re-encoding under the
same principle — though K+Q vs K is symmetric enough that the canonical flip has limited
effect on the ground-truth positions themselves.

---

## 4. The Stage 2 Resignation Hypothesis

### Current state (Run 8, end of run)

RESIGN_MATERIAL = 7 is active. When the losing side's material deficit exceeds 7 points,
the training loop terminates the game and assigns a decisive outcome. This is a hardcoded
heuristic that bypasses the value head for endgame termination.

The b-move regression reached −0.921 at game 2010 — within reach of the −0.95 threshold
but not yet there. The w-wins reading oscillates, confirming the value head cannot yet
take sole responsibility for resign decisions.

### Expected outcome with canonical encoding

With the canonical fix in place, the gradient signals for w-wins and b-move are unified.
Both positions ("K+Q vs K, my turn" — regardless of colour) produce the same canonical
tensor and should converge to the same value: near +1 for the player with the queen,
which means near −1 when viewed from the opponent's perspective via the MCTS backup
sign flip.

The regression table should look like this in a healthy Run 9:

| Position | Expected value |
|----------|---------------|
| K+Q vs K (current player has queen) | near +1 |
| K+Q vs K (current player is losing) | near −1 |
| Start position | ~0.0 |
| Current player missing queen | clearly negative |

Once both readings cross ±0.90 consistently, RESIGN_MATERIAL can be removed. The value
head takes sole responsibility for recognising losing positions. This is Stage 2 resign.

### Why this matters

RESIGN_MATERIAL is a crutch. It terminates games that the value head should be learning
to evaluate on its own. Removing it exposes the network to the full complexity of
endgame conversion — which forces the policy head to learn actual mating sequences rather
than relying on a hardcoded shortcut to end the game. The K+Q vs K regression measures
exactly this: can the value head recognise the position without external help?

---

## 5. Permanent Partition Outcome Labelling

### The problem under Run 8

The permanent partition holds ground-truth canonical endgame positions — K+Q vs K,
K+R vs K, and similar — with hardcoded outcome values. Under the old fixed-orientation
encoder, these outcomes were stored as absolute values: +1 meant White is winning,
−1 meant Black is winning.

This caused a subtle inconsistency. When a K+Q vs K position was sampled with Black to
move, the stored outcome (+1, White wins) and the network's expected output (−1, current
player is losing) were in conflict. The permanent partition was providing ground-truth
signal that contradicted what the network was trying to learn. This is one reason the
w-wins regression was harder to stabilise than b-move — the partition was helping b-move
and confusing w-wins.

### The fix under canonical encoding

With canonical encoding, all outcomes must be stored from the **current player's
perspective**:
- +1 means **I am winning** (regardless of colour)
- −1 means **I am losing** (regardless of colour)

When `curate_buffer.py` builds the permanent partition for Run 9, every position must
be encoded canonically (mirror if Black to move) and its outcome assigned from that
player's perspective. For K+Q vs K:

| Actual position | Canonical encoding | Stored outcome |
|----------------|-------------------|---------------|
| White has queen, White to move | No mirror needed | +1 (I'm winning) |
| White has queen, Black to move | Mirror applied | −1 (I'm losing) |
| Black has queen, Black to move | Mirror applied | +1 (I'm winning) |
| Black has queen, White to move | No mirror needed | −1 (I'm losing) |

All four cases produce consistent gradient signal toward the same target: the player
with the queen should see +1, the player without should see −1.

### Implementation note for curate_buffer.py

When seeding the permanent partition, the outcome must be assigned **after** the
canonical flip, not before. The flip changes who is "current player" — so the outcome
label must reflect the post-flip perspective:

```python
is_black_turn = (board.turn == chess.BLACK)
canonical_board = board.mirror() if is_black_turn else board.copy()
# outcome: +1 if current player (post-flip, always "White") is winning
outcome = +1.0 if current_player_is_winning else -1.0
```

Getting this wrong — storing absolute (White-wins) outcomes with canonical-encoded
inputs — would reintroduce the same gradient conflict that caused the Geometry Trap.

---

## 6. Other Run 9 Configuration

| Parameter | Run 8 | Run 9 |
|-----------|-------|-------|
| Encoder planes | 54 (colour-blind, fixed orientation) | 54 (colour-blind, canonical orientation) |
| Perspective | Fixed White-up | Always current player |
| Seed buffer | Run 7 decisive games (13,152 positions) | Run 8 late decisive games (re-encoded) |
| Weights | Fresh | Fresh |
| MCTS sims | 200 | 200 (unchanged) |
| Permanent partition | 25% per batch | 25% per batch (unchanged) |
| Cap draw outcomes | ±0.8 when material > 3 | ±0.8 unchanged |
| RESIGN_MATERIAL | 7 | 7 (remove once b-move ≥ ±0.90 sustained) |

Fresh weights are required. The Run 8 checkpoint learned under the old encoder — its
internal weight representations correspond to the fixed-orientation input space. Loading
those weights into a canonical-encoder network would produce garbage outputs.

---

## 6. What We Are Not Changing

- **Network architecture** — 10 residual blocks, 160 channels. Unchanged.
- **54-plane structure** — the planes themselves are identical. Only the orientation of
  the input changes.
- **MCTS algorithm** — backup sign, visit count, temperature. All unchanged.
- **Buffer structure** — tiered with permanent partition. Unchanged.
- **Training loop** — self-play, loss function, batch size. Unchanged.

Run 9 is a surgical fix. If the Geometry Trap is the root cause of the value head
instability (and the regression data strongly suggests it is), the canonical encoding
should resolve it cleanly. If it does not, the architecture itself needs revisiting.

---

*Architecture document — chess-ai Phase 3, Run 9. See `paper/phase3_architecture.md`
for full Run 8 context and `paper/run_notes.md` for training history.*

---

## Addendum — 2026-06-07: Canonical Encoding Was Already Implemented

When preparing to implement the canonical encoding fix for Run 9, we read `chessai/encoder.py`
and discovered that the canonical flip has been present since commit `05b3e53` — the same
commit that removed the colour plane (Run 6). It was never absent. Every run from Run 6
onwards, including Runs 7 and 8, encoded positions canonically.

The git history confirms this: encoder.py has had only three commits in the project's
lifetime. The canonical flip (`board.mirror() if player == chess.BLACK else board`) was
written at the same time the colour plane was removed and has not changed since.

### What this means for the sections above

Sections 1–6 of this document describe a fix for a problem that did not exist in the form
we diagnosed. The planned implementation (adding mirror logic to encoder.py) requires no
code change. The decode-side mirror in agent.py/mcts.py also requires no change — the
network has always operated on canonical inputs and returned canonical move indices.

**The Geometry Trap phenomenon is real.** The regression data is unambiguous: b-move
converges reliably (reaching −0.921 by game 2010) while w-wins oscillates across the
entire run (+0.05 to −0.84). The phenomenon exists. Our explanation of its cause was wrong.

### Revised diagnosis — training data distribution asymmetry

With canonical encoding already in place, the gradient signals for w-wins and b-move are
not contradictory in the way we described. Both positions encode correctly from the current
player's perspective. The value head should in principle converge symmetrically.

The actual cause of the asymmetry is **training data distribution**:

1. **RESIGN_MATERIAL ends games before the endgame plays out.** When White reaches a 7-point
   material advantage, the game terminates. The buffer receives positions from the final moves
   before resignation — predominantly Black-to-move losing positions, because the losing side
   keeps making moves right up to the threshold. White-to-move positions in clean K+Q vs K
   configurations are underrepresented because White rarely gets to *move* in those positions
   before the game is resigned.

2. **The permanent partition covers only one spatial configuration.** The K+Q vs K positions
   in the permanent partition place the queens and kings at fixed squares. The value head
   learns those specific configurations well but generalises imperfectly to the regression
   test FEN (different piece locations). The losing side generalises better because its
   positions appear naturally across many game endings with varied geometry.

3. **The asymmetry compounds over training.** As b-move strengthens (reaching −0.921), the
   gradient from losing-side positions grows more confident, further crowding out the
   weaker signal from winning-side positions.

### Actual Run 9 changes

The encoder requires no changes. The actual improvements for Run 9 are:

| Change | Rationale |
|--------|-----------|
| Fresh weights | Run 8 weights are still valid starting points but contain all of Run 8's biases — starting fresh gives Run 9 a clean slate to train correctly from the first gradient step |
| Seed buffer from Run 8 (games 1500–2010) | Late Run 8 games reflect better play quality — stronger signal than early noise |
| More diverse permanent partition | Add K+Q vs K in multiple queen/king configurations (not just one FEN per colour). More spatial variety forces generalisation rather than memorisation of specific positions |
| Reduce or remove RESIGN_MATERIAL sooner | Let games play deeper into endgames so the buffer contains more White-to-move winning positions. Consider dropping RESIGN_MATERIAL from 7 to 4 at game 500 once the value head shows b-move ≤ −0.5 |

The config table in Section 6 should be read as: the encoder row ("canonical orientation")
was already true for Run 8. Run 9's meaningful changes are the seed buffer, permanent
partition diversity, and RESIGN_MATERIAL schedule.

### Why we are keeping the original sections

The original diagnosis and planned fix are preserved above for three reasons:

1. The reasoning was internally consistent given what we believed about the encoder.
   The flaw was an empirical one (not reading the code before planning), not a logical one.
2. The Geometry Trap framing — the phenomenon, the regression evidence, the oscillation
   pattern — remains valid and useful for the paper.
3. The permanent partition outcome labelling discussion (Section 5) is still correct
   and still worth implementing carefully, even though the encoder was not the problem.

**Key lesson for the paper:** systematic verification of assumptions before designing fixes.
We diagnosed the Geometry Trap correctly from the regression data, then assumed an incorrect
cause without confirming it in the source code. Read the code first.
