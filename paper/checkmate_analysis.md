# Checkmate Analysis — Run 8
**Rob Kirkland, Ellis Ward**

---

## Overview

This document records every game in Run 8 that ended in checkmate, with annotations on notable patterns and exceptional games. It serves as a record of HAL-4000's emerging tactical understanding across the first ~860 games of training.

**Total checkmates (games 1–860): 150**
- Black delivers checkmate: 85
- White delivers checkmate: 65

For context: Run 6 (colour plane included, 5,000 games) produced its first checkmate at game 658. Run 7 (buffer seeding, but inverted MCTS backup) averaged roughly 8 checkmates per 50-game window. Run 8, with the MCTS backup sign fix, produced 8 checkmates in the very first 50 games — and game 3 was already a checkmate.

---

## Checkmate Frequency by Training Window

| Window | Checkmates | Notes |
|--------|-----------|-------|
| 1–50   | 8  | First checkmate at game 3 |
| 51–100 | 9  | |
| 101–150 | 13 | |
| 151–200 | 11 | |
| 201–250 | 7  | |
| 251–300 | 10 | |
| 301–350 | 2  | Temporary dip |
| 351–400 | 9  | Recovery |
| 401–450 | 6  | |
| 451–500 | 13 | |
| 501–550 | 6  | |
| 551–600 | 8  | |
| 601–650 | 7  | |
| 651–700 | 4  | Post-buffer-transition dip |
| 701–750 | 5  | |
| 751–800 | 4  | |
| 801–850 | 13 | Notable spike — equals highest window |

The dip at games 651–750 coincides with the seed buffer being fully displaced (~game 670) and the network entering pure self-play. The spike back to 13 at window 851–850 suggests the value head is beginning to stabilise again after the transition.

---

## The Exceptional Games

### Game 823 — Fool's Mate (4 moves) ★★★
**HAL plays Black. Delivers the fastest legal checkmate in chess.**

```
1. g4   e5
2. f3   Qh4#
```

White pushes g4 (weakening h4) then f3 (removing the last defender of h4). Black queen snaps to h4 — checkmate. There is no faster checkmate in chess. This is the Fool's Mate, and it requires white to make specifically the two worst kingside pawn moves available.

HAL recognised and punished this in two moves.

---

### Game 830 — Scholar's Mate variant (5 moves) ★★★
**HAL plays White. Queen to h5 checkmate.**

```
1. e4   g5
2. c3   f6
3. Qh5#
```

Black weakens g5 then plays f6, exposing the f7 square and creating a diagonal threat on the kingside. HAL brings the queen to h5 — checkmate. This mirrors the Scholar's Mate pattern (Qh5# when the king's diagonal defence has been dismantled).

The pairing with game 823 is striking: HAL has learned both sides of the same tactical idea. As black it exploits f3+g4 with Qh4. As white it exploits g5+f6 with Qh5.

---

### Games 326 & 782 — Queen to g3 (6 moves) ★★
**HAL plays Black. Queen sweeps to g3 checkmate.**

Game 326:
```
1. f3   d5
2. b3   Qd6
3. h4   Qg3#
```

Game 782:
```
1. b4   d5
2. f3   Qd6
3. h3   Qg3#
```

Both games follow the same structure: white plays f3 (weakening g3), then makes a further pawn move on the queenside or rook file. HAL advances the queen to d6 on move 2, threatening g3, and delivers checkmate on move 3. The repetition of this pattern across two separate games suggests HAL has genuinely learned the idea, not stumbled on it once.

---

### Game 767 — Bishop to h4 checkmate (8 moves) ★★
**HAL plays Black. Unusual — checkmate delivered by a bishop.**

```
1. b3   Nh6
2. g4   e5
3. f3   Be7
4. a3   Bxh4#
```

White creates the same kingside weakness (g4, f3) that appears in many short games, but this time HAL uses the bishop rather than the queen. The bishop slides from e7 to h4, delivering check along the h4–e1 diagonal through the hole left by f3. A less common pattern that shows HAL isn't simply repeating a single queen manoeuvre.

---

### Games 289, 699 — Qh4 with setup (8 moves) ★

**Game 289:** `Na3 e5 g4 Nh6 b3 a6 f3 Qh4#`
**Game 699:** `f4 h6 g4 b6 a3 e5 fxe5 Qh4#`

Both games end with Qh4#, the same pattern as Fool's Mate but requiring more setup moves because white doesn't immediately open both kingside weaknesses. In game 699, HAL waits for white to capture on e5 (fxe5), which removes the last f-pawn defender before playing Qh4.

---

### Games 111, 443 — Qh4 from distance (12 moves) ★

These games show HAL sustaining the Qh4 plan across more moves of play, waiting for the right moment rather than rushing. In game 443, a knight sacrifice on e5 clears a pawn before Qh4# lands.

---

### Games 112, 13 — HAL as White finds Qh5 (13 and 21 moves) ★

**Game 112** (13 moves): HAL plays `d4 b6 Kd2 Na6 e4 Rb8 h4 f6 Bd3 Bb7 Bf1 g5 Qh5#` — a longer build-up to the same Qh5# finish seen in game 830, this time against more developed black pieces.

**Game 13** (21 moves): White lures the black king forward across the board (it walks to f5, e4) and checkmates it with the queen on c2. The king was hunted rather than caught in an opening trap — a qualitatively different kind of checkmate.

---

## Longer Games of Note

### Game 763 (108 moves) — First queen promotion leading to checkmate
Game ends with `...g2g1q` (pawn promotes to queen) followed by checkmate shortly after. HAL queens a pawn and converts the resulting material advantage. A significant milestone — demonstrates the full conversion chain: pawn structure → promotion → checkmate.

### Game 805 (30 moves) ★
**HAL plays Black.** Black king walks forward to c5 as part of an aggressive plan, and the game ends with HAL delivering `Qd4#` after a complex middlegame. The king's active role makes this one of the more unusual wins.

### Game 761 (25 moves) — Bishop and queen coordination
**HAL plays White.** `Qg6#` delivered after a setup involving bishop to d6 and pawn advances on g and f files. Shows HAL beginning to coordinate two pieces toward a mating attack rather than relying on a single queen.

### Game 768 (135 moves) — Longest checkmate game
A 135-move game ending in checkmate — the longest in Run 8 so far. HAL sustains winning pressure across a very long ending, eventually converting. This kind of game is hard to win cleanly without a strong value head, which suggests the network has developed enough evaluation depth to maintain its advantage without blundering it away over many moves.

---

## Patterns and Observations

### 1. Dominant pattern: Qh4 / Qh5 exploitation
The single most common short checkmate pattern is the queen sweeping to h4 or h5 when white creates f3+g4 or g5+f6 weaknesses. HAL has learned this reliably from both colours. This is not a coincidence — the self-play loop means white is also HAL, and HAL as white sometimes makes exactly these pawn moves, which HAL as black has learned to punish immediately.

### 2. Mirror learning
Games 823 and 830 represent the same tactical idea from opposite sides of the board. HAL delivers Fool's Mate as black and Scholar's Mate as white within 7 games of each other. This suggests the underlying pattern (queen to the opposite corner when the king's pawn defence is dismantled) has been abstracted to some degree.

### 3. Unusual pieces delivering mate
Game 767 features a bishop checkmate on h4. Several longer games end with rook or queen mates after piece coordination. The network is not locked into a single mating pattern.

### 4. King-walking punishments
Several longer games (13, 257, 805) involve HAL trapping an exposed or actively walking king in the middlegame. These are different in character from the short queen-to-h4 traps — they require sustained tactical pressure across 20+ moves.

### 5. Black wins more checkmates (85 vs 65)
Black delivers checkmate more often than white in Run 8. This is partly a self-play artefact: HAL as white sometimes plays the f3+g4 pawn structure that loses immediately to Qh4#, while HAL as black has learned to exploit it. As training continues, this asymmetry should reduce as white learns to avoid those pawn moves.

---

## What This Tells Us About Learning Progress

At game 560 (full eval), HAL's win rate vs random was 4% — first wins in project history. Most games still ended in cap draws, meaning HAL could get a winning position but not convert.

The checkmate data gives more texture to that picture:
- HAL can deliver tactical checkmates reliably in positions it recognises (queen to h4/h5 openings)
- HAL is beginning to convert material advantages into checkmate in longer games
- HAL has learned both sides of the same tactical patterns, not just one
- The progression from game 3 (checkmate by game 3 of training) to game 823 (Fool's Mate at 823) shows continuous tactical awareness across the run

The main gap remaining is converting a won endgame (queen vs king) into checkmate reliably — the K+Q vs K regression still shows the value head knows it's winning but the policy head doesn't consistently execute the mating sequence. That's the next convergence milestone.

---

---

## Window 1051–1100 — 15 Checkmates

The highest-count window since game 150. Notable shift: the shortest checkmate here is 11 moves. The 4 and 5-move Scholar's/Fool's Mate traps that appeared in early windows have disappeared — HAL as its own opponent has learned not to create those weaknesses. Checkmates are now earned through play rather than exploited from opening blunders.

**W/B split:** 6 White, 9 Black.

---

### Game 1094 — White wins, 11 moves ★★
**Qh5# — black's own pieces seal the king**

```
1. f4   g5
2. h4   gxf4
3. a3   h6
4. b3   b6
5. e4   f6
6. Qh5#
```

The Qh5# pattern seen in games 830 and 112, but with a different dynamic. After black plays f6 on move 5, the king on e8 has every escape route blocked by its own pieces: d7 covered by own pawn, d8 by own queen, f8 by own bishop — and f7 attacked by the queen along the h5–e8 diagonal. HAL placed nothing on those squares. Black built the cage itself.

---

### Game 1081 — Black wins, 24 moves ★★
**White king marches to f4, mated by a pawn**

```
1. h4   d5    2. e4   a5    3. g4   f6
4. c3   c5    5. f3   Nc6   6. d4   Bd7
7. Kd2  g5    8. a3   Nh6   9. Ke3  Rg8
10. Be2  gxh4  11. Kf4  cxd4  12. Nh3  e5#
```

White walked the king forward — d2, e3, f4 — across three consecutive moves while HAL developed quietly. On move 12, pawn to e5 delivers checkmate. The king on f4, hemmed in by its own pawns on e4 and g4, is mated by a single pawn push. HAL waited, developed, then closed the trap with the simplest possible move.

---

### Games 1068 and 1086 — Black wins, 142 and 134 moves ★
**Sustained endgame conversion**

The two longest checkmate games in Run 8 so far. Neither ends in a tactical trap — both are full games where HAL maintained winning pressure across a long endgame and converted to checkmate rather than triggering a material resign. At 142 moves, game 1068 is the new Run 8 record.

This is qualitatively different from the early-window checkmates. HAL is no longer just finding short tactical patterns; it is playing long games to a decisive conclusion.

---

### Remaining games (1054, 1058, 1069, 1078, 1083, 1087, 1088, 1092, 1096, 1097, 1099)

Mix of 40–90 move games, no single dominant pattern. All involve complex middlegame or endgame play without recognisable opening traps. White and black both finding checkmates across a range of game lengths — consistent with both sides of self-play developing more robust play.

---

## Evolution of Checkmate Patterns — Run 8

| Phase | Shortest mate | Dominant pattern |
|-------|--------------|-----------------|
| Games 1–500 | 4 moves (Fool's Mate, game 823) | Queen to h4/h5 exploiting f3+g4 or g5+f6 |
| Games 501–860 | 6 moves | Short queen traps still present; longer conversions emerging |
| Games 861–1100 | 11 moves | Opening traps gone; checkmates earned through sustained play |

The disappearance of sub-10-move checkmates by game 1100 is a meaningful signal. Both sides have learned the patterns well enough that neither walks into an immediate queen trap. The tactical floor has risen.

---

---

## Games 1101–1440 — 54 Checkmates

**W/B split: 30 White, 24 Black** — first period in the run where white wins more checkmates than black. The early Qh4/Qh5 queen trap patterns that favoured Black have fully disappeared; checkmates are now distributed more evenly.

Shortest checkmate: 14 moves (game 1188). The floor continues to rise — sub-10-move mates are gone and sub-15-move mates are rare.

---

### Game 1188 — Black wins, 14 moves ★
**HAL plays Black. Queen sweep to g3.**

```
1. b4   h5    2. Nh3  e5
3. a4   c5    4. Nxg5 Qxg5
5. f3   Na6   6. h4   Rh6
7. Rg1  Qg3#
```

White plays Nh3-Ng5 (knight to g5), black takes with the queen. White plays h4 — opening the g-file for the rook, but also giving the black queen a clear path. HAL swings the rook to h6, then queen to g3 checkmate. The g3 square was undefended because f3 removed the pawn and no white piece covered it.

---

### Game 1152 — White wins, 15 moves ★
**HAL plays White. Queen to f5 checkmate.**

```
1. f3   h5    2. h3   a6
3. f4   c5    4. e4   f5
5. d4   a5    6. Be3  Kf7
7. Qh5  Kf6   8. Qf5#
```

Black walks the king forward to f6 — perhaps evaluating this as active defence. HAL's queen reaches h5 with check, black steps to f6, and Qf5 is checkmate. The king had walked into the centre and ran out of space. HAL lured it forward rather than chasing it.

---

### Game 1149 — Black wins, 22 moves ★
**HAL plays Black. Queen to f2 checkmate.**

```
1. e4   Nf6   2. Na3  a6
3. e5   e6    4. g4   Ke7
5. Nh3  b5    6. Ng1  c6
7. b3   Ne4   8. h4   Qb6
9. d3   b4    10. d3xe4? b4xa3
11. Bf4  Qf2#
```

White's knights go to the rim (Na3, Nh3→Ng1) and stall. Black advances queenside pawns, develops the knight to e4 (central), and after white captures on e4, the a3-pawn falls. White plays Bf4 — and HAL delivers Qf2#. The f2 square, already weakened by e4, is undefended. Queen to f2 is checkmate.

---

### Broader observations: games 1101–1440

**The tactical vocabulary is expanding.** Where early windows showed almost exclusively queen-to-h4/h5/g3 patterns, this window shows:
- Qg3# (game 1188)
- Qf5# (game 1152)
- Qf2# (game 1149)
- A range of longer checkmates (49, 51, 60, 62, 63 moves) across both colours

**No king-walking exploits from white**, unlike the early windows. The opponent (also HAL) has stopped advancing the king prematurely. Checkmates now require either luring the king or applying sustained piece pressure.

**White wins more checkmates (30 vs 24)** — a reversal of the early-run pattern. As the Qh4 opening traps disappeared, the slight first-mover advantage for white in generating winning plans is showing up.

---

---

## Games 1441–1910 — Gemini-Identified Highlights

An independent assessment of games 1500–1900 (401 games in PGN format) was run through
Gemini to get an external perspective on play quality for the Phase 3 paper. Full response
is in `paper/gemini_run8_assessment.md`. Two games were flagged as paper-worthy.

---

### Game 1509 — Black wins, 46 moves ★★
**Bg7# — bishop checkmate after king march to e5**

```
...
21. Nc3    f5
22. Ke5    a5
23. Nce2   Bg7#
```

White's king walks to e5 by move 22 — deep into the centre. HAL coordinates quietly,
then closes with Bg7 checkmate. The bishop on g7 covers f6 and h6; the king has no square.
A clean example of the king-hunting pattern that defines late Run 8: not a tactical trap
sprung in the opening, but a systematic net drawn around an exposed king across the middlegame.

---

### Game 1886 — Black wins, 34 moves ★★★
**Na5# — knight checkmate after bishop check forces king to c4**

```
...
15. Be2    Be4+
16. Kc4    Nc6
17. Bd3    Na5#
```

Black drives White's king to c4 with a bishop check on e4, then delivers checkmate with
the knight on a5. The king on c4, hemmed in by its own bishop on d3 and the pawn structure,
has no escape. Knight checkmates require precise coordination — the knight must reach a
square from which it both delivers check and covers every king flight square. HAL calculated
this three-move sequence precisely.

Gemini's assessment: *"This sequence demonstrates genuine tactical awareness that any human
club player would praise."*

---

### Broader observations: games 1441–1910

Gemini identified two dominant patterns across the late window:

**F-pawn fixation** — HAL overuses f3/f4 (White) and f6/f5 (Black) for early centre
influence, a legacy of the Qh4/Qh5 patterns it learned to exploit and deliver in the
early run. Both sides now know not to walk into an immediate queen trap, but the f-pawn
instinct remains, occasionally creating self-inflicted diagonal weaknesses.

**King-exposure hunting** — Once a king is driven into the open, HAL shows predatory
instinct: continuous queen checks, diagonal pressure, piece coordination to strip escape
squares. Games 1509 and 1886 are the clearest examples, but the pattern appears across
many late-window games.

**The geometric legacy** — Gemini independently noted that HAL compensates for the
colour-blind encoder by preferring forcing tactical lines where board geometry is
unambiguous. Unable to rely on stable perspective evaluation, the network defaults to
positions where concrete calculation overrides positional ambiguity. This is the adaptive
response to the Geometry Trap identified in our regression testing.

---

## Full Run 8 Checkmate Summary (games 1–1910, partial)

**Total checkmates through game 1440: ~231** (full count to be updated after game 2000 eval)
- Black delivers: ~119
- White delivers: ~112

| Phase | Games | Checkmates | Rate | Shortest | Dominant pattern |
|-------|-------|-----------|------|---------|-----------------|
| Early | 1–500 | ~88 | 18% | 4 moves | Queen to h4/h5/g3 opening traps |
| Mid | 501–1100 | ~89 | 16% | 6 moves | Traps fading; longer conversions emerging |
| Late | 1101–1440 | ~54 | 16% | 14 moves | No opening traps; earned through play |
| Final | 1441–1910 | TBC | — | — | King hunting; multi-piece coordination |

The transition from opening-trap checkmates to sustained-play checkmates is the clearest
evidence of both sides improving. The knight and bishop checkmates in the final window
(Na5#, Bg7#) represent the most sophisticated tactical coordination in the run.

---

*Data source: logs/run8/games.csv. External analysis: Gemini assessment of games 1500–1900,
full response in `paper/gemini_run8_assessment.md`.*
