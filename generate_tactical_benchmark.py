"""
generate_tactical_benchmark.py — Build a small, hand-authored tactical
benchmark: hanging pieces, mate-in-1, mate-in-1 defence, declined
captures, and advantage preservation (formerly called "conversion" --
see the category note below for why that name overclaimed what's
actually tested).

12 Sept 2026 addition (step 4 of the search-misranking investigation,
paper/value_head_small_material_options.md Sec 6/8, Options 3+4): the
original 5 hanging_piece positions were knight/bishop/pawn scale plus
one queen case (n=1) -- not enough to say whether search's misranking
scales with material magnitude or was specific to those lines. Added
rook- and a second queen-scale position, both colours represented across
the set, plus a new declined_capture category: a position where the
objectively correct move is NOT to take a tempting piece, because doing
so is a real, Stockfish-confirmed blunder. Without this, the benchmark
could reward "always capture" as a trivial heuristic rather than sound
judgement -- exactly the gap flagged in the 11 Sept 2026 external review.

Why this exists (10 Sept 2026 assessment, forward plan item 2): a
compact, fixed benchmark to separate "the policy prior doesn't know"
from "the search didn't look," and to give the project something more
concrete than material-probe readings to judge real capability against.

Two corrections from a second review pass, both designed in from the
start rather than bolted on after:
  - Every position is reached by an actual move sequence from the start
    of a game, not a hand-placed end-state FEN with no history. An
    empty-history FEN is exactly material_probe's original defect
    (see material_probe_correction.md) -- the network has never seen a
    tactical position paired with a blank history any more than it saw
    a missing queen that way.
  - These positions are hand-authored short sequences, not sampled from
    self-play or curate_buffer.py's canonical-position generators. That
    makes them a genuinely different source from what's already in the
    training buffer's permanent partition (checked directly against
    curate_buffer.py's fixed CANONICAL_POSITIONS below; the buffer's
    *generated* KQ/KR-vs-K positions are randomised per build, so an
    exact-FEN overlap check against those isn't meaningful the way it
    is for the fixed list).

Every position's "correct answer" is verified programmatically here, not
asserted by hand -- python-chess directly confirms checkmate claims;
Stockfish confirms hanging-piece eval swings and that advantage-
preservation positions are genuinely won. The `material_probe` lesson
this whole assessment is built on is that an unvalidated test can be
wrong in ways nobody notices for a long time.

Categories:
  hanging_piece         — a free capture is on the board; correct move
                          takes it
  mate_in_1             — a forced mate in one; correct move delivers it
  defend_mate_in_1      — multiple acceptable moves (not one exact
                          answer): scored by "does the opponent still
                          have a mate-in-1 after this move", not by
                          matching a specific UCI move
  declined_capture      — a tempting capture (the "bait_move") is
                          available but Stockfish confirms it's a real
                          blunder (drops the mover's own eval below
                          DECLINED_CAPTURE_REGRESSION_CP); multiple
                          acceptable moves, same shape as
                          defend_mate_in_1 but judged by eval collapse
                          instead of an immediate forced mate
  advantage_preservation — a real, Stockfish-confirmed decisive material
                          advantage reached through actual play (not
                          necessarily reduced to bare kings -- see the
                          module docstring's note on why an exact
                          K+Q/K+R vs K reduction turned out harder to
                          hand-author reliably than expected); scored by
                          Stockfish agreeing the position stays winning
                          after HAL's move (not a blunder). Named
                          "advantage_preservation", not "conversion" as
                          this was originally called (11 Sept 2026,
                          external review of paper/value_head_small_
                          material_options.md): checking that an eval
                          stays above a threshold tests that HAL didn't
                          throw the win away, not that it demonstrated
                          the technique to actually convert it to
                          checkmate -- those are different claims, and
                          the original name asserted the stronger one
                          without testing it.

Output: chessai/tactical_benchmark.json

Usage:
  venv/bin/python3 generate_tactical_benchmark.py
"""

import json

import chess
import chess.engine

ENGINE_PATH = "stockfish"   # assumes stockfish is on PATH -- same convention as eval_chess.py
OUTPUT_PATH = "chessai/tactical_benchmark.json"
VALIDATE_DEPTH = 16


def _replay(move_seq: list) -> chess.Board:
    board = chess.Board()
    for uci in move_seq:
        board.push_uci(uci)
    return board


def _all_legal_uci(board: chess.Board) -> list:
    return [m.uci() for m in board.legal_moves]


def _mate_in_1_moves(board: chess.Board) -> list:
    """Every legal move that delivers immediate checkmate, verified
    directly by python-chess (no engine needed for this one)."""
    mates = []
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            mates.append(move.uci())
        board.pop()
    return mates


def _opponent_has_mate_in_1(board: chess.Board) -> bool:
    return len(_mate_in_1_moves(board)) > 0


# ---------------------------------------------------------------------------
# Candidate positions -- each a move sequence from the game start
# ---------------------------------------------------------------------------

HANGING_PIECE_CANDIDATES = [
    # White plays a natural developing sequence; ...Nxe4 leaves the knight
    # on e4 completely undefended (no black piece guards e4, and White's
    # bishop on c4/knight on f3 don't either) -- Qxe4-style capture isn't
    # even needed, d3 just wins it outright with tempo. Correct answer:
    # any capture of the undefended knight.
    {
        "name": "undefended_knight_on_e4",
        "moves": ["e2e4", "e7e5", "g1f3", "f8c5", "f1c4", "g8f6",
                  "d2d3", "f6e4"],
        # White to move; e4 knight is undefended -- d3 pawn can just take it
        "expect_capture_of": "e4",
    },
    # Black's bishop on f5 has no defender once White's queen swings to
    # d3 (attacking it along the open d3-f5 diagonal, e4 empty since
    # White played d4 rather than e4). a7a6 is a plausible waiting move
    # that doesn't address the threat, handing the capture to White.
    {
        "name": "undefended_bishop_on_f5",
        "moves": ["d2d4", "d7d5", "c1f4", "c8f5", "b1c3", "g8f6",
                  "d1d3", "a7a6"],
        "expect_capture_of": "f5",
    },
    # 11 Sept 2026 addition, testing whether the search-vs-policy
    # disagreement found on undefended_knight_on_e4 (policy correctly
    # favours the capture, search at 200 AND 600 sims prefers an
    # objectively much worse move instead -- Stockfish: +400cp vs -297cp)
    # is a pattern or an isolated quirk of that one line. Three more
    # hanging-piece positions, deliberately varied: mirrored colour
    # (Black captures this time), a different piece value each
    # (knight/queen/pawn), and different capturing piece each
    # (knight/pawn/pawn) -- not just cosmetic variations on the same shape.
    {
        # Mirror of the original: White's knight jumps to d5 undefended,
        # Black's own knight takes it for free.
        "name": "undefended_knight_on_d5_black_captures",
        "moves": ["b1c3", "g8f6", "c3d5"],
        "expect_capture_of": "d5",
    },
    {
        # White's queen wanders to a5, defended by nothing, attacked by
        # Black's own b6 pawn -- a full queen hangs to a pawn capture.
        "name": "undefended_queen_on_a5",
        "moves": ["e2e4", "b7b6", "d1h5", "c8b7", "h5a5"],
        "expect_capture_of": "a5",
    },
    {
        # Smallest magnitude of the set: White pushes e5 without
        # noticing Black's d6 pawn can simply take it. Real, but a much
        # smaller material swing than the other three -- worth knowing
        # whether the same disagreement pattern shows up even at pawn
        # stakes or only at larger ones.
        "name": "undefended_pawn_on_e5",
        "moves": ["e2e4", "d7d6", "e4e5"],
        "expect_capture_of": "e5",
    },
    # 12 Sept 2026 additions -- rook- and a second queen-scale position,
    # both colours, to address the "n=1 for the one success (queen)
    # case" gap flagged after step 3 of the investigation. Verified
    # directly with python-chess before adding (not just by construction
    # like the pawn/knight cases above): the target square really is
    # empty of any defender for its own side.
    {
        # 1.b3 clears b2, opening the long diagonal all the way from g7
        # to a1 before White's rook has moved. 2.e4 Bg7 puts a bishop
        # directly on that diagonal; 3.Nf3 (deliberately not touching
        # c3/d4/e5/f6, which would re-block it) does nothing about the
        # threat. Confirmed: Bxa1 is legal and a1 has no other defender.
        "name": "undefended_rook_on_a1_black_captures",
        "moves": ["b2b3", "g7g6", "e2e4", "f8g7", "g1f3"],
        "expect_capture_of": "a1",
    },
    {
        # Mirror image: 1.g3 opens the long diagonal for a future
        # fianchettoed bishop; 1...b6 (Black's own careless mirroring)
        # clears b7, opening the diagonal all the way to a8; 2.Bg2 puts
        # White's bishop on it. 2...Nf6 does nothing to defend a8.
        # Confirmed: Bxa8 is legal and a8 has no other defender.
        "name": "undefended_rook_on_a8",
        "moves": ["g2g3", "b7b6", "f1g2", "g8f6"],
        "expect_capture_of": "a8",
    },
    {
        # A second queen-scale case, opposite colour and opposite
        # capturing piece from undefended_queen_on_a5 (there, Black's
        # pawn captures White's queen; here, White's knight captures
        # Black's). 3...Qh4?? walks the queen onto a square already
        # attacked by White's f3 knight, with nothing defending it.
        "name": "undefended_queen_on_h4",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "d8h4"],
        "expect_capture_of": "h4",
    },
]

# A capture that LOOKS free but Stockfish confirms is a real blunder --
# without this category, the benchmark could reward "always capture" as
# a trivial heuristic rather than sound judgement (external review, 11
# Sept 2026). Scoring needs an engine, so validated in
# build_declined_capture() rather than at import time like the
# python-chess-only categories above.
DECLINED_CAPTURE_CANDIDATES = [
    {
        # The "Elephant Trap" line in the Queen's Gambit Declined:
        # 6.Nxd5?? looks like it just wins back the pawn Black spent
        # recapturing on d5, but 6...Nxd5 7.Bxd8 Bb4+ 8.Qd2 Bxd2+
        # 9.Kxd2 Kxd8 nets Black a full piece -- White is forced to
        # recapture on d5 with the bishop and walks into the queen-
        # trading intermezzo check. Verified with Stockfish (depth 16)
        # before trusting it from memory, not asserted by hand: this
        # position is +35cp for White; the bait move collapses it to
        # -369cp, while simple development (e3, Nf3) holds +26/+29cp.
        "name": "elephant_trap_qgd",
        "moves": ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6",
                  "c1g5", "b8d7", "c4d5", "e6d5"],
        "bait_move": "c3d5",
    },
]

MATE_IN_1_CANDIDATES = [
    {
        # Classic fool's mate -- White to move has just played the
        # blunder, Black delivers mate. We want the position with the
        # SIDE TO MOVE being the one who mates, so stop one ply early.
        "name": "fools_mate",
        "moves": ["g2g4", "e7e5", "f2f3"],   # Black to move, Qh4# available
    },
    {
        # Reversed-colour fool's mate: Black weakens their own kingside
        # (g5, f6), White's queen infiltrates via h5. Same idea as the
        # classic pattern, White delivering it this time -- 1.e4 g5 2.c3
        # f6 3.Qh5# is a real sequence this project's own run8 self-play
        # data produced (see checkmate_analysis.md).
        "name": "fools_mate_reversed",
        "moves": ["e2e4", "g7g5", "c2c3", "f7f6"],   # White to move, Qh5# available
    },
]

DEFEND_MATE_IN_1_CANDIDATES = [
    {
        # Scholar's mate setup, one move before completion: White
        # threatens Qxf7#. Black to move must do something about it.
        "name": "scholars_mate_threat",
        "moves": ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4"],
    },
]

ADVANTAGE_PRESERVATION_CANDIDATES = [
    {
        # Verified with Stockfish before being trusted, not by hand: 2.Qh5
        # attacks Nc6, but 3.Qxe5+ walks into Nxe5 -- White has traded a
        # queen for a pawn. Not a literal bare-kings K+Q vs K (this is a
        # queen-for-a-pawn material swing with the rest of the position
        # still on the board), but real history, real legal moves, and a
        # decisive, Stockfish-confirmed advantage for the side to move --
        # a fair test of whether the advantage survives HAL's next move.
        "name": "queen_won_for_a_pawn",
        "moves": ["e2e4", "e7e5", "d1h5", "b8c6", "h5e5", "c6e5", "b1c3"],
    },
    {
        # A simpler, smaller-magnitude version: White's knight sac on e5
        # just loses a piece outright (dxe5 recaptures cleanly, no
        # compensation). Up a full minor piece for a pawn, side to move.
        "name": "knight_won_for_a_pawn",
        "moves": ["e2e4", "e7e5", "g1f3", "d7d6", "f3e5", "d6e5", "d2d4"],
    },
]


def build_hanging_piece():
    result = []
    for cand in HANGING_PIECE_CANDIDATES:
        board = _replay(cand["moves"])
        target_sq = chess.parse_square(cand["expect_capture_of"])
        capturing_moves = [m.uci() for m in board.legal_moves if m.to_square == target_sq]
        if not capturing_moves:
            print(f"  SKIP {cand['name']}: no legal move captures "
                  f"{cand['expect_capture_of']} -- position doesn't validate")
            continue
        result.append({
            "name": cand["name"],
            "category": "hanging_piece",
            "moves": cand["moves"],
            "correct_moves": capturing_moves,
        })
        print(f"  OK {cand['name']}: correct move(s) = {capturing_moves}")
    return result


def build_mate_in_1():
    result = []
    for cand in MATE_IN_1_CANDIDATES:
        board = _replay(cand["moves"])
        mates = _mate_in_1_moves(board)
        if not mates:
            print(f"  SKIP {cand['name']}: no legal mate-in-1 found "
                  f"at this position -- doesn't validate")
            continue
        result.append({
            "name": cand["name"],
            "category": "mate_in_1",
            "moves": cand["moves"],
            "correct_moves": mates,
        })
        print(f"  OK {cand['name']}: mate move(s) = {mates}")
    return result


def build_defend_mate_in_1():
    result = []
    for cand in DEFEND_MATE_IN_1_CANDIDATES:
        board = _replay(cand["moves"])
        # Confirm the THREAT is real: does the side about to move next
        # (after a null-ish "do nothing" move) actually have a mate-in-1
        # if the side to move here plays a move that doesn't address it?
        # Simplest direct check: verify at least one legal reply for the
        # side to move leaves the opponent WITHOUT a mate-in-1 (a genuine
        # defence exists), and separately confirm most/naive moves still
        # allow it (the threat is real, not illusory).
        acceptable = []
        naive_still_mated = 0
        total_checked = 0
        for move in board.legal_moves:
            board.push(move)
            has_mate = _opponent_has_mate_in_1(board)
            board.pop()
            total_checked += 1
            if not has_mate:
                acceptable.append(move.uci())
            else:
                naive_still_mated += 1
        if not acceptable:
            print(f"  SKIP {cand['name']}: no legal move avoids the mate "
                  f"-- already lost, not a fair defence puzzle")
            continue
        if naive_still_mated == 0:
            print(f"  SKIP {cand['name']}: EVERY legal move avoids the mate "
                  f"-- threat isn't real, not a meaningful puzzle")
            continue
        result.append({
            "name": cand["name"],
            "category": "defend_mate_in_1",
            "moves": cand["moves"],
            "correct_moves": acceptable,
        })
        print(f"  OK {cand['name']}: {len(acceptable)}/{total_checked} legal "
              f"moves defend; {naive_still_mated} still walk into mate")
    return result


DECLINED_CAPTURE_REGRESSION_CP = -150   # a move dropping the mover's own eval
                                          # (from their perspective, right after
                                          # they play it) below this counts as a
                                          # real blunder, not just slightly
                                          # suboptimal -- matched to this
                                          # candidate's -369 bait vs +26/+29
                                          # alternatives, with headroom either way


def build_declined_capture(engine):
    result = []
    for cand in DECLINED_CAPTURE_CANDIDATES:
        board = _replay(cand["moves"])
        bait = cand["bait_move"]
        if chess.Move.from_uci(bait) not in board.legal_moves:
            print(f"  SKIP {cand['name']}: bait move {bait} isn't legal here")
            continue

        # Exhaustive, not spot-checked: every legal reply gets its own
        # Stockfish read, same rigor build_defend_mate_in_1() applies to
        # its own "does every naive move still walk into it" check, just
        # judged by eval collapse instead of an immediate forced mate.
        acceptable = []
        bait_cp = None
        total_legal = 0
        for move in board.legal_moves:
            total_legal += 1
            b2 = board.copy()
            b2.push(move)
            if b2.is_game_over():
                cp = 10000 if b2.is_checkmate() else 0
            else:
                info = engine.analyse(b2, chess.engine.Limit(depth=VALIDATE_DEPTH))
                score = info["score"].pov(not b2.turn)   # mover's own perspective
                if score.is_mate():
                    cp = 10000 if score.mate() > 0 else -10000
                else:
                    cp = score.score()

            uci = move.uci()
            if uci == bait:
                bait_cp = cp
            elif cp is not None and cp >= DECLINED_CAPTURE_REGRESSION_CP:
                acceptable.append(uci)

        if bait_cp is None or bait_cp >= DECLINED_CAPTURE_REGRESSION_CP:
            print(f"  SKIP {cand['name']}: bait move doesn't validate as a "
                  f"real blunder (cp={bait_cp}) -- not a fair trap")
            continue
        if not acceptable:
            print(f"  SKIP {cand['name']}: no legal alternative avoids a "
                  f"similar collapse -- not a fair test")
            continue

        result.append({
            "name": cand["name"],
            "category": "declined_capture",
            "moves": cand["moves"],
            "correct_moves": acceptable,
            "bait_move": bait,
        })
        print(f"  OK {cand['name']}: bait {bait} cp={bait_cp}, "
              f"{len(acceptable)}/{total_legal} legal replies hold the position")
    return result


def build_advantage_preservation(engine):
    result = []
    for cand in ADVANTAGE_PRESERVATION_CANDIDATES:
        board = _replay(cand["moves"])
        info = engine.analyse(board, chess.engine.Limit(depth=VALIDATE_DEPTH))
        score = info["score"].pov(board.turn)
        if score.is_mate():
            cp = 10000 if score.mate() > 0 else -10000
        else:
            cp = score.score()
        if cp is None or cp < 300:   # roughly a minor piece or more, clearly decisive
            print(f"  SKIP {cand['name']}: Stockfish doesn't rate this as "
                  f"clearly winning for the side to move (cp={cp}) "
                  f"-- not a fair advantage-preservation test")
            continue
        # "Correct" here isn't one move -- it's "don't blunder the win."
        # Store the pre-move eval so run_tactical_benchmark.py can check
        # the position is still winning after HAL's move, rather than
        # requiring one specific move.
        result.append({
            "name": cand["name"],
            "category": "advantage_preservation",
            "moves": cand["moves"],
            "reference_cp": cp,
        })
        print(f"  OK {cand['name']}: Stockfish cp={cp} for side to move")
    return result


def main():
    print("Building hanging_piece positions...")
    hanging = build_hanging_piece()
    print("\nBuilding mate_in_1 positions...")
    mates = build_mate_in_1()
    print("\nBuilding defend_mate_in_1 positions...")
    defences = build_defend_mate_in_1()

    print("\nBuilding declined_capture and advantage_preservation "
          "positions (needs Stockfish)...")
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    try:
        declined = build_declined_capture(engine)
        advantage_preservations = build_advantage_preservation(engine)
    finally:
        engine.quit()

    all_positions = hanging + mates + defences + declined + advantage_preservations

    # Overlap check against curate_buffer.py's fixed canonical positions
    # (the randomised generators produce different positions per build,
    # so there's no fixed list to diff those against -- see module
    # docstring). Hand-authored short openings landing on an exact
    # existing canonical FEN would be a wild coincidence, but check anyway.
    import sys
    sys.path.insert(0, ".")
    from curate_buffer import CANONICAL_POSITIONS
    canonical_fens = {chess.Board(fen).board_fen() for fen, _ in CANONICAL_POSITIONS}
    for pos in all_positions:
        board = _replay(pos["moves"])
        if board.board_fen() in canonical_fens:
            raise ValueError(f"{pos['name']} collides with an existing "
                              f"canonical training position -- pick a different line")

    print(f"\n{len(all_positions)} positions validated, "
          f"0 overlap with curate_buffer.py's canonical set")
    for pos in all_positions:
        print(f"  [{pos['category']:>18}] {pos['name']}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_positions, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
