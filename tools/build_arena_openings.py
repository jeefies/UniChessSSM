#!/usr/bin/env python3
"""build_arena_openings.py - Arena 200+ Balanced Openings Construction & Loop-Bias Elimination.

Constructs 200 diverse, high-quality chess opening move sequences in SAN format
(6 to 12 plies, i.e. 3 to 6 full moves) drawn from:
1. data/sample_real.pgn (Lichess rated games between master-level players)
2. Canonical classical opening variations across ECO groups A, B, C, D, E.

Validations:
- Strictly legal move sequences via chess.Board.
- Non-terminal, balanced positions (material balance, basic material delta <= 1 pawn).
- Deduplicated by position FEN / Zobrist key.
- Balanced representation across ECO groups A, B, C, D, E.
- Outputs data/openings_200.txt and data/openings_200_audit.json.
"""

import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import chess
import chess.pgn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PGN_PATH = os.path.join(REPO_ROOT, "data", "sample_real.pgn")
OUTPUT_TXT = os.path.join(REPO_ROOT, "data", "openings_200.txt")
OUTPUT_AUDIT = os.path.join(REPO_ROOT, "data", "openings_200_audit.json")

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def compute_material_balance(board: chess.Board) -> Tuple[int, int, int]:
    """Returns (white_material, black_material, eval_diff_white_minus_black)."""
    white_val = 0
    black_val = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            val = PIECE_VALUES.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_val += val
            else:
                black_val += val
    return white_val, black_val, white_val - black_val


def classify_eco_group(board: chess.Board, moves_san: List[str]) -> str:
    """Classifies an opening line into ECO group A, B, C, D, or E based on standard rules."""
    if not moves_san:
        return "A"
    first_white = moves_san[0]
    first_black = moves_san[1] if len(moves_san) > 1 else ""

    # Group B: 1. e4 with black not playing 1... e5 or 1... e6 (Sicilian, Caro-Kann, Pirc, Alekhine, Scandinavian, etc.)
    if first_white == "e4":
        if first_black == "e5":
            return "C"  # Open Games (1. e4 e5)
        elif first_black == "e6":
            return "C"  # French Defence (1. e4 e6)
        else:
            return "B"  # Semi-Open other than French (Sicilian c5, Caro-Kann c6, etc.)

    # 1. d4 without 1... d5 or 1... Nf6 is mostly A
    if first_white == "d4":
        if first_black == "d5":
            # Group D: Closed games (1. d4 d5), Queen's Gambit, Slav, etc.
            # (Note: Gruenfeld 1. d4 Nf6 2. c4 g6 3... d5 is D, handled below)
            return "D"
        elif first_black == "Nf6":
            # 1. d4 Nf6
            # Can be A (Old Indian, Trompowsky), D (Gruenfeld), or E (Indian systems: King's Indian, Nimzo, Queen's Indian, Bogo)
            if len(moves_san) >= 3 and moves_san[2] == "c4":
                if len(moves_san) >= 4:
                    if moves_san[3] == "g6":
                        # King's Indian or Gruenfeld
                        if len(moves_san) >= 6 and moves_san[5] == "d5":
                            return "D"  # Gruenfeld
                        return "E"  # King's Indian (E60-E99)
                    elif moves_san[3] in ("e6", "c5"):
                        if moves_san[3] == "c5":
                            return "A"  # Benoni
                        return "E"  # Nimzo-Indian, Queen's Indian, Bogo-Indian, Catalan
                return "E"
            return "A"
        else:
            # 1. d4 f5 (Dutch - A), 1. d4 e6 (can transpose), 1. d4 g6, etc.
            return "A"

    # Any other first move: 1. c4 (English - A), 1. Nf3 (Reti - A), 1. f4 (Bird - A), 1. b3 (Nimzo-Larsen - A), etc.
    return "A"


# Curated catalog of standard, balanced classical & modern opening sequences across ECO A-E
CANONICAL_OPENINGS: List[Tuple[str, str, str]] = [
    # (Name, ECO_Group, SAN_string)
    # --- ECO Group A (Flank openings, English, Reti, Dutch, Benoni) ---
    ("English Opening: Symmetrical", "A", "c4 c5 Nf3 Nf6 Nc3 Nc6"),
    ("English Opening: Four Knights", "A", "c4 e5 Nc3 Nf6 Nf3 Nc6"),
    ("English Opening: Anglo-Indian", "A", "c4 Nf6 Nc3 e6 Nf3 Bb4"),
    ("English Opening: King's English", "A", "c4 e5 Nc3 Nc6 g3 g6 Bg2 Bg7"),
    ("English Opening: Closed System", "A", "c4 e5 Nc3 d6 g3 Nc6 Bg2 g6"),
    ("Reti Opening: Classical", "A", "Nf3 d5 c4 c6 b3 Nf6 Bb2"),
    ("Reti Opening: King's Indian Setup", "A", "Nf3 Nf6 c4 g6 g3 Bg7 Bg2 O-O"),
    ("Reti Opening: Advance Variation", "A", "Nf3 d5 c4 d4 b4 Nf6 Bb2"),
    ("Modern Defense: Standard", "A", "g3 g6 Bg2 Bg7 d4 d6 Nf3 Nf6"),
    ("King's Indian Attack: vs French", "A", "Nf3 d5 g3 Nf6 Bg2 e6 O-O Be7 d3"),
    ("King's Indian Attack: vs Sicilian", "A", "Nf3 c5 g3 Nc6 Bg2 g6 O-O Bg7 d3"),
    ("Nimzo-Larsen Attack: Classical", "A", "b3 e5 Bb2 Nc6 e3 Nf6 Bb5"),
    ("Dutch Defense: Classical", "A", "d4 f5 c4 Nf6 g3 e6 Bg2 Be7"),
    ("Dutch Defense: Leningrad", "A", "d4 f5 g3 Nf6 Bg2 g6 Nf3 Bg7 O-O"),
    ("Dutch Defense: Stonewall", "A", "d4 f5 c4 e6 g3 d5 Bg2 c6 Nf3"),
    ("Benoni Defense: Modern", "A", "d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6"),
    ("Benko Gambit: Accepted Main", "A", "d4 Nf6 c4 c5 d5 b5 cxb5 a6 bxa6 g6"),
    ("Old Indian Defense: Standard", "A", "d4 Nf6 c4 d6 Nc3 e5 Nf3 Nbd7"),
    ("English: Hedgehog Formation", "A", "c4 c5 Nf3 Nf6 g3 b6 Bg2 Bb7 O-O e6"),
    ("Reti: Anglo-Slav", "A", "Nf3 d5 c4 c6 e3 Nf6 Nc3 e6 b3"),
    ("English: Reversed Sicilian", "A", "c4 e5 g3 Nf6 Bg2 d5 cxd5 Nxd5"),
    ("English: Double Fianchetto", "A", "c4 c5 b3 Nf6 Bb2 g6 g3 Bg7 Bg2"),
    ("Reti: Neo-Catalan", "A", "Nf3 Nf6 c4 e6 g3 d5 Bg2 Be7 O-O"),
    ("English: Flohr-Mikenas", "A", "c4 Nf6 Nc3 e6 e4 d5 e5 d4"),
    ("Reti: Queenside Fianchetto", "A", "Nf3 d5 b3 Nf6 Bb2 Bf5 e3 e6"),
    ("English: Botvinnik System", "A", "c4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 e4 d6 Nge2"),
    ("Bird Opening: Dutch Reversed", "A", "f4 d5 Nf3 Nf6 e3 g6 b3 Bg7 Bb2"),
    ("Nimzo-Larsen: Classical Center", "A", "b3 d5 Bb2 c5 e3 Nc6 Bb5 Nf6"),
    ("English: Kramnik-Shirov Counter", "A", "c4 e5 Nc3 Bb4 Nd5 Be7 d4 d6"),
    ("Reti: Advance Defense", "A", "Nf3 d5 c4 d4 g3 Nc6 Bg2 e5"),
    ("English: Agincourt Defense", "A", "c4 e6 Nf3 d5 g3 Nf6 Bg2 Be7 O-O O-O"),
    ("English: Wimpy System", "A", "c4 e5 Nc3 Nc6 Nf3 f5 d4 e4"),
    ("Reti: Capablanca System", "A", "Nf3 d5 c4 c6 b3 Bg4 Ne5 Bh5"),
    ("English: Carls-Bremen System", "A", "c4 e5 Nc3 Nf6 g3 c6 Nf3 e4 Nd4 d5"),
    ("Modern Defense: Averbakh Setup", "A", "d4 g6 c4 Bg7 Nc3 d6 e4 Nf6 Be2"),
    ("English: Reversed Dragon", "A", "c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5 cxd5 Nxd5"),
    ("Dutch Defense: Semi-Leningrad", "A", "d4 f5 Nf3 Nf6 c4 g6 Nc3 Bg7 Bg5"),
    ("English: Keres Defense", "A", "c4 e5 g3 c6 Nf3 e4 Nd4 d5 cxd5 Qxd5"),
    ("Reti: King's Gambit Reversed", "A", "Nf3 d5 c4 dxc4 Na3 c5 Nxc4 Nc6"),
    ("English: Smyslov System", "A", "c4 e5 Nc3 Nf6 Nf3 Nc6 g3 Bb4 Bg2 O-O"),

    # --- ECO Group B (1. e4 without 1... e5 or 1... e6: Sicilian, Caro-Kann, Pirc, Alekhine, Scandi) ---
    ("Sicilian: Open Najdorf", "B", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6"),
    ("Sicilian: Scheveningen", "B", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 e6"),
    ("Sicilian: Classical", "B", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6"),
    ("Sicilian: Dragon Yugoslav Setup", "B", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6"),
    ("Sicilian: Closed Main", "B", "e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6"),
    ("Sicilian: Alapin 2. c3", "B", "e4 c5 c3 d5 exd5 Qxd5 d4 Nf6 Nf3 e6"),
    ("Sicilian: Grand Prix Attack", "B", "e4 c5 Nc3 Nc6 f4 g6 Nf3 Bg7 Bc4 e6"),
    ("Sicilian: Kan Variation", "B", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Bd3 Nf6"),
    ("Sicilian: Taimanov", "B", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7"),
    ("Sicilian: Four Knights", "B", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6"),
    ("Sicilian: Richter-Rauzer", "B", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bg5 e6"),
    ("Sicilian: Sveshnikov Preparation", "B", "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5"),
    ("Sicilian: Rossolimo Attack", "B", "e4 c5 Nf3 Nc6 Bb5 g6 O-O Bg7 Re1"),
    ("Sicilian: Moscow Variation", "B", "e4 c5 Nf3 d6 Bb5+ Bd7 Bxd7+ Qxd7 O-O Nf6"),
    ("Caro-Kann: Classical Main", "B", "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6"),
    ("Caro-Kann: Advance Short System", "B", "e4 c6 d4 d5 e5 Bf5 Nf3 e6 Be2 Nd7"),
    ("Caro-Kann: Advance Tal Variation", "B", "e4 c6 d4 d5 e5 Bf5 h4 h5 Bd3 Bxd3"),
    ("Caro-Kann: Panov-Botvinnik Attack", "B", "e4 c6 d4 d5 exd5 cxd5 c4 Nf6 Nc3 e6"),
    ("Caro-Kann: Tartakower (4... Nf6)", "B", "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Nf6 Nxf6+ exf6"),
    ("Caro-Kann: Korchnoi (4... Nf6 gxf6)", "B", "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Nf6 Nxf6+ gxf6"),
    ("Caro-Kann: Two Knights", "B", "e4 c6 Nc3 d5 Nf3 Bg4 h3 Bxf3 Qxf3 e6"),
    ("Caro-Kann: Exchange", "B", "e4 c6 d4 d5 exd5 cxd5 Bd3 Nc6 c3 Nf6"),
    ("Scandinavian: Main 3... Qa5", "B", "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6"),
    ("Scandinavian: Modern 2... Nf6", "B", "e4 d5 exd5 Nf6 d4 Nxd5 Nf3 Bg4 Be2 e6"),
    ("Scandinavian: 3... Qd6", "B", "e4 d5 exd5 Qxd5 Nc3 Qd6 d4 Nf6 Nf3 a6"),
    ("Alekhine's Defense: Modern", "B", "e4 Nf6 e5 Nd5 d4 d6 Nf3 g6 Bc4 Nb6"),
    ("Alekhine's Defense: Exchange", "B", "e4 Nf6 e5 Nd5 d4 d6 c4 Nb6 exd6 cxd6"),
    ("Pirc Defense: Classical", "B", "e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7 Be2 O-O"),
    ("Pirc Defense: Austrian Attack", "B", "e4 d6 d4 Nf6 Nc3 g6 f4 Bg7 Nf3 O-O"),
    ("Pirc Defense: 150 Attack", "B", "e4 d6 d4 Nf6 Nc3 g6 Be3 c6 Qd2 b5"),
    ("Modern Defense: Standard Setup", "B", "e4 g6 d4 Bg7 Nc3 d6 Be3 a6 Qd2 Nd7"),
    ("Sicilian: O'Kelly Variation", "B", "e4 c5 Nf3 a6 c3 d5 exd5 Qxd5 d4"),
    ("Sicilian: Nimzowitsch 2... Nf6", "B", "e4 c5 Nf3 Nf6 e5 Nd5 Nc3 e6 Nxd5 exd5"),
    ("Sicilian: Pin Variation", "B", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 Bb4"),
    ("Sicilian: Hyper-Accelerated Dragon", "B", "e4 c5 Nf3 g6 d4 cxd4 Nxd4 Nc6 Nc3 Bg7"),
    ("Sicilian: Accelerated Dragon Gurgenidze", "B", "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6 c4 Bg7"),
    ("Sicilian: Chameleon Variation", "B", "e4 c5 Nc3 Nc6 Nge2 g6 d4 cxd4 Nxd4"),
    ("Caro-Kann: Gurgenidze System", "B", "e4 c6 d4 d5 Nc3 g6 e5 Bg7 f4 h5"),
    ("Caro-Kann: Modern 4... Nd7", "B", "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Nd7 Nf3 Ngf6"),
    ("Pirc Defense: Byrne System", "B", "e4 d6 d4 Nf6 Nc3 g6 Bg5 Bg7 Qd2 h6"),

    # --- ECO Group C (1. e4 e5 and 1. e4 e6 French) ---
    ("Ruy Lopez: Berlin Defense", "C", "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 d4 Nd6"),
    ("Ruy Lopez: Closed Main Line", "C", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6"),
    ("Ruy Lopez: Open Variation", "C", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Nxe4 d4 b5 Bb3 d5"),
    ("Ruy Lopez: Marshall Attack Prep", "C", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O c3 d5"),
    ("Ruy Lopez: Exchange Variation", "C", "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O f6 d4 exd4"),
    ("Ruy Lopez: Breyer Defense", "C", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8"),
    ("Ruy Lopez: Chigorin Defense", "C", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Na5"),
    ("Italian Game: Giuoco Pianissimo", "C", "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O a6"),
    ("Italian Game: Two Knights Defense", "C", "e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Bc5 c3 O-O O-O d6"),
    ("Italian Game: Main Center Attack", "C", "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d4 exd4 cxd4 Bb4+"),
    ("Four Knights Game: Spanish", "C", "e4 e5 Nf3 Nc6 Nc3 Nf6 Bb5 Bb4 O-O O-O"),
    ("Four Knights Game: Scotch", "C", "e4 e5 Nf3 Nc6 Nc3 Nf6 d4 exd4 Nxd4 Bb4"),
    ("Scotch Game: Classical", "C", "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Be3 Qf6"),
    ("Scotch Game: Mieses Variation", "C", "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7"),
    ("Petrov's Defense: Classical 3. Nxe5", "C", "e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3"),
    ("Petrov's Defense: Modern 3. d4", "C", "e4 e5 Nf3 Nf6 d4 Nxe4 Bd3 d5 Nxe5 Nd7"),
    ("Vienna Game: Falkbeer Defense", "C", "e4 e5 Nc3 Nf6 f4 d5 fxe5 Nxe4 Nf3 Bc5"),
    ("Vienna Game: Max Lange Defense", "C", "e4 e5 Nc3 Nf6 Bc4 Bc5 d3 d6 f4 Nc6"),
    ("King's Gambit Declined: Classical", "C", "e4 e5 f4 Bc5 Nf3 d6 Nc3 Nf6 Bc4 Nc6"),
    ("French Defense: Winawer Main Line", "C", "e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3"),
    ("French Defense: Classical 3... Nf6", "C", "e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7 e5 Nfd7"),
    ("French Defense: Tarrasch Open", "C", "e4 e6 d4 d5 Nd2 c5 exd5 exd5 Ngf3 Nf6 Bb5+"),
    ("French Defense: Tarrasch Closed", "C", "e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7 Bd3 c5 c3 Nc6"),
    ("French Defense: Advance Paulsen", "C", "e4 e6 d4 d5 e5 c5 c3 Nc6 Nf3 Qb6 Bd3"),
    ("French Defense: Exchange Variation", "C", "e4 e6 d4 d5 exd5 exd5 Nf3 Nf6 Bd3 Bd6 O-O"),
    ("French Defense: Rubinstein 3... dxe4", "C", "e4 e6 d4 d5 Nc3 dxe4 Nxe4 Nd7 Nf3 Ngf6"),
    ("French Defense: McCutcheon", "C", "e4 e6 d4 d5 Nc3 Nf6 Bg5 Bb4 e5 h6 Bd2"),
    ("French Defense: Guimard Variation", "C", "e4 e6 d4 d5 Nd2 Nc6 Ngf3 Nf6 e5 Nd7"),
    ("Philidor Defense: Hanham Variation", "C", "e4 e5 Nf3 d6 d4 Nd7 Bc4 c6 O-O Be7"),
    ("Bishop's Opening: Berlin Defense", "C", "e4 e5 Bc4 Nf6 d3 c6 Nf3 d5 Bb3 Bd6"),
    ("Ruy Lopez: Steinitz Modern", "C", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 d6 c3 Bd7 d4"),
    ("Ruy Lopez: Cozio Defense", "C", "e4 e5 Nf3 Nc6 Bb5 Nge7 O-O g6 c3 Bg7"),
    ("Italian: Evans Gambit Declined", "C", "e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bb6 a4 a6"),
    ("Scotch Four Knights", "C", "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nc3 Bb4"),
    ("French: King's Indian Attack", "C", "e4 e6 d3 d5 Nd2 Nf6 Ngf3 Nc6 g3 dxe4"),
    ("French: Burn Variation", "C", "e4 e6 d4 d5 Nc3 Nf6 Bg5 dxe4 Nxe4 Be7"),
    ("Petrov: Three Knights Game", "C", "e4 e5 Nf3 Nf6 Nc3 Bb4 Nxe5 O-O Be2 Re8"),
    ("Vienna: Quiet Variation", "C", "e4 e5 Nc3 Nc6 g3 Bc5 Bg2 d6 Nge2"),
    ("Ruy Lopez: Worrall Attack", "C", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 Qe2 b5 Bb3 Be7"),
    ("Italian: Hungarian Defense", "C", "e4 e5 Nf3 Nc6 Bc4 Be7 d4 d6 h3 Nf6"),

    # --- ECO Group D (Closed Games: 1. d4 d5, Queen's Gambit, Slav, Semi-Slav, Gruenfeld) ---
    ("Queen's Gambit Declined: Classical Orthodox", "D", "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3"),
    ("Queen's Gambit Declined: Tartakower", "D", "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bh4 b6"),
    ("Queen's Gambit Declined: Lasker Defense", "D", "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bh4 Ne4"),
    ("Queen's Gambit Declined: Exchange Carlsbad", "D", "d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 c6 e3 Be7 Bd3"),
    ("Queen's Gambit Declined: Semi-Tarrasch", "D", "d4 d5 c4 e6 Nc3 Nf6 Nf3 c5 cxd5 Nxd5 e4 Nxc3 bxc3"),
    ("Queen's Gambit Declined: Tarrasch", "D", "d4 d5 c4 e6 Nc3 c5 cxd5 exd5 Nf3 Nc6 g3 Nf6"),
    ("Queen's Gambit Accepted: Classical", "D", "d4 d5 c4 dxc4 Nf3 Nf6 e3 e6 Bxc4 c5 O-O a6"),
    ("Queen's Gambit Accepted: Central", "D", "d4 d5 c4 dxc4 e4 e5 Nf3 exd4 Bxc4 Bb4+ Bd2"),
    ("Slav Defense: Classical Main", "D", "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6"),
    ("Slav Defense: Chebanenko 4... a6", "D", "d4 d5 c4 c6 Nf3 Nf6 Nc3 a6 e3 b5 b3"),
    ("Slav Defense: Exchange", "D", "d4 d5 c4 c6 cxd5 cxd5 Nc3 Nf6 Nf3 Nc6 Bf4"),
    ("Slav Defense: Quiet 4. e3", "D", "d4 d5 c4 c6 Nf3 Nf6 e3 Bf5 Nc3 e6 Nh4"),
    ("Semi-Slav: Meran Variation", "D", "d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 e3 Nbd7 Bd3 dxc4 Bxc4 b5"),
    ("Semi-Slav: Botvinnik System", "D", "d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 Bg5 dxc4 e4 b5 e5"),
    ("Semi-Slav: Moscow Variation", "D", "d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 Bg5 h6 Bxf6 Qxf6 e3"),
    ("Semi-Slav: Anti-Meran 5... a6", "D", "d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 e3 a6 b3 Bb4"),
    ("Gruenfeld: Russian System", "D", "d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4"),
    ("Gruenfeld: Exchange Classical", "D", "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bc4 c5 Ne2"),
    ("Gruenfeld: Exchange Modern", "D", "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 c5"),
    ("Gruenfeld: 3. f3", "D", "d4 Nf6 c4 g6 f3 d5 cxd5 Nxd5 e4 Nb6 Nc3 Bg7 Be3"),
    ("Gruenfeld: Bf4 System", "D", "d4 Nf6 c4 g6 Nc3 d5 Bf4 Bg7 e3 O-O Rc1 c5"),
    ("Queen's Pawn: London System Main", "D", "d4 d5 Bf4 Nf6 e3 c5 c3 Nc6 Nd2 e6 Ngf3 Bd6"),
    ("Queen's Pawn: Torre Attack", "D", "d4 d5 Nf3 Nf6 Bg5 e6 e3 Be7 Nbd2 Nbd7 Bd3 c5"),
    ("Queen's Pawn: Colle System", "D", "d4 d5 Nf3 Nf6 e3 e6 Bd3 c5 c3 Nbd7 Nbd2 Bd6"),
    ("Queen's Pawn: Veresov Attack", "D", "d4 d5 Nc3 Nf6 Bg5 Nbd7 Qd3 c6 e4 dxe4 Nxe4"),
    ("Chigorin Defense: Main", "D", "d4 d5 c4 Nc6 Nf3 Bg4 cxd5 Bxf3 gxf3 Qxd5 e3 e5"),
    ("Albin Countergambit: 4. Nf3", "D", "d4 d5 c4 e5 dxe5 d4 Nf3 Nc6 g3 Nge7 Bg2"),
    ("QGD: Vienna Variation", "D", "d4 d5 c4 e6 Nc3 Nf6 Nf3 dxc4 e4 Bb4 Bg5 c5"),
    ("QGD: Ragozin Defense", "D", "d4 d5 c4 e6 Nc3 Nf6 Nf3 Bb4 cxd5 exd5 Bg5 h6"),
    ("QGD: Modern 4... Nbd7", "D", "d4 d5 c4 e6 Nc3 Nf6 Nf3 Nbd7 Bg5 Be7 e3 O-O"),
    ("QGD: Harrwitz Attack", "D", "d4 d5 c4 e6 Nc3 Nf6 Bf4 Be7 e3 O-O Nf3 c5"),
    ("Slav Defense: Schlechter", "D", "d4 d5 c4 c6 Nf3 Nf6 Nc3 g6 e3 Bg7 Be2 O-O"),
    ("Gruenfeld: Bg5 System", "D", "d4 Nf6 c4 g6 Nc3 d5 Bg5 Ne4 Bh4 Nxc3 bxc3 dxc4"),
    ("Queen's Pawn: Jobava London", "D", "d4 d5 Nc3 Nf6 Bf4 c5 e3 a6 Nf3 Nc6 Be2"),
    ("Slav: Steiner Variation", "D", "d4 d5 c4 c6 Nc3 dxc4 e4 b5 a4 b4 Na2"),
    ("QGD: Alatortsev Variation", "D", "d4 d5 c4 e6 Nc3 Be7 cxd5 exd5 Bf4 c6 e3 Bf5"),
    ("QGA: 3... a6", "D", "d4 d5 c4 dxc4 Nf3 a6 e3 b5 a4 Bb7 b3"),
    ("Gruenfeld: Sevilla Variation", "D", "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bc4 O-O Ne2 c5 Be3 Nc6 O-O Bg4 f3 Na5"),
    ("Colle-Zukertort System", "D", "d4 d5 Nf3 Nf6 e3 e6 Bd3 c5 b3 Nc6 Bb2 Bd6 O-O"),
    ("QGD: Cambridge Springs", "D", "d4 d5 c4 e6 Nc3 Nf6 Bg5 Nbd7 e3 c6 Nf3 Qa5"),

    # --- ECO Group E (Indian Systems: King's Indian, Nimzo-Indian, Queen's Indian, Catalan, Bogo) ---
    ("King's Indian: Classical Mar del Plata", "E", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7"),
    ("King's Indian: Classical Gligoric", "E", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 Be3 Ng4 Bg5 f6"),
    ("King's Indian: Saemisch Main", "E", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 d5 c6"),
    ("King's Indian: Averbakh System", "E", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Bg5 c5 d5 e6"),
    ("King's Indian: Four Pawns Attack", "E", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f4 O-O Nf3 c5 d5 e6"),
    ("King's Indian: Fianchetto Panno", "E", "d4 Nf6 c4 g6 Nf3 Bg7 g3 O-O Bg2 d6 O-O Nc6 Nc3 a6"),
    ("King's Indian: Fianchetto Classical", "E", "d4 Nf6 c4 g6 Nf3 Bg7 g3 O-O Bg2 d6 O-O Nbd7 Nc3 e5"),
    ("King's Indian: Zinnowitz Variation", "E", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 Bg4 Be3 Nfd7"),
    ("Nimzo-Indian: Rubinstein Main", "E", "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5 O-O"),
    ("Nimzo-Indian: Classical (4. Qc2)", "E", "d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3+ Qxc3 b6"),
    ("Nimzo-Indian: Leningrad (4. Bg5)", "E", "d4 Nf6 c4 e6 Nc3 Bb4 Bg5 h6 Bh4 c5 d5 d6"),
    ("Nimzo-Indian: Kasparov (4. Nf3 c5)", "E", "d4 Nf6 c4 e6 Nc3 Bb4 Nf3 c5 g3 cxd4 Nxd4 O-O"),
    ("Nimzo-Indian: Saemisch (4. a3)", "E", "d4 Nf6 c4 e6 Nc3 Bb4 a3 Bxc3+ bxc3 c5 e3 b6"),
    ("Nimzo-Indian: Hubner Variation", "E", "d4 Nf6 c4 e6 Nc3 Bb4 e3 c5 Bd3 Nc6 Ne2 cxd4 exd4 d5"),
    ("Queen's Indian: Classical 4. g3 Ba6", "E", "d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 Bb4+ Bd2 Be7"),
    ("Queen's Indian: Classical 4. g3 Bb7", "E", "d4 Nf6 c4 e6 Nf3 b6 g3 Bb7 Bg2 Be7 O-O O-O"),
    ("Queen's Indian: Miles Variation", "E", "d4 Nf6 c4 e6 Nf3 b6 Bf4 Bb7 e3 Be7 h3 O-O"),
    ("Queen's Indian: Petrosian System", "E", "d4 Nf6 c4 e6 Nf3 b6 a3 Ba6 Qc2 Bb7 Nc3 c5"),
    ("Bogo-Indian Defense: 4. Bd2", "E", "d4 Nf6 c4 e6 Nf3 Bb4+ Bd2 Qe7 g3 Nc6 Bg2 Bxd2+"),
    ("Bogo-Indian Defense: 4. Nbd2", "E", "d4 Nf6 c4 e6 Nf3 Bb4+ Nbd2 b6 a3 Bxd2+ Bxd2 Bb7"),
    ("Catalan Opening: Open Classical", "E", "d4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O O-O dxc4 Qc2 a6"),
    ("Catalan Opening: Closed Main", "E", "d4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O O-O Nbd7 Qc2 c6"),
    ("Catalan Opening: Modern 4... dxc4", "E", "d4 Nf6 c4 e6 g3 d5 Bg2 dxc4 Nf3 a6 O-O Nc6"),
    ("King's Indian: Petrosian System", "E", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 d5 a5 Bg5 Na6"),
    ("King's Indian: Makogonov 6. h3", "E", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O h3 e5 d5 a5"),
    ("King's Indian: Larsen Variation", "E", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be3 Ng4 Bg5 c5"),
    ("Nimzo-Indian: Fischer 4. e3 b6", "E", "d4 Nf6 c4 e6 Nc3 Bb4 e3 b6 Ne2 Ba6 a3 Be7"),
    ("Nimzo-Indian: Kmoch Variation", "E", "d4 Nf6 c4 e6 Nc3 Bb4 f3 d5 a3 Bxc3+ bxc3 c5"),
    ("Catalan: Hungarian Variation", "E", "d4 Nf6 c4 e6 g3 d5 Bg2 Bb4+ Bd2 Be7 Nf3 O-O"),
    ("Catalan: Botvinnik Setup", "E", "d4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O O-O c6 Qc2 Nbd7"),
    ("Queen's Indian: Nimzowitsch Variation", "E", "d4 Nf6 c4 e6 Nf3 b6 g3 Bb7 Bg2 Bb4+ Bd2 a5"),
    ("King's Indian: Pomar System", "E", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O exd4 Nxd4 Re8"),
    ("Nimzo-Indian: Romanishin System", "E", "d4 Nf6 c4 e6 Nc3 Bb4 g3 c5 Nf3 cxd4 Nxd4 O-O"),
    ("Catalan: Semi-Closed 5. Nf3 c6", "E", "d4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O O-O c6 b3 Nbd7"),
    ("King's Indian: Bayonet Attack", "E", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 b4 Nh5"),
    ("King's Indian: Simagin Variation", "E", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Ne1 Nd7"),
    ("Queen's Indian: Ribli Variation", "E", "d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 d5 cxd5 Nxd5"),
    ("Nimzo-Indian: Reshevsky Variation", "E", "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Ne2 d5 a3 Be7"),
    ("Catalan: Closed 4... c6", "E", "d4 Nf6 c4 e6 g3 d5 Bg2 c6 Nf3 Nbd7 O-O Be7"),
    ("King's Indian: Gallagher Variation", "E", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Nd2 a5"),
]


def test_and_build_candidate(
    moves_san: List[str],
    source: str,
    name: str = "",
    min_plies: int = 6,
    max_plies: int = 12,
) -> Optional[Dict[str, Any]]:
    """Plays out moves_san up to [min_plies..max_plies], verifies legality, balance, non-terminal."""
    if len(moves_san) < min_plies:
        return None

    # Determine candidate ply length
    # Target 8 plies (4 moves) or 10 plies (5 moves), clamped by available moves and max_plies
    target_len = min(len(moves_san), max_plies)
    if target_len < min_plies:
        return None

    board = chess.Board()
    san_played = []
    for m_san in moves_san[:target_len]:
        try:
            mv = board.parse_san(m_san)
            board.push(mv)
            san_played.append(m_san)
        except Exception:
            return None

    # Check non-terminal
    if board.is_game_over(claim_draw=True):
        return None

    # Check material balance: difference between White and Black piece value <= 1 (e.g. gambited pawn max)
    w_mat, b_mat, diff = compute_material_balance(board)
    if abs(diff) > 1:
        return None

    eco_grp = classify_eco_group(board, san_played)
    fen = board.fen()
    epd = board.epd()  # position without halfmove/fullmove

    return {
        "san": " ".join(san_played),
        "ply_len": len(san_played),
        "eco_group": eco_grp,
        "name": name,
        "source": source,
        "w_mat": w_mat,
        "b_mat": b_mat,
        "mat_diff": diff,
        "epd": epd,
        "fen": fen,
    }


def extract_from_sample_real_pgn(
    pgn_path: str,
    min_plies: int = 6,
    max_plies: int = 12,
) -> List[Dict[str, Any]]:
    """Extracts valid, balanced opening prefixes from data/sample_real.pgn."""
    if not os.path.isfile(pgn_path):
        print(f"Warning: {pgn_path} not found.")
        return []

    candidates = []
    with open(pgn_path, "r", encoding="utf-8") as f:
        game_idx = 0
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            game_idx += 1
            moves = list(g.mainline_moves())
            if len(moves) < min_plies:
                continue

            # Test multiple prefixes (e.g. 8 plies, 10 plies, 12 plies, 6 plies)
            board = chess.Board()
            sans = []
            for m in moves[:max_plies]:
                sans.append(board.san(m))
                board.push(m)

            # Try lengths from min_plies to max_plies
            for length in (8, 10, 6, 12, 7, 9, 11):
                if len(sans) >= length:
                    sub_sans = sans[:length]
                    cand = test_and_build_candidate(
                        sub_sans,
                        source=f"sample_real_pgn_game_{game_idx}",
                        name=f"Lichess Master Game {game_idx}",
                        min_plies=min_plies,
                        max_plies=max_plies,
                    )
                    if cand:
                        candidates.append(cand)
    return candidates


def main():
    print(f"=== UniChessSSM Direction 1: Arena 200+ Balanced Openings Builder ===")
    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

    # 1. Gather candidates from canonical definitions
    canonical_candidates: List[Dict[str, Any]] = []
    for name, eco_hint, san_seq in CANONICAL_OPENINGS:
        moves_san = san_seq.strip().split()
        cand = test_and_build_candidate(
            moves_san,
            source="canonical_ecopool",
            name=name,
            min_plies=6,
            max_plies=12,
        )
        if cand:
            canonical_candidates.append(cand)

    print(f"Canonical valid candidates: {len(canonical_candidates)}")

    # 2. Gather candidates from sample_real.pgn
    pgn_candidates = extract_from_sample_real_pgn(PGN_PATH, min_plies=6, max_plies=12)
    print(f"Extracted candidates from sample_real.pgn: {len(pgn_candidates)}")

    # 3. Deduplicate and balance across ECO groups A, B, C, D, E
    # Target exactly 200 openings, with roughly equal distribution across groups:
    # 200 / 5 = 40 per ECO group (A: 40, B: 40, C: 40, D: 40, E: 40)
    seen_epds: Set[str] = set()
    selected_by_group: Dict[str, List[Dict[str, Any]]] = {
        "A": [], "B": [], "C": [], "D": [], "E": []
    }

    # First pass: canonical candidates (top quality, standard named variations)
    for c in canonical_candidates:
        grp = c["eco_group"]
        epd = c["epd"]
        if epd not in seen_epds:
            seen_epds.add(epd)
            selected_by_group[grp].append(c)

    print("After canonical pass:")
    for grp in sorted(selected_by_group.keys()):
        print(f"  Group {grp}: {len(selected_by_group[grp])} openings")

    # Second pass: fill from sample_real.pgn candidates
    for c in pgn_candidates:
        grp = c["eco_group"]
        epd = c["epd"]
        if epd not in seen_epds:
            seen_epds.add(epd)
            selected_by_group[grp].append(c)

    print("After PGN candidate pooling:")
    for grp in sorted(selected_by_group.keys()):
        print(f"  Group {grp}: {len(selected_by_group[grp])} openings available")

    # Now curate exactly 40 per group for a perfectly balanced 200-opening set
    # Target: 40 per group. If any group has slightly fewer, we rebalance or adjust.
    final_selected: List[Dict[str, Any]] = []
    target_per_group = 40

    # Ensure canonical ones take precedence, then real PGN
    for grp in ["A", "B", "C", "D", "E"]:
        pool = selected_by_group[grp]
        # Prioritize canonical candidates, then sample_real
        canonical_in_pool = [x for x in pool if x["source"] == "canonical_ecopool"]
        real_in_pool = [x for x in pool if x["source"] != "canonical_ecopool"]
        combined = canonical_in_pool + real_in_pool

        if len(combined) < target_per_group:
            print(f"Warning: Group {grp} has {len(combined)} < {target_per_group}")
            chosen = combined
        else:
            chosen = combined[:target_per_group]
        selected_by_group[grp] = chosen
        final_selected.extend(chosen)

    # If total < 200, take remaining best from any pool
    if len(final_selected) < 200:
        needed = 200 - len(final_selected)
        leftover = []
        for grp in ["A", "B", "C", "D", "E"]:
            pool = selected_by_group[grp]
            chosen_epds = {x["epd"] for x in pool}
            # All candidates for this group not yet chosen
            for c in canonical_candidates + pgn_candidates:
                if c["eco_group"] == grp and c["epd"] not in chosen_epds and c["epd"] not in {x["epd"] for x in final_selected}:
                    leftover.append(c)
        final_selected.extend(leftover[:needed])

    # Trim to exactly 200
    final_selected = final_selected[:200]

    # Verification of final selection
    print(f"\nTotal selected openings: {len(final_selected)}")
    assert len(final_selected) == 200, f"Expected 200 openings, got {len(final_selected)}"

    # Audit checks
    eco_dist = Counter(x["eco_group"] for x in final_selected)
    ply_lengths = [x["ply_len"] for x in final_selected]
    mat_diffs = [x["mat_diff"] for x in final_selected]
    sources = Counter(x["source"].split("_")[0] for x in final_selected)

    audit_report = {
        "total_openings": len(final_selected),
        "target_openings": 200,
        "eco_distribution": dict(sorted(eco_dist.items())),
        "ply_length_stats": {
            "min": int(min(ply_lengths)),
            "max": int(max(ply_lengths)),
            "mean": round(float(sum(ply_lengths) / len(ply_lengths)), 2),
        },
        "material_balance": {
            "all_abs_diff_le_1": bool(all(abs(d) <= 1 for d in mat_diffs)),
            "equal_material_count": sum(1 for d in mat_diffs if d == 0),
            "gambit_1p_count": sum(1 for d in mat_diffs if abs(d) == 1),
            "max_abs_material_diff": max(abs(d) for d in mat_diffs),
        },
        "sources_summary": dict(sources),
        "openings_sample": [
            {
                "index": i + 1,
                "san": x["san"],
                "eco_group": x["eco_group"],
                "name": x["name"],
                "ply_len": x["ply_len"],
                "mat_diff": x["mat_diff"],
            }
            for i, x in enumerate(final_selected[:10])
        ],
    }

    # Write data/openings_200.txt
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for x in final_selected:
            f.write(x["san"].strip() + "\n")

    # Write data/openings_200_audit.json
    with open(OUTPUT_AUDIT, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)

    print("\n--- Summary Table ---")
    print(f"{'ECO Group':<12} | {'Count':<8} | {'Canonical':<10} | {'Real Master':<12}")
    print("-" * 50)
    for grp in ["A", "B", "C", "D", "E"]:
        grp_items = [x for x in final_selected if x["eco_group"] == grp]
        canon_count = sum(1 for x in grp_items if x["source"] == "canonical_ecopool")
        real_count = len(grp_items) - canon_count
        print(f"{grp:<12} | {len(grp_items):<8} | {canon_count:<10} | {real_count:<12}")
    print("-" * 50)
    print(f"{'Total':<12} | {len(final_selected):<8} | {sum(1 for x in final_selected if x['source'] == 'canonical_ecopool'):<10} | {sum(1 for x in final_selected if x['source'] != 'canonical_ecopool'):<12}")

    print(f"\nPly lengths: Min={min(ply_lengths)}, Max={max(ply_lengths)}, Mean={audit_report['ply_length_stats']['mean']}")
    print(f"Material balance: All |diff| <= 1: {audit_report['material_balance']['all_abs_diff_le_1']} (Equal: {audit_report['material_balance']['equal_material_count']}, 1-Pawn gambit: {audit_report['material_balance']['gambit_1p_count']})")
    print(f"Saved: {OUTPUT_TXT} (lines: {len(final_selected)})")
    print(f"Saved: {OUTPUT_AUDIT}")


if __name__ == "__main__":
    main()
