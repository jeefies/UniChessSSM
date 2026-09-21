#!/usr/bin/env python3
"""Loop 6: Selfplay Opening Book Injection & Endgame Variety Ecosystem Simulation.

Compares two generation regimes across 200 simulated full games (100 per regime):
- Regime A (Baseline Scratch B0): Start from standard initial board, pure Gumbel Top-16 exploration.
- Regime B (Opening Book Injection): Randomly sample an opening from data/openings_200.txt (ply 6~12),
  then continue with Gumbel Top-16 exploration.

Compares:
1. Endgame pawn structure entropy (quantified by pawn hash distribution at ply 40+).
2. Piece activity distribution (occupancy heatmap across 64 squares).
3. Tactical sharpness (frequency of checkmate vs repetition vs 50-move draws).
4. Invalid truncation rate (games reaching ply 300 without conclusion).

Saves results to runs/loop6_opening_injection.json and prints summary table.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.polyglot
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stateseq.actions import FROM_ACTION, move_to_action
from stateseq.adapter import classify_final_board
from stateseq.gumbel import (
    C_SCALE,
    C_VISIT,
    M0,
    N_SIMS,
    Node,
    _Candidate,
    _n_rounds,
    gumbel_topm,
    qtransform_completed,
    select_action,
)

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}
SWEET_CENTER = {chess.C3, chess.C4, chess.C5, chess.C6,
                chess.D3, chess.D6, chess.E3, chess.E6,
                chess.F3, chess.F4, chess.F5, chess.F6}


def resolve_move_fast(action: int, board: chess.Board) -> chess.Move | None:
    """Fast resolve action to legal move handling queen promotion."""
    frm, to, promo = FROM_ACTION[action]
    p = board.piece_at(frm)
    if p is not None and p.piece_type == chess.PAWN:
        to_rank = to >> 3
        if (to_rank == 7 or to_rank == 0) and promo is None:
            promo = chess.QUEEN
    m = chess.Move(frm, to, promotion=promo)
    return m if board.is_legal(m) else None


def evaluate_board(board: chess.Board) -> Tuple[np.ndarray, np.ndarray, float]:
    """Lightweight policy & value evaluation for selfplay simulation."""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32), 0.0

    actions = []
    scores = []
    for m in legal_moves:
        a = move_to_action(m)
        if a is None:
            continue
        actions.append(a)
        score = 1.0
        # Capture bonus with MVV-LVA flavor
        if board.is_capture(m):
            victim = board.piece_at(m.to_square)
            attacker = board.piece_at(m.from_square)
            vic_val = PIECE_VALUES.get(victim.piece_type, 1) if victim else 1
            att_val = PIECE_VALUES.get(attacker.piece_type, 1) if attacker else 1
            score += 2.0 + max(0.0, float(vic_val - att_val * 0.2))
        # Check incentive
        if board.gives_check(m):
            score += 1.5
        # Center development
        if m.to_square in CENTER_SQUARES:
            score += 0.8
        elif m.to_square in SWEET_CENTER:
            score += 0.3
        scores.append(score)

    actions_arr = np.array(actions, dtype=np.int64)
    scores_arr = np.array(scores, dtype=np.float32)
    probs = scores_arr / scores_arr.sum()
    logits = np.log(probs + 1e-12).astype(np.float32)
    logits -= logits.mean()

    # Dynamic material & positional eval for q
    w_mat = sum(len(board.pieces(pt, chess.WHITE)) * PIECE_VALUES[pt] for pt in PIECE_VALUES)
    b_mat = sum(len(board.pieces(pt, chess.BLACK)) * PIECE_VALUES[pt] for pt in PIECE_VALUES)
    mat_diff = (w_mat - b_mat) / 25.0
    q = float(np.tanh(mat_diff))
    if board.turn == chess.BLACK:
        q = -q
    return actions_arr, logits, q


class SimulationSearcher:
    """Gumbel Top-16 + Sequential Halving searcher matching production spec."""

    def __init__(
        self,
        n_sims: int = N_SIMS,
        m0: int = M0,
        g: float = 1.0,
        c_visit: float = C_VISIT,
        c_scale: float = C_SCALE,
    ):
        self.n_sims = n_sims
        self.m0 = m0
        self.g = g
        self.c_visit = c_visit
        self.c_scale = c_scale

    def search_action(self, root_board: chess.Board, rng: np.random.Generator) -> int:
        legal_arr, logits_arr, q_val = evaluate_board(root_board)
        if len(legal_arr) <= 1:
            return int(legal_arr[0]) if len(legal_arr) == 1 else -1

        root = Node(legal=legal_arr, logits=logits_arr, q=q_val, depth=0, path=())
        root._board = root_board

        def expand(parent: Node, action: int) -> Node:
            b_sim = parent._board.copy()
            mv = resolve_move_fast(action, b_sim)
            if mv is None:
                raise RuntimeError(f"Action {action} is not legal on board {b_sim.fen()}")
            b_sim.push(mv)

            if b_sim.is_game_over(claim_draw=True):
                outcome = b_sim.outcome(claim_draw=True)
                if outcome is None or outcome.winner is None:
                    val = 0.0
                else:
                    val = 1.0 if outcome.winner == b_sim.turn else -1.0
                child = Node(
                    legal=np.array([], dtype=np.int64),
                    logits=np.array([], dtype=np.float32),
                    q=val,
                    depth=parent.depth + 1,
                    path=parent.path + (action,),
                    terminal=True,
                )
                child._board = b_sim
                return child

            sub_legal, sub_logits, sub_q = evaluate_board(b_sim)
            child = Node(
                legal=sub_legal,
                logits=sub_logits,
                q=sub_q,
                depth=parent.depth + 1,
                path=parent.path + (action,),
            )
            child._board = b_sim
            return child

        cands = gumbel_topm(root, m0=self.m0, rng=rng, g=self.g)
        m = len(cands)
        rounds = _n_rounds(m)
        surv = [_Candidate(action=a, noise=ns) for a, ns in cands]

        base, rem = divmod(self.n_sims, rounds)
        budget_per_round = [base + (1 if i < rem else 0) for i in range(rounds)]

        def _simulate(node: Node) -> float:
            if node.is_terminal:
                return float(node.q)
            a = select_action(node, self.c_visit, self.c_scale)
            edge_idx = int(np.flatnonzero(node.legal == a)[0])
            child = node.children.get(int(a))
            if child is None:
                child = expand(node, a)
                node.children[int(a)] = child
                val = -float(child.q)
            else:
                val = -_simulate(child)
            node.record_child(edge_idx, val)
            return val

        def do_sim_root(c: _Candidate) -> None:
            if c.child is None:
                c.child = expand(root, c.action)
                val = -float(c.child.q)
            elif c.child.is_terminal:
                val = -float(c.child.q)
            else:
                val = -_simulate(c.child)
            idx = int(np.flatnonzero(root.legal == c.action)[0])
            root.record_child(idx, val)

        for r, budget in enumerate(budget_per_round):
            if len(surv) == 1:
                budget = sum(budget_per_round[r:])
            per_base, per_rem = divmod(budget, len(surv))
            for i, c in enumerate(surv):
                k = per_base + (1 if i < per_rem else 0)
                for _ in range(k):
                    do_sim_root(c)
            if len(surv) == 1:
                break

            l_root = {int(a): float(x) for a, x in zip(root.legal, root.logits)}
            s_root_vals = qtransform_completed(root, self.c_visit, self.c_scale)
            s_map = {int(a): float(x) for a, x in zip(root.legal, s_root_vals)}
            scored = sorted(
                ((c.noise + l_root[c.action] + s_map[c.action], c) for c in surv),
                key=lambda t: -t[0],
            )
            keep = max(1, (len(surv) + 1) // 2)
            surv = [c for _, c in scored[:keep]]

        return int(surv[0].action)


def compute_pawn_hash(board: chess.Board) -> int:
    """Computes Polyglot Zobrist hash of the pawn structure exclusively."""
    pawn_board = chess.Board(None)
    for sq in chess.scan_forward(board.pawns & board.occupied_co[chess.WHITE]):
        pawn_board.set_piece_at(sq, chess.Piece(chess.PAWN, chess.WHITE))
    for sq in chess.scan_forward(board.pawns & board.occupied_co[chess.BLACK]):
        pawn_board.set_piece_at(sq, chess.Piece(chess.PAWN, chess.BLACK))
    return chess.polyglot.zobrist_hash(pawn_board)


def shannon_entropy_from_counts(counts: Counter) -> float:
    """Computes Shannon entropy in nats from discrete occurrence counts."""
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    probs = np.array(list(counts.values()), dtype=np.float64) / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


@dataclass
class GameResult:
    regime: str
    game_id: int
    plies: int
    result: int  # 0=win, 1=draw, 2=loss
    termination_reason: str
    is_truncated: bool
    opening_san: List[str]
    pawn_hashes_ply40_plus: List[int]
    square_occupancy: List[int]  # 64-element piece occupancy sum across all plies


def simulate_single_game(args: Tuple[str, int, Optional[List[str]], int]) -> Dict[str, Any]:
    """Simulates a single full game under Regime A or Regime B."""
    regime, game_id, opening_san, seed = args
    rng = np.random.default_rng(seed)
    searcher = SimulationSearcher(n_sims=N_SIMS, m0=M0, g=1.0)

    board = chess.Board()
    occupancy = np.zeros(64, dtype=np.int32)
    pawn_hashes: List[int] = []

    # If opening book injection, apply opening moves
    if opening_san:
        for san in opening_san:
            mv = board.parse_san(san)
            board.push(mv)
            # Record occupancy
            for sq in chess.scan_forward(board.occupied):
                occupancy[sq] += 1
            if len(board.move_stack) >= 40:
                pawn_hashes.append(compute_pawn_hash(board))

    # Search remainder of game until termination or ply 300
    while len(board.move_stack) < 300:
        if board.is_game_over(claim_draw=True):
            break
        action = searcher.search_action(board, rng)
        mv = resolve_move_fast(action, board)
        if mv is None:
            break
        board.push(mv)

        # Record occupancy
        for sq in chess.scan_forward(board.occupied):
            occupancy[sq] += 1

        if len(board.move_stack) >= 40:
            pawn_hashes.append(compute_pawn_hash(board))

    # Classify final outcome via authoritative adapter
    result, reason, is_truncated = classify_final_board(board)

    return {
        "regime": regime,
        "game_id": game_id,
        "plies": len(board.move_stack),
        "result": result,
        "termination_reason": reason,
        "is_truncated": is_truncated,
        "opening_san": opening_san or [],
        "pawn_hashes": pawn_hashes,
        "occupancy": occupancy.tolist(),
    }


def analyze_regime_results(regime_name: str, game_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyzes game records for a regime and extracts comparative metrics."""
    n_games = len(game_records)
    total_plies = sum(g["plies"] for g in game_records)
    avg_plies = total_plies / max(1, n_games)

    # 1. Pawn structure entropy at ply 40+
    all_pawn_hashes = []
    pawn_hashes_at_ply40 = []
    for g in game_records:
        hashes = g["pawn_hashes"]
        if hashes:
            pawn_hashes_at_ply40.append(hashes[0])  # pawn hash exactly at ply 40 or first >=40
        all_pawn_hashes.extend(hashes)

    pawn_hash_counts_all = Counter(all_pawn_hashes)
    pawn_entropy_all = shannon_entropy_from_counts(pawn_hash_counts_all)

    pawn_hash_counts_ply40 = Counter(pawn_hashes_at_ply40)
    pawn_entropy_ply40 = shannon_entropy_from_counts(pawn_hash_counts_ply40)
    unique_pawn_structures_ply40 = len(pawn_hash_counts_ply40)

    # 2. Piece activity distribution (occupancy heatmap across 64 squares)
    total_occupancy = np.zeros(64, dtype=np.float64)
    for g in game_records:
        total_occupancy += np.array(g["occupancy"], dtype=np.float64)

    sum_occ = total_occupancy.sum()
    occ_distribution = total_occupancy / max(1.0, sum_occ)
    occ_entropy = float(-np.sum(occ_distribution[occ_distribution > 0] * np.log(occ_distribution[occ_distribution > 0])))

    # Group occupancy into zones
    center_occ = float(sum(occ_distribution[sq] for sq in CENTER_SQUARES))
    sweet_center_occ = float(sum(occ_distribution[sq] for sq in SWEET_CENTER))
    flank_occ = float(1.0 - (center_occ + sweet_center_occ))

    # Gini coefficient of square occupancy (higher = more concentrated/less uniform)
    sorted_occ = np.sort(occ_distribution)
    n = len(sorted_occ)
    index = np.arange(1, n + 1)
    gini_occupancy = float((np.sum((2 * index - n - 1) * sorted_occ)) / (n * np.sum(sorted_occ)))

    # 3. Tactical sharpness & termination reasons
    reason_counts = Counter(g["termination_reason"] for g in game_records)
    checkmate_count = reason_counts.get("checkmate", 0)
    repetition_count = reason_counts.get("threefold", 0)
    fifty_moves_count = reason_counts.get("fifty_move", 0)
    stalemate_count = reason_counts.get("stalemate", 0)
    insufficient_count = reason_counts.get("insufficient_material", 0)
    truncated_count = sum(1 for g in game_records if g["is_truncated"])

    tactical_sharpness = {
        "checkmate_count": checkmate_count,
        "checkmate_rate": round(checkmate_count / n_games, 4),
        "threefold_repetition_count": repetition_count,
        "threefold_rate": round(repetition_count / n_games, 4),
        "fifty_move_draw_count": fifty_moves_count,
        "fifty_move_rate": round(fifty_moves_count / n_games, 4),
        "stalemate_count": stalemate_count,
        "insufficient_material_count": insufficient_count,
        "decisive_rate": round((checkmate_count) / n_games, 4),
        "draw_rate": round((n_games - checkmate_count - truncated_count) / n_games, 4),
    }

    # 4. Invalid truncation rate
    invalid_truncation_rate = round(truncated_count / n_games, 4)

    return {
        "regime": regime_name,
        "num_games": n_games,
        "avg_game_plies": round(avg_plies, 2),
        "endgame_pawn_metrics": {
            "unique_pawn_structures_ply40": unique_pawn_structures_ply40,
            "pawn_entropy_ply40_nats": round(pawn_entropy_ply40, 4),
            "pawn_entropy_all_endgame_nats": round(pawn_entropy_all, 4),
            "total_endgame_positions_sampled": len(all_pawn_hashes),
        },
        "piece_activity_distribution": {
            "occupancy_spatial_entropy_nats": round(occ_entropy, 4),
            "occupancy_gini_index": round(gini_occupancy, 4),
            "center_occupancy_share": round(center_occ, 4),
            "sweet_center_share": round(sweet_center_occ, 4),
            "flank_occupancy_share": round(flank_occ, 4),
            "heatmap_64": [round(float(x), 5) for x in occ_distribution],
        },
        "tactical_sharpness": tactical_sharpness,
        "invalid_truncation": {
            "truncated_count": truncated_count,
            "truncation_rate": invalid_truncation_rate,
        },
    }


def main():
    print("=" * 78)
    print("Loop 6: Selfplay Opening Book Injection & Endgame Variety Simulation")
    print("Environment: Python with multiprocessing pool")
    print("=" * 78)

    # 1. Load opening book from data/openings_200.txt
    openings_file = os.path.join(REPO_ROOT, "data", "openings_200.txt")
    if not os.path.exists(openings_file):
        raise FileNotFoundError(f"Opening book not found at {openings_file}")

    with open(openings_file, "r", encoding="utf-8") as f:
        openings_list = [line.strip().split() for line in f if line.strip()]

    print(f"Loaded {len(openings_list)} opening sequences from {openings_file}")

    num_games_per_regime = 100
    base_seed = 42000

    # Build job list
    tasks_a = [
        ("Regime_A_Baseline_Scratch_B0", i, None, base_seed + i)
        for i in range(num_games_per_regime)
    ]

    rng_openings = np.random.default_rng(20260921)
    tasks_b = []
    for i in range(num_games_per_regime):
        chosen_opening = list(openings_list[rng_openings.integers(0, len(openings_list))])
        tasks_b.append(
            ("Regime_B_Opening_Book_Injection", i, chosen_opening, base_seed + 10000 + i)
        )

    all_tasks = tasks_a + tasks_b
    total_games = len(all_tasks)

    # Determine CPU workers
    num_cpus = min(12, os.cpu_count() or 4)
    print(f"Simulating {total_games} games across 2 regimes ({num_games_per_regime} each) using {num_cpus} processes...")
    t0 = time.time()

    from concurrent.futures import ProcessPoolExecutor, as_completed
    all_results = [None] * total_games
    done_count = 0
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        future_to_idx = {executor.submit(simulate_single_game, task): idx for idx, task in enumerate(all_tasks)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            res = future.result()
            all_results[idx] = res
            done_count += 1
            if done_count % 20 == 0 or done_count == total_games:
                curr_t = time.time() - t0
                print(f"  [{done_count}/{total_games}] games completed ({curr_t:.1f}s, {done_count/curr_t:.2f} games/s)...")

    t1 = time.time()
    elapsed = t1 - t0
    print(f"\nAll {total_games} games completed in {elapsed:.2f}s ({total_games / elapsed:.2f} games/s)!")

    # Separate records
    records_a = [r for r in all_results if r["regime"] == "Regime_A_Baseline_Scratch_B0"]
    records_b = [r for r in all_results if r["regime"] == "Regime_B_Opening_Book_Injection"]

    analysis_a = analyze_regime_results("Regime_A_Baseline_Scratch_B0", records_a)
    analysis_b = analyze_regime_results("Regime_B_Opening_Book_Injection", records_b)

    # Save output json
    runs_dir = os.path.join(REPO_ROOT, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    out_json_path = os.path.join(runs_dir, "loop6_opening_injection.json")

    final_payload = {
        "metadata": {
            "task": "Loop 6: Selfplay Opening Book Injection & Endgame Variety Simulation",
            "date": "2026-09-21",
            "total_games_simulated": total_games,
            "games_per_regime": num_games_per_regime,
            "elapsed_seconds": round(elapsed, 2),
            "search_config": {
                "n_sims": N_SIMS,
                "m0": M0,
                "g": 1.0,
                "c_visit": C_VISIT,
                "c_scale": C_SCALE,
                "max_plies": 300,
            },
        },
        "regimes": {
            "Regime_A_Baseline_Scratch_B0": analysis_a,
            "Regime_B_Opening_Book_Injection": analysis_b,
        },
        "comparison_summary": {
            "pawn_entropy_ply40_gain": round(
                analysis_b["endgame_pawn_metrics"]["pawn_entropy_ply40_nats"]
                - analysis_a["endgame_pawn_metrics"]["pawn_entropy_ply40_nats"],
                4,
            ),
            "pawn_entropy_ply40_gain_percent": round(
                (analysis_b["endgame_pawn_metrics"]["pawn_entropy_ply40_nats"]
                 - analysis_a["endgame_pawn_metrics"]["pawn_entropy_ply40_nats"])
                / max(0.01, analysis_a["endgame_pawn_metrics"]["pawn_entropy_ply40_nats"]) * 100.0,
                2,
            ),
            "unique_pawn_structures_ply40_ratio": round(
                analysis_b["endgame_pawn_metrics"]["unique_pawn_structures_ply40"]
                / max(1, analysis_a["endgame_pawn_metrics"]["unique_pawn_structures_ply40"]),
                3,
            ),
            "spatial_entropy_gain": round(
                analysis_b["piece_activity_distribution"]["occupancy_spatial_entropy_nats"]
                - analysis_a["piece_activity_distribution"]["occupancy_spatial_entropy_nats"],
                4,
            ),
            "tactical_decisive_delta": round(
                analysis_b["tactical_sharpness"]["checkmate_rate"]
                - analysis_a["tactical_sharpness"]["checkmate_rate"],
                4,
            ),
            "invalid_truncation_delta": round(
                analysis_b["invalid_truncation"]["truncation_rate"]
                - analysis_a["invalid_truncation"]["truncation_rate"],
                4,
            ),
        },
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\nSaved structured experimental results to {out_json_path}")
    print(f"File exists: {os.path.exists(out_json_path)} (size: {os.path.getsize(out_json_path)} bytes)")

    # Print Summary Table
    print("\n" + "=" * 88)
    print("LOOP 6: EXPERIMENTAL RESULTS SUMMARY TABLE")
    print("=" * 88)
    fmt_hdr = "{:<38} | {:<20} | {:<20} | {:<12}"
    fmt_row = "{:<38} | {:<20} | {:<20} | {:<12}"
    print(fmt_hdr.format("Metric", "Regime A (Scratch B0)", "Regime B (Book Injected)", "Delta / Ratio"))
    print("-" * 88)

    # Pawn structure
    p_ent_a = analysis_a["endgame_pawn_metrics"]["pawn_entropy_ply40_nats"]
    p_ent_b = analysis_b["endgame_pawn_metrics"]["pawn_entropy_ply40_nats"]
    p_diff = p_ent_b - p_ent_a
    print(fmt_row.format("Pawn Structure Entropy @ ply 40+", f"{p_ent_a:.4f} nats", f"{p_ent_b:.4f} nats", f"+{p_diff:.4f} nats"))

    u_pawn_a = analysis_a["endgame_pawn_metrics"]["unique_pawn_structures_ply40"]
    u_pawn_b = analysis_b["endgame_pawn_metrics"]["unique_pawn_structures_ply40"]
    print(fmt_row.format("Unique Pawn Structures @ ply 40", f"{u_pawn_a} / {num_games_per_regime}", f"{u_pawn_b} / {num_games_per_regime}", f"x{u_pawn_b/max(1,u_pawn_a):.2f}"))

    # Piece activity
    s_ent_a = analysis_a["piece_activity_distribution"]["occupancy_spatial_entropy_nats"]
    s_ent_b = analysis_b["piece_activity_distribution"]["occupancy_spatial_entropy_nats"]
    s_diff = s_ent_b - s_ent_a
    print(fmt_row.format("Square Occupancy Spatial Entropy", f"{s_ent_a:.4f} nats", f"{s_ent_b:.4f} nats", f"+{s_diff:.4f} nats"))

    gini_a = analysis_a["piece_activity_distribution"]["occupancy_gini_index"]
    gini_b = analysis_b["piece_activity_distribution"]["occupancy_gini_index"]
    print(fmt_row.format("Occupancy Gini Index (Dispersion)", f"{gini_a:.4f}", f"{gini_b:.4f}", f"{gini_b - gini_a:+.4f}"))

    c_occ_a = analysis_a["piece_activity_distribution"]["center_occupancy_share"] * 100
    c_occ_b = analysis_b["piece_activity_distribution"]["center_occupancy_share"] * 100
    print(fmt_row.format("Center (d4/e4/d5/e5) Occupancy %", f"{c_occ_a:.2f}%", f"{c_occ_b:.2f}%", f"{c_occ_b - c_occ_a:+.2f}%"))

    # Tactical Sharpness
    mate_a = analysis_a["tactical_sharpness"]["checkmate_rate"] * 100
    mate_b = analysis_b["tactical_sharpness"]["checkmate_rate"] * 100
    print(fmt_row.format("Checkmate Rate (Decisive %)", f"{mate_a:.1f}% ({analysis_a['tactical_sharpness']['checkmate_count']})", f"{mate_b:.1f}% ({analysis_b['tactical_sharpness']['checkmate_count']})", f"{mate_b - mate_a:+.1f}%"))

    rep_a = analysis_a["tactical_sharpness"]["threefold_rate"] * 100
    rep_b = analysis_b["tactical_sharpness"]["threefold_rate"] * 100
    print(fmt_row.format("Threefold Repetition Rate", f"{rep_a:.1f}% ({analysis_a['tactical_sharpness']['threefold_repetition_count']})", f"{rep_b:.1f}% ({analysis_b['tactical_sharpness']['threefold_repetition_count']})", f"{rep_b - rep_a:+.1f}%"))

    fifty_a = analysis_a["tactical_sharpness"]["fifty_move_rate"] * 100
    fifty_b = analysis_b["tactical_sharpness"]["fifty_move_rate"] * 100
    print(fmt_row.format("50-Move Draw Rate", f"{fifty_a:.1f}% ({analysis_a['tactical_sharpness']['fifty_move_draw_count']})", f"{fifty_b:.1f}% ({analysis_b['tactical_sharpness']['fifty_move_draw_count']})", f"{fifty_b - fifty_a:+.1f}%"))

    # Invalid truncation
    trunc_a = analysis_a["invalid_truncation"]["truncation_rate"] * 100
    trunc_b = analysis_b["invalid_truncation"]["truncation_rate"] * 100
    print(fmt_row.format("Invalid Truncation (ply 300)", f"{trunc_a:.1f}% ({analysis_a['invalid_truncation']['truncated_count']})", f"{trunc_b:.1f}% ({analysis_b['invalid_truncation']['truncated_count']})", f"{trunc_b - trunc_a:+.1f}%"))

    avg_p_a = analysis_a["avg_game_plies"]
    avg_p_b = analysis_b["avg_game_plies"]
    print(fmt_row.format("Average Game Length (plies)", f"{avg_p_a:.1f}", f"{avg_p_b:.1f}", f"{avg_p_b - avg_p_a:+.1f}"))
    print("=" * 88)


if __name__ == "__main__":
    mp.freeze_support()
    main()
