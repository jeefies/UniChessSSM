#!/usr/bin/env python3
"""Loop 8: Selfplay Early Adjudication Thresholds (Resign / Draw Cutoff Tradeoff).

Simulates and analyzes game termination behavior in selfplay:
1. When one side has overwhelming advantage (Q > 0.95 or Q < -0.90 for K consecutive plies),
   does continuing the game up to ply 150-250 waste compute on trivial conversions?
2. When both sides repeat or shuffle in dead draws with Q in [-0.05, 0.05] for 20+ plies,
   does waiting for the 50-move rule (which takes up to 100 plies) inflate average game length
   and trigger ply 300 truncations?

Evaluates 4 adjudication policies across 200 self-play game trajectories:
- Baseline: No early adjudication (play until checkmate, 50-move rule, 3-fold repetition, or ply 300 cutoff).
- Policy 1 (Conservative Resignation): Resign if Q < -0.90 for 4 consecutive plies after ply 30.
- Policy 2 (Conservative Early Draw): Claim draw if |Q| < 0.05 and material is symmetrical with no pawn moves for 12 consecutive plies after ply 40.
- Policy 3 (Combined Resign + Draw Adjudication): Policy 1 + Policy 2.

Measures:
- Average plies per game (sample throughput gain).
- Invalidation / error rate (% of adjudicated games where the trailing side actually had a saving tactic or opponent blundered in ground truth).
- Truncation elimination (% reduction in ply 300 truncations).
- Effective GPU generation speedup factor.

Saves results to runs/loop8_adjudication.json and prints summary table.
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
SWEET_CENTER = {
    chess.C3, chess.C4, chess.C5, chess.C6,
    chess.D3, chess.D6, chess.E3, chess.E6,
    chess.F3, chess.F4, chess.F5, chess.F6,
}


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
    """Lightweight policy & value evaluation for selfplay simulation.
    Returns (legal_actions, logits, q_to_move) where q_to_move in [-1, 1].
    """
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

    # Material & mobility eval for q
    w_mat = sum(len(board.pieces(pt, chess.WHITE)) * PIECE_VALUES[pt] for pt in PIECE_VALUES)
    b_mat = sum(len(board.pieces(pt, chess.BLACK)) * PIECE_VALUES[pt] for pt in PIECE_VALUES)
    # Scaled so a full piece / rook advantage reaches |Q| ~ 0.90 - 0.98
    mat_diff = (w_mat - b_mat) / 3.5
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

    def search_action_and_q(self, root_board: chess.Board, rng: np.random.Generator) -> Tuple[int, float]:
        legal_arr, logits_arr, q_val = evaluate_board(root_board)
        if len(legal_arr) <= 1:
            return (int(legal_arr[0]) if len(legal_arr) == 1 else -1), q_val

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

        chosen_action = int(surv[0].action)
        # Compute root q_est from simulated children or root evaluation
        best_cand = surv[0]
        if best_cand.child is not None:
            root_q = -float(best_cand.child.q)
        else:
            root_q = float(root.q)
        return chosen_action, root_q


def is_material_symmetrical(board: chess.Board) -> bool:
    """Checks if white and black have identical piece counts for each piece type."""
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        if len(board.pieces(pt, chess.WHITE)) != len(board.pieces(pt, chess.BLACK)):
            return False
    return True


@dataclass
class PlyState:
    ply: int
    turn: chess.Color
    fen: str
    move_san: str
    q_to_move: float
    q_white: float
    is_pawn_move_or_capture: bool
    is_material_sym: bool


def simulate_full_game_trajectory(args: Tuple[int, Optional[List[str]], int]) -> Dict[str, Any]:
    """Simulates a full self-play game trajectory up to ply 300 without early adjudication.
    Records per-ply metrics (q_to_move, material symmetry, pawn moves, check, etc.).
    """
    game_id, opening_san, seed = args
    rng = np.random.default_rng(seed)
    searcher = SimulationSearcher(n_sims=N_SIMS, m0=M0, g=1.0)

    board = chess.Board()
    trajectory: List[Dict[str, Any]] = []

    # 1. Apply opening moves if present
    if opening_san:
        for san in opening_san:
            mv = board.parse_san(san)
            p = board.piece_at(mv.from_square)
            is_pawn_move = (p is not None and p.piece_type == chess.PAWN)
            is_capture = board.is_capture(mv)
            turn_before = board.turn
            _, _, q_eval = evaluate_board(board)
            q_white = q_eval if turn_before == chess.WHITE else -q_eval
            sym = is_material_symmetrical(board)

            trajectory.append({
                "ply": len(board.move_stack),
                "turn": "white" if turn_before == chess.WHITE else "black",
                "move_san": san,
                "q_to_move": round(float(q_eval), 4),
                "q_white": round(float(q_white), 4),
                "is_pawn_move": is_pawn_move,
                "is_capture": is_capture,
                "is_material_sym": sym,
            })
            board.push(mv)

    # 2. Simulate until game over or ply 300
    while len(board.move_stack) < 300:
        if board.is_game_over(claim_draw=True):
            break
        turn_before = board.turn
        action, root_q = searcher.search_action_and_q(board, rng)
        mv = resolve_move_fast(action, board)
        if mv is None:
            break

        p = board.piece_at(mv.from_square)
        is_pawn_move = (p is not None and p.piece_type == chess.PAWN)
        is_capture = board.is_capture(mv)
        q_white = root_q if turn_before == chess.WHITE else -root_q
        san = board.san(mv)
        sym = is_material_symmetrical(board)

        trajectory.append({
            "ply": len(board.move_stack),
            "turn": "white" if turn_before == chess.WHITE else "black",
            "move_san": san,
            "q_to_move": round(float(root_q), 4),
            "q_white": round(float(q_white), 4),
            "is_pawn_move": is_pawn_move,
            "is_capture": is_capture,
            "is_material_sym": sym,
        })
        board.push(mv)

    # Classify ground truth outcome via authoritative adapter
    ground_truth_result, ground_truth_reason, ground_truth_truncated = classify_final_board(board)

    return {
        "game_id": game_id,
        "opening_san": opening_san or [],
        "ground_truth_plies": len(board.move_stack),
        "ground_truth_result": ground_truth_result,  # 0=White win, 1=Draw, 2=Black win
        "ground_truth_reason": ground_truth_reason,
        "ground_truth_truncated": ground_truth_truncated,
        "trajectory": trajectory,
    }


def evaluate_policies_on_trajectory(game: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Evaluates Baseline and Policies 1, 2, 3 against the recorded trajectory."""
    traj = game["trajectory"]
    gt_plies = game["ground_truth_plies"]
    gt_result = game["ground_truth_result"]
    gt_reason = game["ground_truth_reason"]
    gt_truncated = game["ground_truth_truncated"]

    # Baseline: no early adjudication
    res_baseline = {
        "adjudicated": False,
        "adjudication_type": "none",
        "plies": gt_plies,
        "result": gt_result,
        "reason": gt_reason,
        "is_truncated": gt_truncated,
        "is_invalid_or_blunder": False,
    }

    # Policy 1: Resign if Q < -0.90 for 4 consecutive plies of the trailing side (i.e. |q_white| > 0.90 for 4 consecutive plies with consistent advantage sign) after ply 30.
    # Note: If one side has Q > 0.95 or opponent has Q < -0.90, the trailing side resigns.
    # Across consecutive plies, if q_white > +0.90 for 4 plies, White is winning and Black resigns (result = 0).
    # If q_white < -0.90 for 4 plies, Black is winning and White resigns (result = 2).
    p1_adjudicated = False
    p1_ply = gt_plies
    p1_result = gt_result
    p1_reason = gt_reason
    p1_truncated = gt_truncated
    p1_invalid = False

    c_white_win = 0
    c_black_win = 0
    for i, step in enumerate(traj):
        ply = step["ply"]
        qw = step["q_white"]
        if ply >= 30:
            if qw > 0.90:
                c_white_win += 1
                c_black_win = 0
                if c_white_win >= 4:
                    p1_adjudicated = True
                    p1_ply = ply + 1
                    p1_reason = "resignation_adjudicated"
                    p1_truncated = False
                    p1_result = 0  # White wins, Black resigned
                    if gt_result != p1_result:
                        p1_invalid = True
                    break
            elif qw < -0.90:
                c_black_win += 1
                c_white_win = 0
                if c_black_win >= 4:
                    p1_adjudicated = True
                    p1_ply = ply + 1
                    p1_reason = "resignation_adjudicated"
                    p1_truncated = False
                    p1_result = 2  # Black wins, White resigned
                    if gt_result != p1_result:
                        p1_invalid = True
                    break
            else:
                c_white_win = 0
                c_black_win = 0
        else:
            c_white_win = 0
            c_black_win = 0

    res_p1 = {
        "adjudicated": p1_adjudicated,
        "adjudication_type": "resign" if p1_adjudicated else "none",
        "plies": p1_ply,
        "result": p1_result,
        "reason": p1_reason,
        "is_truncated": p1_truncated,
        "is_invalid_or_blunder": p1_invalid,
    }

    # Policy 2: Early Draw if |Q| < 0.05 and material is symmetrical with no pawn moves for 12 consecutive plies after ply 40.
    p2_adjudicated = False
    p2_ply = gt_plies
    p2_result = gt_result
    p2_reason = gt_reason
    p2_truncated = gt_truncated
    p2_invalid = False

    consecutive_draw_plies = 0
    for i, step in enumerate(traj):
        ply = step["ply"]
        q_to_move = step["q_to_move"]
        is_sym = step["is_material_sym"]
        is_pawn_move = step["is_pawn_move"]
        if ply >= 40:
            if abs(q_to_move) < 0.05 and is_sym and not is_pawn_move:
                consecutive_draw_plies += 1
                if consecutive_draw_plies >= 12:
                    p2_adjudicated = True
                    p2_ply = ply + 1
                    p2_reason = "early_draw_adjudicated"
                    p2_result = 1  # Draw
                    p2_truncated = False
                    # Ground truth invalidation check:
                    # Invalidation occurs if ground truth game was decisive (result != 1)
                    if gt_result != 1:
                        p2_invalid = True
                    break
            else:
                consecutive_draw_plies = 0
        else:
            consecutive_draw_plies = 0

    res_p2 = {
        "adjudicated": p2_adjudicated,
        "adjudication_type": "draw" if p2_adjudicated else "none",
        "plies": p2_ply,
        "result": p2_result,
        "reason": p2_reason,
        "is_truncated": p2_truncated,
        "is_invalid_or_blunder": p2_invalid,
    }

    # Policy 3: Combined Resign + Draw Adjudication (First triggered)
    # Walk ply by ply and check both conditions
    p3_adjudicated = False
    p3_type = "none"
    p3_ply = gt_plies
    p3_result = gt_result
    p3_reason = gt_reason
    p3_truncated = gt_truncated
    p3_invalid = False

    c_w_win = 0
    c_b_win = 0
    c_draw = 0
    for i, step in enumerate(traj):
        ply = step["ply"]
        qw = step["q_white"]
        is_sym = step["is_material_sym"]
        is_pawn_move = step["is_pawn_move"]

        # Resign condition
        if ply >= 30:
            if qw > 0.90:
                c_w_win += 1
                c_b_win = 0
            elif qw < -0.90:
                c_b_win += 1
                c_w_win = 0
            else:
                c_w_win = 0
                c_b_win = 0
        else:
            c_w_win = 0
            c_b_win = 0

        # Draw condition
        if ply >= 40 and abs(qw) < 0.05 and is_sym and not is_pawn_move:
            c_draw += 1
        else:
            c_draw = 0

        if c_w_win >= 4:
            p3_adjudicated = True
            p3_type = "resign"
            p3_ply = ply + 1
            p3_reason = "resignation_adjudicated"
            p3_truncated = False
            p3_result = 0  # White wins
            if gt_result != p3_result:
                p3_invalid = True
            break
        elif c_b_win >= 4:
            p3_adjudicated = True
            p3_type = "resign"
            p3_ply = ply + 1
            p3_reason = "resignation_adjudicated"
            p3_truncated = False
            p3_result = 2  # Black wins
            if gt_result != p3_result:
                p3_invalid = True
            break
        elif c_draw >= 12:
            p3_adjudicated = True
            p3_type = "draw"
            p3_ply = ply + 1
            p3_reason = "early_draw_adjudicated"
            p3_result = 1
            p3_truncated = False
            if gt_result != 1:
                p3_invalid = True
            break

    res_p3 = {
        "adjudicated": p3_adjudicated,
        "adjudication_type": p3_type,
        "plies": p3_ply,
        "result": p3_result,
        "reason": p3_reason,
        "is_truncated": p3_truncated,
        "is_invalid_or_blunder": p3_invalid,
    }

    return {
        "baseline": res_baseline,
        "policy1_resign": res_p1,
        "policy2_draw": res_p2,
        "policy3_combined": res_p3,
    }


def analyze_policy_metrics(policy_name: str, policy_results: List[Dict[str, Any]], baseline_plies_total: int) -> Dict[str, Any]:
    n = len(policy_results)
    total_plies = sum(r["plies"] for r in policy_results)
    avg_plies = total_plies / max(1, n)
    adjudicated_count = sum(1 for r in policy_results if r["adjudicated"])
    invalid_count = sum(1 for r in policy_results if r["is_invalid_or_blunder"])
    truncated_count = sum(1 for r in policy_results if r["is_truncated"])
    
    # Plies saved vs baseline
    plies_saved = baseline_plies_total - total_plies
    plies_saved_pct = (plies_saved / max(1, baseline_plies_total)) * 100.0
    speedup_factor = baseline_plies_total / max(1, total_plies)

    # Invalidation rate: % of adjudicated games that were invalidated
    invalidation_rate_of_adjudicated = (invalid_count / max(1, adjudicated_count)) * 100.0 if adjudicated_count > 0 else 0.0
    overall_error_rate = (invalid_count / max(1, n)) * 100.0

    # Truncation reduction
    baseline_trunc_count = sum(1 for r in policy_results if r.get("is_truncated_baseline", False))

    reasons = Counter(r["reason"] for r in policy_results)
    adjudication_types = Counter(r["adjudication_type"] for r in policy_results)

    return {
        "policy_name": policy_name,
        "total_games": n,
        "total_plies": total_plies,
        "avg_plies_per_game": round(avg_plies, 2),
        "adjudicated_games": adjudicated_count,
        "adjudication_rate_pct": round((adjudicated_count / n) * 100.0, 2),
        "adjudication_breakdown": dict(adjudication_types),
        "invalid_adjudications": invalid_count,
        "invalidation_rate_pct": round(invalidation_rate_of_adjudicated, 2),
        "overall_game_error_rate_pct": round(overall_error_rate, 2),
        "truncated_games": truncated_count,
        "truncation_rate_pct": round((truncated_count / n) * 100.0, 2),
        "plies_saved_vs_baseline": plies_saved,
        "plies_saved_pct": round(plies_saved_pct, 2),
        "throughput_speedup_factor": round(speedup_factor, 3),
        "termination_reasons": dict(reasons),
    }


def main():
    print("=" * 84)
    print("Loop 8: Selfplay Early Adjudication Thresholds (Resign / Draw Cutoff Tradeoff)")
    print("=" * 84)

    openings_file = os.path.join(REPO_ROOT, "data", "openings_200.txt")
    openings_list: List[List[str]] = []
    if os.path.exists(openings_file):
        with open(openings_file, "r", encoding="utf-8") as f:
            openings_list = [line.strip().split() for line in f if line.strip()]
        print(f"Loaded {len(openings_list)} opening sequences from {openings_file}")
    else:
        print("Opening book not found, generating from scratch.")

    num_games = 200
    base_seed = 48000
    rng_openings = np.random.default_rng(20260921)

    tasks = []
    for i in range(num_games):
        opening = list(openings_list[rng_openings.integers(0, len(openings_list))]) if openings_list else None
        tasks.append((i, opening, base_seed + i))

    num_cpus = min(12, os.cpu_count() or 4)
    print(f"Simulating {num_games} selfplay trajectories using {num_cpus} worker processes...")
    t0 = time.time()

    from concurrent.futures import ProcessPoolExecutor, as_completed
    game_trajectories: List[Optional[Dict[str, Any]]] = [None] * num_games
    done_count = 0
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        future_to_idx = {executor.submit(simulate_full_game_trajectory, task): idx for idx, task in enumerate(tasks)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            res = future.result()
            game_trajectories[idx] = res
            done_count += 1
            if done_count % 20 == 0 or done_count == num_games:
                curr_t = time.time() - t0
                print(f"  [{done_count}/{num_games}] games completed ({curr_t:.1f}s, {done_count/curr_t:.2f} games/s)...")

    t1 = time.time()
    elapsed = t1 - t0
    print(f"\nAll {num_games} trajectory simulations completed in {elapsed:.2f}s ({num_games / elapsed:.2f} games/s)!")

    # Now evaluate policies across all 200 trajectories
    all_evaluations = [evaluate_policies_on_trajectory(g) for g in game_trajectories if g is not None]

    # Trajectory behavior inspection stats
    # 1. Overwhelming advantage: When Q > 0.95 or Q < -0.90 for 4+ consecutive plies,
    # how many plies did it take to actually finish?
    overwhelming_adv_plies_wasted: List[int] = []
    for g in game_trajectories:
        if g is None:
            continue
        traj = g["trajectory"]
        streak = 0
        first_trigger_ply = None
        for step in traj:
            if abs(step["q_white"]) > 0.90:
                streak += 1
                if streak >= 4 and first_trigger_ply is None:
                    first_trigger_ply = step["ply"]
            else:
                streak = 0
        if first_trigger_ply is not None:
            wasted = g["ground_truth_plies"] - first_trigger_ply
            overwhelming_adv_plies_wasted.append(wasted)

    # 2. Dead draw shuffling: |Q| < 0.05 for 20+ plies, how long until 50-move or termination?
    dead_draw_plies_wasted: List[int] = []
    for g in game_trajectories:
        if g is None:
            continue
        traj = g["trajectory"]
        streak = 0
        first_draw_ply = None
        for step in traj:
            if abs(step["q_white"]) < 0.05 and step["is_material_sym"] and not step["is_pawn_move"]:
                streak += 1
                if streak >= 12 and first_draw_ply is None:
                    first_draw_ply = step["ply"]
            else:
                streak = 0
        if first_draw_ply is not None:
            wasted = g["ground_truth_plies"] - first_draw_ply
            dead_draw_plies_wasted.append(wasted)

    # Baseline plies total
    baseline_results = [e["baseline"] for e in all_evaluations]
    baseline_plies_total = sum(r["plies"] for r in baseline_results)
    baseline_truncations = sum(1 for r in baseline_results if r["is_truncated"])

    metrics_baseline = analyze_policy_metrics("Baseline (No Early Adjudication)", baseline_results, baseline_plies_total)

    p1_results = [e["policy1_resign"] for e in all_evaluations]
    for i, r in enumerate(p1_results):
        r["is_truncated_baseline"] = baseline_results[i]["is_truncated"]
    metrics_p1 = analyze_policy_metrics("Policy 1 (Conservative Resignation)", p1_results, baseline_plies_total)

    p2_results = [e["policy2_draw"] for e in all_evaluations]
    for i, r in enumerate(p2_results):
        r["is_truncated_baseline"] = baseline_results[i]["is_truncated"]
    metrics_p2 = analyze_policy_metrics("Policy 2 (Conservative Early Draw)", p2_results, baseline_plies_total)

    p3_results = [e["policy3_combined"] for e in all_evaluations]
    for i, r in enumerate(p3_results):
        r["is_truncated_baseline"] = baseline_results[i]["is_truncated"]
    metrics_p3 = analyze_policy_metrics("Policy 3 (Combined Resign + Draw)", p3_results, baseline_plies_total)

    # Truncation reduction calculations
    p1_trunc_red = round(((baseline_truncations - metrics_p1["truncated_games"]) / max(1, baseline_truncations)) * 100.0, 2)
    p2_trunc_red = round(((baseline_truncations - metrics_p2["truncated_games"]) / max(1, baseline_truncations)) * 100.0, 2)
    p3_trunc_red = round(((baseline_truncations - metrics_p3["truncated_games"]) / max(1, baseline_truncations)) * 100.0, 2)

    # Behavior analysis summary
    behavior_analysis = {
        "overwhelming_advantage_waste": {
            "total_games_with_adv": len(overwhelming_adv_plies_wasted),
            "pct_of_games": round((len(overwhelming_adv_plies_wasted) / num_games) * 100.0, 2),
            "avg_plies_spent_converting": round(float(np.mean(overwhelming_adv_plies_wasted)), 2) if overwhelming_adv_plies_wasted else 0.0,
            "max_plies_spent_converting": int(np.max(overwhelming_adv_plies_wasted)) if overwhelming_adv_plies_wasted else 0,
            "total_plies_spent_converting": int(np.sum(overwhelming_adv_plies_wasted)) if overwhelming_adv_plies_wasted else 0,
        },
        "dead_draw_shuffling_waste": {
            "total_games_with_dead_draw": len(dead_draw_plies_wasted),
            "pct_of_games": round((len(dead_draw_plies_wasted) / num_games) * 100.0, 2),
            "avg_plies_spent_shuffling": round(float(np.mean(dead_draw_plies_wasted)), 2) if dead_draw_plies_wasted else 0.0,
            "max_plies_spent_shuffling": int(np.max(dead_draw_plies_wasted)) if dead_draw_plies_wasted else 0,
            "total_plies_spent_shuffling": int(np.sum(dead_draw_plies_wasted)) if dead_draw_plies_wasted else 0,
        },
    }

    # Save output to runs/loop8_adjudication.json
    runs_dir = os.path.join(REPO_ROOT, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    out_json_path = os.path.join(runs_dir, "loop8_adjudication.json")

    final_payload = {
        "metadata": {
            "task": "Loop 8: Selfplay Early Adjudication Thresholds (Resign / Draw Cutoff Tradeoff)",
            "date": "2026-09-21",
            "num_trajectories": num_games,
            "elapsed_seconds": round(elapsed, 2),
            "hardware_concurrency": num_cpus,
            "search_config": {
                "n_sims": N_SIMS,
                "m0": M0,
                "g": 1.0,
                "c_visit": C_VISIT,
                "c_scale": C_SCALE,
                "max_plies": 300,
            },
            "policies": {
                "baseline": "No early adjudication (play until checkmate, 50-move rule, 3-fold, or ply 300 cutoff)",
                "policy1_resign": "Resign if Q < -0.90 for 4 consecutive plies after ply 30",
                "policy2_draw": "Claim draw if |Q| < 0.05 and material symmetrical with no pawn moves for 12 consecutive plies after ply 40",
                "policy3_combined": "Combined Policy 1 (Resignation) + Policy 2 (Early Draw)",
            },
        },
        "behavior_analysis": behavior_analysis,
        "policy_metrics": {
            "baseline": metrics_baseline,
            "policy1_resign": metrics_p1,
            "policy2_draw": metrics_p2,
            "policy3_combined": metrics_p3,
        },
        "comparative_summary": {
            "throughput_gain_pct": {
                "policy1_resign": metrics_p1["plies_saved_pct"],
                "policy2_draw": metrics_p2["plies_saved_pct"],
                "policy3_combined": metrics_p3["plies_saved_pct"],
            },
            "speedup_factors": {
                "policy1_resign": metrics_p1["throughput_speedup_factor"],
                "policy2_draw": metrics_p2["throughput_speedup_factor"],
                "policy3_combined": metrics_p3["throughput_speedup_factor"],
            },
            "invalidation_rates_pct": {
                "policy1_resign": metrics_p1["invalidation_rate_pct"],
                "policy2_draw": metrics_p2["invalidation_rate_pct"],
                "policy3_combined": metrics_p3["invalidation_rate_pct"],
            },
            "truncation_elimination_pct": {
                "baseline_truncation_count": baseline_truncations,
                "policy1_reduction_pct": p1_trunc_red,
                "policy2_reduction_pct": p2_trunc_red,
                "policy3_reduction_pct": p3_trunc_red,
            },
        },
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\nSaved structured experimental results to {out_json_path}")
    print(f"File exists: {os.path.exists(out_json_path)} (size: {os.path.getsize(out_json_path)} bytes)")

    # Print Summary Table
    print("\n" + "=" * 98)
    print("LOOP 8: ADJUDICATION POLICY COMPARATIVE SUMMARY TABLE")
    print("=" * 98)
    fmt_hdr = "{:<32} | {:<12} | {:<12} | {:<12} | {:<12} | {:<8}"
    fmt_row = "{:<32} | {:<12} | {:<12} | {:<12} | {:<12} | {:<8}"
    print(fmt_hdr.format("Policy", "Avg Plies", "Adjudicated", "Error Rate", "Truncations", "Speedup"))
    print("-" * 98)

    pols = [
        ("Baseline (None)", metrics_baseline, 0.0),
        ("Policy 1 (Resign Q<-0.90)", metrics_p1, p1_trunc_red),
        ("Policy 2 (Draw |Q|<0.05)", metrics_p2, p2_trunc_red),
        ("Policy 3 (Combined 1+2)", metrics_p3, p3_trunc_red),
    ]

    for name, m, t_red in pols:
        avg_p = f"{m['avg_plies_per_game']:.1f}"
        adj_str = f"{m['adjudicated_games']}/{m['total_games']} ({m['adjudication_rate_pct']:.1f}%)"
        err_str = f"{m['invalid_adjudications']}/{max(1, m['adjudicated_games'])} ({m['invalidation_rate_pct']:.1f}%)"
        trunc_str = f"{m['truncated_games']} (-{t_red:.0f}%)" if m != metrics_baseline else f"{m['truncated_games']}"
        spd_str = f"x{m['throughput_speedup_factor']:.3f}"
        print(fmt_row.format(name, avg_p, adj_str, err_str, trunc_str, spd_str))

    print("=" * 98)

    print("\nInspection Findings:")
    print(f"1. Overwhelming Advantage Waste: {behavior_analysis['overwhelming_advantage_waste']['pct_of_games']}% of games entered Q > 0.90. After entering Q > 0.90, games wasted an average of {behavior_analysis['overwhelming_advantage_waste']['avg_plies_spent_converting']} plies before natural termination.")
    print(f"2. Dead Draw Shuffling Waste: {behavior_analysis['dead_draw_shuffling_waste']['pct_of_games']}% of games entered symmetric |Q| < 0.05 shuffling. Games lingered for an average of {behavior_analysis['dead_draw_shuffling_waste']['avg_plies_spent_shuffling']} plies waiting for 50-move rule or ply 300 cutoff.")
    print(f"3. Policy 3 Efficiency: Combined early resignation & draw adjudication saved {metrics_p3['plies_saved_vs_baseline']} total plies ({metrics_p3['plies_saved_pct']:.1f}% reduction), speeding up data generation by x{metrics_p3['throughput_speedup_factor']:.2f} with an invalidation/error rate of {metrics_p3['invalidation_rate_pct']}%.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
