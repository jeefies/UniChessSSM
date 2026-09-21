"""Empirical exploration of real chess dynamics on actual games and v3 shards.

Investigates across real positions from opening, middlegame, and endgame:
1. Branching factor & action support distribution in the 1936 action space.
2. Gumbel pi' scale dynamics on real positions across c_scale in [0.05, 0.1, 0.2, 1.0].
3. Soft Cross-Entropy loss & gradient scaling on real board layouts.
4. Saves structured results to runs/offline_exp_real_chess.json.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import chess
import chess.pgn
import numpy as np

# Add repo root to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stateseq.actions import (
    NUM_ACTIONS,
    NUM_KNIGHT_MOVES,
    NUM_PROMOTION_MOVES,
    NUM_QUEEN_MOVES,
    action_to_move,
    legal_mask,
    move_to_action,
)
from stateseq.data.gshards import V3ShardReader
from stateseq.gumbel import (
    C_VISIT,
    M0,
    N_SIMS,
    NEG_LOGIT,
    Node,
    export_pi_prime,
    order_halving,
    pi_prime,
    policy_probs,
    qtransform_completed,
    softmax,
)

C_SCALES = [0.05, 0.1, 0.2, 1.0]


def calc_entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def calc_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL divergence KL(p || q) = sum p * log(p / q)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def logsumexp_fp32(logits: np.ndarray, axis: int = -1, keepdims: bool = True) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    max_val = np.max(logits, axis=axis, keepdims=True)
    diff = logits - max_val
    exp_diff = np.exp(diff, dtype=np.float32)
    sum_exp = np.sum(exp_diff, axis=axis, keepdims=keepdims, dtype=np.float32)
    return max_val + np.log(np.maximum(sum_exp, 1e-37))


def soft_cross_entropy_fp32(
    logits: np.ndarray,
    target_probs: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Computes soft CE loss L = - sum target_probs * log_softmax(logits).
    
    Returns (loss, grad_logits) where grad_logits = softmax(logits) - target_probs.
    """
    logits = np.asarray(logits, dtype=np.float32)
    target_probs = np.asarray(target_probs, dtype=np.float32)
    
    lse = logsumexp_fp32(logits, axis=-1, keepdims=True)
    log_p = logits - lse
    probs = np.exp(log_p, dtype=np.float32)
    
    loss = -float(np.sum(target_probs * log_p))
    grad = probs - target_probs
    return loss, grad


def sample_real_positions(pgn_path: str, target_count: int = 500, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample diverse real positions across opening (ply 1..15), middlegame (16..45), and endgame (46..150)."""
    rng = np.random.default_rng(seed)
    
    with open(pgn_path, "r", encoding="utf-8") as f:
        games = []
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            if g.headers.get("Variant", "Standard").lower() != "standard":
                continue
            if g.headers.get("Result", "*") not in ("1-0", "0-1", "1/2-1/2"):
                continue
            moves = list(g.mainline_moves())
            if len(moves) >= 4:
                games.append((g, moves))
    
    print(f"Loaded {len(games)} valid games from {pgn_path}.")
    
    # Collect positions with phase annotation
    # Opening: ply 1..15, Middlegame: ply 16..45, Endgame: ply 46..150
    opening_pool = []
    middlegame_pool = []
    endgame_pool = []
    
    for g_idx, (g, moves) in enumerate(games):
        board = g.board()
        for ply, mv in enumerate(moves, start=1):
            if ply > 150:
                break
            if not board.is_game_over() and board.legal_moves.count() > 0:
                pos_info = {
                    "game_idx": g_idx,
                    "ply": ply,
                    "fen": board.fen(),
                    "turn": board.turn,
                    "legal_moves_count": board.legal_moves.count(),
                    "board": board.copy(),
                }
                if ply <= 15:
                    opening_pool.append(pos_info)
                elif ply <= 45:
                    middlegame_pool.append(pos_info)
                else:
                    endgame_pool.append(pos_info)
            board.push(mv)
            
    print(f"Position pool: opening={len(opening_pool)}, middlegame={len(middlegame_pool)}, endgame={len(endgame_pool)}")
    
    # Stratified sampling
    n_open = min(len(opening_pool), int(target_count * 0.30))
    n_end = min(len(endgame_pool), int(target_count * 0.35))
    n_mid = target_count - n_open - n_end
    if n_mid > len(middlegame_pool):
        n_mid = len(middlegame_pool)
        
    sampled = []
    if opening_pool:
        idx_open = rng.choice(len(opening_pool), size=n_open, replace=False)
        for i in idx_open:
            p = opening_pool[i]
            p["phase"] = "opening"
            sampled.append(p)
    if middlegame_pool:
        idx_mid = rng.choice(len(middlegame_pool), size=n_mid, replace=False)
        for i in idx_mid:
            p = middlegame_pool[i]
            p["phase"] = "middlegame"
            sampled.append(p)
    if endgame_pool:
        idx_end = rng.choice(len(endgame_pool), size=n_end, replace=False)
        for i in idx_end:
            p = endgame_pool[i]
            p["phase"] = "endgame"
            sampled.append(p)
            
    # Sort by game_idx and ply
    sampled.sort(key=lambda x: (x["game_idx"], x["ply"]))
    print(f"Sampled {len(sampled)} real positions (open: {n_open}, mid: {n_mid}, end: {n_end}).")
    return sampled


def analyze_branching_and_action_support(
    positions: List[Dict[str, Any]],
    shard_reader: V3ShardReader,
) -> Dict[str, Any]:
    """1. Branching Factor & Action Support Distribution."""
    legal_counts = [p["legal_moves_count"] for p in positions]
    quantiles = {
        "min": int(np.min(legal_counts)),
        "p25": float(np.percentile(legal_counts, 25)),
        "p50": float(np.percentile(legal_counts, 50)),
        "p75": float(np.percentile(legal_counts, 75)),
        "p90": float(np.percentile(legal_counts, 90)),
        "max": int(np.max(legal_counts)),
        "mean": float(np.mean(legal_counts)),
        "std": float(np.std(legal_counts)),
    }
    
    # By phase
    phase_quantiles = {}
    for phase in ["opening", "middlegame", "endgame"]:
        sub_counts = [p["legal_moves_count"] for p in positions if p["phase"] == phase]
        if sub_counts:
            phase_quantiles[phase] = {
                "count": len(sub_counts),
                "min": int(np.min(sub_counts)),
                "p25": float(np.percentile(sub_counts, 25)),
                "p50": float(np.percentile(sub_counts, 50)),
                "p75": float(np.percentile(sub_counts, 75)),
                "p90": float(np.percentile(sub_counts, 90)),
                "max": int(np.max(sub_counts)),
                "mean": float(np.mean(sub_counts)),
            }
            
    # Action ID distribution across the 1936 space
    action_counts_played = np.zeros(NUM_ACTIONS, dtype=np.int64)
    action_counts_legal = np.zeros(NUM_ACTIONS, dtype=np.int64)
    
    # Count played moves from shard
    num_games_shard = len(shard_reader.meta_all)
    for g_idx in range(num_games_shard):
        rec = shard_reader.game(g_idx)
        acts = rec["actions"]
        for a in acts:
            action_counts_played[a] += 1
            
    # Count legal move occurrences across sampled positions
    for p in positions:
        b = p["board"]
        mask = legal_mask(b)
        legal_ids = np.flatnonzero(mask)
        action_counts_legal[legal_ids] += 1
        
    # Categorize into Queen (0..1455), Knight (1456..1791), Promotion (1792..1935)
    def region_stats(counts: np.ndarray) -> Dict[str, Any]:
        q_cnt = int(np.sum(counts[0:NUM_QUEEN_MOVES]))
        k_cnt = int(np.sum(counts[NUM_QUEEN_MOVES : NUM_QUEEN_MOVES + NUM_KNIGHT_MOVES]))
        p_cnt = int(np.sum(counts[NUM_QUEEN_MOVES + NUM_KNIGHT_MOVES : NUM_ACTIONS]))
        
        q_act = int(np.count_nonzero(counts[0:NUM_QUEEN_MOVES]))
        k_act = int(np.count_nonzero(counts[NUM_QUEEN_MOVES : NUM_QUEEN_MOVES + NUM_KNIGHT_MOVES]))
        p_act = int(np.count_nonzero(counts[NUM_QUEEN_MOVES + NUM_KNIGHT_MOVES : NUM_ACTIONS]))
        
        tot = int(np.sum(counts))
        return {
            "queen": {"total_events": q_cnt, "active_actions": q_act, "pct_events": q_cnt / max(1, tot) * 100, "capacity": NUM_QUEEN_MOVES},
            "knight": {"total_events": k_cnt, "active_actions": k_act, "pct_events": k_cnt / max(1, tot) * 100, "capacity": NUM_KNIGHT_MOVES},
            "promotion": {"total_events": p_cnt, "active_actions": p_act, "pct_events": p_cnt / max(1, tot) * 100, "capacity": NUM_PROMOTION_MOVES},
            "total_events": tot,
            "total_active_actions": int(np.count_nonzero(counts)),
            "coverage_pct": float(np.count_nonzero(counts) / NUM_ACTIONS * 100),
        }
        
    return {
        "branching_overall": quantiles,
        "branching_by_phase": phase_quantiles,
        "played_actions_support": region_stats(action_counts_played),
        "legal_actions_support": region_stats(action_counts_legal),
    }


def analyze_gumbel_scale_dynamics_on_real_positions(
    positions: List[Dict[str, Any]],
    seed: int = 12345,
) -> Dict[str, Any]:
    """2. Gumbel pi' Scale Dynamics on Real Positions.
    
    Across c_scale in [0.05, 0.1, 0.2, 1.0].
    """
    rng = np.random.default_rng(seed)
    results_by_c_scale = {cs: [] for cs in C_SCALES}
    
    # Track statistics across all positions for each c_scale
    for p_idx, pos in enumerate(positions):
        board = pos["board"]
        mask = legal_mask(board)
        legal_ids = np.flatnonzero(mask)
        num_legal = len(legal_ids)
        if num_legal == 0:
            continue
            
        # Simulate realistic network policy prior: Shannon entropy around 2.0 .. 3.0
        # Realistic policy logits: top moves have higher logits, followed by decaying tail
        raw_prior_logits = rng.gumbel(loc=0.0, scale=1.0, size=num_legal).astype(np.float32)
        # Add slight exponential spread to create realistic entropy
        perm = rng.permutation(num_legal)
        raw_prior_logits[perm] += np.linspace(2.5, 0.0, num_legal, dtype=np.float32)
        prior_p = softmax(raw_prior_logits)
        prior_ent = calc_entropy(prior_p)
        
        # Simulate realistic child evaluations Q in [-1, 1]
        # In real play, root Q is around pos eval, child moves have varying quality:
        # 1-3 strong moves, several mediocre, some blunders
        child_qs = np.zeros(num_legal, dtype=np.float32)
        root_v = rng.uniform(-0.3, 0.3)
        # Generate spread: best move ~ root_v + 0.1, others downward
        child_qs = root_v - rng.exponential(scale=0.25, size=num_legal).astype(np.float32)
        best_child_idx = int(perm[0]) if num_legal > 0 else 0
        child_qs[best_child_idx] = max(child_qs[best_child_idx], root_v + rng.uniform(0.05, 0.2))
        child_qs = np.clip(child_qs, -1.0, 1.0)
        
        # Create a search mock with order_halving
        # We define expand function returning child nodes with evaluated child_qs
        for cs in C_SCALES:
            root_node = Node(
                q=float(root_v),
                legal=legal_ids.astype(np.int64),
                logits=raw_prior_logits.copy(),
            )
            
            def expand_fn(parent: Node, act: int) -> Node:
                if parent.depth == 0:
                    idx = int(np.flatnonzero(legal_ids == act)[0])
                    val = float(child_qs[idx])
                    # child's perspective is -val
                    return Node(
                        legal=np.array([9999], dtype=np.int64),
                        logits=np.array([0.0], dtype=np.float32),
                        q=-val,
                        depth=1,
                        action=act,
                        path=(act,),
                        terminal=True,  # 1-ply evaluation simulation
                    )
                else:
                    return Node(
                        legal=np.array([], dtype=np.int64),
                        logits=np.array([], dtype=np.float32),
                        q=0.0,
                        depth=parent.depth + 1,
                        action=act,
                        path=parent.path + (act,),
                        terminal=True,
                    )
                
            res = order_halving(
                root=root_node,
                expand=expand_fn,
                n_sims=N_SIMS,
                m0=M0,
                seed=rng,
                c_visit=C_VISIT,
                c_scale=cs,
                g=1.0,
            )
            
            # Export pi'
            pi_p = pi_prime(root_node, c_visit=C_VISIT, c_scale=cs)
            pi_ent = calc_entropy(pi_p)
            kl = calc_kl(pi_p, prior_p)
            
            # Delta sigma span
            s_vec = qtransform_completed(root_node, c_visit=C_VISIT, c_scale=cs)
            delta_sigma = float(s_vec.max() - s_vec.min()) if len(s_vec) > 0 else 0.0
            
            # Target collapse: max mass > 0.95 or pi_ent < 0.05
            is_collapsed = (np.max(pi_p) > 0.95) or (pi_ent < 0.05)
            
            results_by_c_scale[cs].append({
                "phase": pos["phase"],
                "num_legal": num_legal,
                "prior_entropy": prior_ent,
                "pi_entropy": pi_ent,
                "kl": kl,
                "delta_sigma": delta_sigma,
                "max_p": float(np.max(pi_p)),
                "is_collapsed": bool(is_collapsed),
            })
            
    # Aggregate stats per c_scale
    summary = {}
    for cs, data_list in results_by_c_scale.items():
        pi_ents = [d["pi_entropy"] for d in data_list]
        prior_ents = [d["prior_entropy"] for d in data_list]
        kls = [d["kl"] for d in data_list]
        delta_sigmas = [d["delta_sigma"] for d in data_list]
        max_ps = [d["max_p"] for d in data_list]
        collapsed = [d["is_collapsed"] for d in data_list]
        
        # Breakdown by phase
        phase_breakdown = {}
        for ph in ["opening", "middlegame", "endgame"]:
            ph_sub = [d for d in data_list if d["phase"] == ph]
            if ph_sub:
                phase_breakdown[ph] = {
                    "count": len(ph_sub),
                    "mean_pi_entropy": float(np.mean([d["pi_entropy"] for d in ph_sub])),
                    "mean_kl": float(np.mean([d["kl"] for d in ph_sub])),
                    "mean_delta_sigma": float(np.mean([d["delta_sigma"] for d in ph_sub])),
                    "collapse_rate_pct": float(np.mean([d["is_collapsed"] for d in ph_sub]) * 100),
                }
                
        summary[str(cs)] = {
            "c_scale": cs,
            "mean_prior_entropy": float(np.mean(prior_ents)),
            "mean_pi_entropy": float(np.mean(pi_ents)),
            "std_pi_entropy": float(np.std(pi_ents)),
            "mean_kl": float(np.mean(kls)),
            "p50_kl": float(np.median(kls)),
            "p90_kl": float(np.percentile(kls, 90)),
            "mean_delta_sigma": float(np.mean(delta_sigmas)),
            "mean_max_p": float(np.mean(max_ps)),
            "collapse_rate_pct": float(np.mean(collapsed) * 100),
            "by_phase": phase_breakdown,
        }
        
    return summary


def analyze_soft_ce_and_gradients(
    positions: List[Dict[str, Any]],
    seed: int = 54321,
) -> Dict[str, Any]:
    """3. Soft Cross-Entropy Loss & Gradient Scaling on Real Board Layouts."""
    rng = np.random.default_rng(seed)
    
    grad_records = []
    
    for pos in positions:
        board = pos["board"]
        mask = legal_mask(board)
        legal_ids = np.flatnonzero(mask)
        num_legal = len(legal_ids)
        if num_legal == 0:
            continue
            
        # Real full 1936-dim logits
        full_logits = np.full(NUM_ACTIONS, NEG_LOGIT, dtype=np.float32)
        
        # Policy logits on legal moves: reasonable variation
        legal_logits = rng.normal(loc=0.0, scale=1.5, size=num_legal).astype(np.float32)
        full_logits[legal_ids] = legal_logits
        
        # Target distribution: realistic pi' on legal actions (sum = 1)
        # Using soft distribution favoring a few best moves
        raw_target = np.exp(rng.normal(loc=0.0, scale=1.0, size=num_legal).astype(np.float32))
        target_legal_p = (raw_target / np.sum(raw_target)).astype(np.float32)
        
        full_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        full_target[legal_ids] = target_legal_p
        
        # Compute soft CE loss and grad
        loss, grad = soft_cross_entropy_fp32(full_logits, full_target)
        
        # Analyze grad:
        # Illegal moves: where mask == 0
        illegal_mask = (mask == 0)
        grad_illegal = grad[illegal_mask]
        max_abs_grad_illegal = float(np.max(np.abs(grad_illegal)))
        l2_grad_illegal = float(np.linalg.norm(grad_illegal))
        
        # Legal moves grad
        grad_legal = grad[legal_ids]
        l2_grad_legal = float(np.linalg.norm(grad_legal))
        
        # Top move (highest target prob) vs non-top legal moves
        top_idx_within_legal = int(np.argmax(target_legal_p))
        top_act_id = legal_ids[top_idx_within_legal]
        grad_top = float(abs(grad[top_act_id]))
        
        other_legal_mask = np.ones(num_legal, dtype=bool)
        other_legal_mask[top_idx_within_legal] = False
        grad_other_mean = float(np.mean(np.abs(grad_legal[other_legal_mask]))) if num_legal > 1 else 0.0
        
        grad_records.append({
            "phase": pos["phase"],
            "num_legal": num_legal,
            "loss": loss,
            "l2_grad_total": float(np.linalg.norm(grad)),
            "l2_grad_legal": l2_grad_legal,
            "l2_grad_illegal": l2_grad_illegal,
            "max_abs_grad_illegal": max_abs_grad_illegal,
            "grad_top_move": grad_top,
            "grad_other_legal_mean": grad_other_mean,
            "snr_top_to_other": float(grad_top / max(1e-8, grad_other_mean)),
        })
        
    # Group by legal move count bins
    # Bins: [1..5] (pawn endgame), [6..15], [16..30], [31..45], [46+] (open middlegame)
    bins = [
        ("2..5 moves (endgame)", lambda n: 2 <= n <= 5),
        ("6..15 moves (restricted)", lambda n: 6 <= n <= 15),
        ("16..30 moves (standard)", lambda n: 16 <= n <= 30),
        ("31..45 moves (open/active)", lambda n: 31 <= n <= 45),
        ("46+ moves (tactical chaos)", lambda n: n >= 46),
    ]
    
    bin_stats = {}
    for label, fn in bins:
        sub = [r for r in grad_records if fn(r["num_legal"])]
        if sub:
            bin_stats[label] = {
                "count": len(sub),
                "mean_legal_moves": float(np.mean([r["num_legal"] for r in sub])),
                "mean_loss": float(np.mean([r["loss"] for r in sub])),
                "mean_l2_grad": float(np.mean([r["l2_grad_total"] for r in sub])),
                "std_l2_grad": float(np.std([r["l2_grad_total"] for r in sub])),
                "max_illegal_grad": float(np.max([r["max_abs_grad_illegal"] for r in sub])),
                "mean_snr": float(np.mean([r["snr_top_to_other"] for r in sub])),
            }
            
    overall_stats = {
        "total_positions_evaluated": len(grad_records),
        "mean_loss": float(np.mean([r["loss"] for r in grad_records])),
        "mean_l2_grad": float(np.mean([r["l2_grad_total"] for r in grad_records])),
        "max_l2_grad_illegal": float(np.max([r["l2_grad_illegal"] for r in grad_records])),
        "max_single_illegal_grad": float(np.max([r["max_abs_grad_illegal"] for r in grad_records])),
        "bins": bin_stats,
    }
    return overall_stats


def main():
    pgn_path = os.path.join(REPO_ROOT, "data", "sample_real.pgn")
    shard_dir = os.path.join(REPO_ROOT, "data", "shards_real_v3")
    out_json = os.path.join(REPO_ROOT, "runs", "offline_exp_real_chess.json")
    
    print("================================================================================")
    print("EXP: Real Chess Dynamics on Actual Positions & V3 Shards")
    print("================================================================================")
    
    # 1. Sample real positions
    positions = sample_real_positions(pgn_path, target_count=500, seed=42)
    shard_reader = V3ShardReader(shard_dir)
    
    # 2. Branching and Action Support
    print("\n[1/3] Analyzing Branching Factor and Action Support Distribution...")
    branching_data = analyze_branching_and_action_support(positions, shard_reader)
    
    # 3. Gumbel scale dynamics
    print("\n[2/3] Simulating Gumbel pi' Scale Dynamics across c_scale on Real Boards...")
    gumbel_data = analyze_gumbel_scale_dynamics_on_real_positions(positions, seed=12345)
    
    # 4. Soft CE Loss and Gradients
    print("\n[3/3] Analyzing Soft CE Loss and Gradients across Real Move Branchings...")
    loss_data = analyze_soft_ce_and_gradients(positions, seed=54321)
    
    # Save structured JSON
    final_output = {
        "metadata": {
            "pgn_source": "data/sample_real.pgn",
            "shards_source": "data/shards_real_v3",
            "positions_sampled": len(positions),
            "c_scales_tested": C_SCALES,
        },
        "branching_and_actions": branching_data,
        "gumbel_scale_dynamics": gumbel_data,
        "soft_ce_and_gradients": loss_data,
    }
    
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved structured experimental results to {out_json}")
    
    # Print formatted summary tables
    print("\n" + "=" * 80)
    print("TABLE 1: BRANCHING FACTOR DISTRIBUTIONS ACROSS REAL PHASES")
    print("=" * 80)
    print(f"{'Phase':<15} {'Count':<8} {'Min':<6} {'P25':<6} {'P50':<6} {'P75':<6} {'P90':<6} {'Max':<6} {'Mean':<6}")
    print("-" * 80)
    ov = branching_data["branching_overall"]
    print(f"{'Overall':<15} {len(positions):<8} {ov['min']:<6} {ov['p25']:<6.1f} {ov['p50']:<6.1f} {ov['p75']:<6.1f} {ov['p90']:<6.1f} {ov['max']:<6} {ov['mean']:<6.1f}")
    for ph, q in branching_data["branching_by_phase"].items():
        print(f"{ph.capitalize():<15} {q['count']:<8} {q['min']:<6} {q['p25']:<6.1f} {q['p50']:<6.1f} {q['p75']:<6.1f} {q['p90']:<6.1f} {q['max']:<6} {q['mean']:<6.1f}")
        
    print("\n" + "=" * 80)
    print("TABLE 2: ACTION SPACE COVERAGE & COMPOSITION (1936 TOTAL)")
    print("=" * 80)
    print(f"{'Move Category':<15} {'Capacity':<10} {'Active (Played)':<16} {'Coverage %':<12} {'Pct Events':<12}")
    print("-" * 80)
    ps = branching_data["played_actions_support"]
    for cat in ["queen", "knight", "promotion"]:
        c_info = ps[cat]
        print(f"{cat.capitalize():<15} {c_info['capacity']:<10} {c_info['active_actions']:<16} {c_info['active_actions']/c_info['capacity']*100:<12.1f} {c_info['pct_events']:<12.1f}%")
    print(f"{'Total':<15} {1936:<10} {ps['total_active_actions']:<16} {ps['coverage_pct']:<12.1f} 100.0%")
    
    print("\n" + "=" * 80)
    print("TABLE 3: GUMBEL pi' DYNAMICS ACROSS c_scale ON REAL POSITIONS")
    print("=" * 80)
    print(f"{'c_scale':<8} {'Prior H':<10} {'pi_prime H':<12} {'KL(pi_prime||pi)':<18} {'Delta_sigma':<14} {'Max pi_prime':<14} {'Collapse %':<10}")
    print("-" * 80)
    for cs in C_SCALES:
        s = gumbel_data[str(cs)]
        print(f"{cs:<8.2f} {s['mean_prior_entropy']:<10.3f} {s['mean_pi_entropy']:<12.3f} {s['mean_kl']:<18.3f} {s['mean_delta_sigma']:<14.2f} {s['mean_max_p']:<14.3f} {s['collapse_rate_pct']:<10.1f}%")
        
    print("\n" + "=" * 80)
    print("TABLE 4: SOFT CE LOSS & GRADIENTS ACROSS BRANCHING FACTOR BINS")
    print("=" * 80)
    print(f"{'Branching Bin':<28} {'Count':<8} {'Mean Moves':<12} {'Loss':<10} {'L2 Grad':<12} {'Max Illegal Grad':<18} {'SNR (Top/Other)':<15}")
    print("-" * 80)
    for label, b_info in loss_data["bins"].items():
        print(f"{label:<28} {b_info['count']:<8} {b_info['mean_legal_moves']:<12.1f} {b_info['mean_loss']:<10.3f} {b_info['mean_l2_grad']:<12.3f} {b_info['max_illegal_grad']:<18.2e} {b_info['mean_snr']:<15.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
