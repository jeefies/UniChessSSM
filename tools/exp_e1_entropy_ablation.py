"""Experiment E1: Policy Entropy Regulation & Target Annealing on Real Chess Positions.

Evaluates mechanisms to control policy target entropy across real chess positions:
Phase partitions from data/sample_real.pgn:
  - Opening: ply 1-15
  - Middlegame: ply 16-45
  - Endgame: ply 46-150

Mechanisms evaluated across 300+ diverse real positions:
1. Mechanism A: Target temperature softening:
   pi'^{(tau)} = Softmax(log pi' / tau) for tau in [0.8, 1.0, 1.25, 1.5] under fixed c_scale=0.1.
2. Mechanism B: Loss-level entropy bonus:
   L = L_{soft_CE} - lambda * H(pi_theta) for lambda in [0.0, 1e-4, 1e-3, 1e-2] under fixed c_scale=0.1.
3. Mechanism C: Adaptive c_scale(t):
   - Step schedule: opening c_scale=0.05, middlegame c_scale=0.10, endgame c_scale=0.20.
   - Baseline: fixed c_scale=0.10.

Metrics per configuration across positions:
  - Empirical target entropy H(pi') in nats.
  - Target collapse rate (max pi' > 0.95 or H(pi') < 0.05).
  - Top-1 move preservation rate (does argmax pi' agree with search-selected survivor action?).
  - L2 gradient norm ||grad_z||_2.
  - Directional stability w.r.t the true best move:
    * pull on best move: -grad_z[best_move] = (target - p_theta)[best_move]
    * cosine similarity to ideal one-hot target gradient
  - Numerical safety: verification of zero NaNs and zero Infs.

Saves structured results to runs/offline_exp_e1_entropy.json.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import chess
import chess.pgn
import numpy as np

# Repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stateseq.actions import legal_mask
from stateseq.gumbel import (
    C_SCALE,
    C_VISIT,
    M0,
    N_SIMS,
    NEG_LOGIT,
    Node,
    export_pi_prime,
    order_halving,
    pi_prime,
    qtransform_completed,
    softmax,
)

# Configurations
TAU_LIST = [0.8, 1.0, 1.25, 1.5]
LAMBDA_LIST = [0.0, 1e-4, 1e-3, 1e-2]
ADAPTIVE_C_SCALE = {
    "opening": 0.05,
    "middlegame": 0.10,
    "endgame": 0.20,
}


def entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL divergence KL(p || q)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def soften_distribution(p: np.ndarray, tau: float, eps: float = 1e-12) -> np.ndarray:
    """Softens probability distribution: p^(tau) = Softmax(log p / tau)."""
    p = np.asarray(p, dtype=np.float64)
    if tau == 1.0:
        return p.copy().astype(np.float32)
    p_safe = np.clip(p, eps, 1.0)
    logits = np.log(p_safe) / float(tau)
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    res = exp_logits / np.sum(exp_logits)
    return res.astype(np.float32)


def sample_real_positions(pgn_path: str, target_count: int = 360, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample diverse real positions across opening (1-15), middlegame (16-45), and endgame (46-150)."""
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
            if len(moves) >= 6:
                games.append((g, moves))
                
    print(f"Loaded {len(games)} valid games from {pgn_path}.")
    
    opening_pool: List[Dict[str, Any]] = []
    middlegame_pool: List[Dict[str, Any]] = []
    endgame_pool: List[Dict[str, Any]] = []
    
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
            
    print(f"Position candidate pool: Opening={len(opening_pool)}, Middlegame={len(middlegame_pool)}, Endgame={len(endgame_pool)}")
    
    # Balanced stratified sampling: 1/3 per phase
    per_phase = target_count // 3
    n_open = min(len(opening_pool), per_phase)
    n_mid = min(len(middlegame_pool), per_phase)
    n_end = min(len(endgame_pool), per_phase)
    
    sampled: List[Dict[str, Any]] = []
    for pool, n, ph in [(opening_pool, n_open, "opening"), (middlegame_pool, n_mid, "middlegame"), (endgame_pool, n_end, "endgame")]:
        chosen_indices = rng.choice(len(pool), size=n, replace=False)
        for idx in chosen_indices:
            item = pool[idx]
            item["phase"] = ph
            sampled.append(item)
            
    sampled.sort(key=lambda x: (x["game_idx"], x["ply"]))
    print(f"Successfully sampled {len(sampled)} real positions (Opening: {n_open}, Middlegame: {n_mid}, Endgame: {n_end}).")
    return sampled


def simulate_real_position_search(
    pos: Dict[str, Any],
    c_scale: float,
    seed_val: int,
) -> Dict[str, Any]:
    """Runs a realistic sequential halving search on a real chess position."""
    rng = np.random.default_rng(seed_val)
    board: chess.Board = pos["board"]
    mask = legal_mask(board)
    legal_ids = np.flatnonzero(mask)
    num_legal = len(legal_ids)
    
    # Realistic prior policy logits (entropy ~ 2.0 - 2.8)
    # Permute and assign decaying scores
    perm = rng.permutation(num_legal)
    raw_logits = rng.gumbel(loc=0.0, scale=0.8, size=num_legal).astype(np.float32)
    raw_logits[perm] += np.linspace(2.2, 0.0, num_legal, dtype=np.float32)
    prior_probs = softmax(raw_logits)
    
    # Realistic child Q-values in [-1, 1]
    root_q = float(rng.uniform(-0.25, 0.25))
    child_qs = root_q - rng.exponential(scale=0.22, size=num_legal).astype(np.float32)
    best_child_local_idx = int(perm[0])
    child_qs[best_child_local_idx] = max(child_qs[best_child_local_idx], root_q + rng.uniform(0.08, 0.25))
    child_qs = np.clip(child_qs, -1.0, 1.0)
    
    # Root node
    root = Node(
        legal=legal_ids.astype(np.int64),
        logits=raw_logits.copy(),
        q=root_q,
    )
    
    def expand_fn(parent: Node, act: int) -> Node:
        if parent.depth == 0:
            idx = int(np.flatnonzero(legal_ids == act)[0])
            val = float(child_qs[idx])
            return Node(
                legal=np.array([9999], dtype=np.int64),
                logits=np.array([0.0], dtype=np.float32),
                q=-val,
                depth=1,
                action=act,
                path=(act,),
                terminal=True,
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
            
    search_out = order_halving(
        root=root,
        expand=expand_fn,
        n_sims=N_SIMS,
        m0=M0,
        seed=rng,
        c_visit=C_VISIT,
        c_scale=c_scale,
        g=1.0,  # Standard Gumbel exploration
    )
    
    pi_p = pi_prime(root, c_visit=C_VISIT, c_scale=c_scale)
    
    return {
        "legal_ids": legal_ids,
        "num_legal": num_legal,
        "prior_logits": raw_logits,
        "prior_probs": prior_probs,
        "child_qs": child_qs,
        "root": root,
        "search_action": search_out["action"],
        "pi_prime": pi_p,
        "best_child_local_idx": best_child_local_idx,
        "best_child_action": int(legal_ids[best_child_local_idx]),
    }


def compute_gradient_metrics(
    prior_probs: np.ndarray,
    target_probs: np.ndarray,
    entropy_lambda: float,
    search_action: int,
    legal_ids: np.ndarray,
) -> Dict[str, Any]:
    """Computes gradient dynamics w.r.t logits z on legal moves.
    
    dL / dz = (pi_theta - pi_target) - lambda * [pi_theta * ( -log pi_theta - H(pi_theta) )]
            = (pi_theta - pi_target) + lambda * pi_theta * (log pi_theta + H)
    """
    p = np.asarray(prior_probs, dtype=np.float32)
    t = np.asarray(target_probs, dtype=np.float32)
    
    # Soft CE gradient
    grad_ce = p - t
    
    # Entropy bonus gradient:
    # H(p) = - sum p log p
    # dH / dz_i = sum_j (dH / dp_j) * (dp_j / dz_i)
    #           = sum_j (-1 - log p_j) * p_j * (delta_ij - p_i)
    #           = -p_i (1 + log p_i) - p_i * sum_j (-1 - log p_j) p_j
    #           = -p_i (1 + log p_i) + p_i (1 - H(p))
    #           = -p_i (log p_i + H(p))
    # Since L = L_CE - lambda * H(p), d(-lambda * H) / dz_i = + lambda * p_i (log p_i + H(p))
    H = entropy(p)
    log_p = np.log(np.maximum(p, 1e-12))
    grad_entropy = lambda_val = float(entropy_lambda) * p * (log_p + H)
    
    grad_total = grad_ce + grad_entropy
    l2_norm = float(np.linalg.norm(grad_total))
    
    # Find local index of search best action
    action_matches = np.flatnonzero(legal_ids == search_action)
    best_idx = int(action_matches[0]) if len(action_matches) > 0 else int(np.argmax(t))
    
    # Pull on search best move: parameter update is z <- z - lr * grad
    # The effective push on logit z[best] is -grad_total[best]
    pull_best = float(-grad_total[best_idx])
    
    # Directional stability: cosine similarity between grad_total and ideal one-hot grad (p - 1_best)
    ideal_grad = p.copy()
    ideal_grad[best_idx] -= 1.0
    denom = (np.linalg.norm(grad_total) * np.linalg.norm(ideal_grad)) + 1e-12
    cos_sim = float(np.dot(grad_total, ideal_grad) / denom)
    
    # Numerical safety
    is_safe = bool(np.all(np.isfinite(grad_total)) and np.all(np.isfinite(t)))
    
    return {
        "l2_norm": l2_norm,
        "pull_best": pull_best,
        "cos_sim": cos_sim,
        "is_safe": is_safe,
    }


def run_experiment_e1():
    pgn_path = os.path.join(REPO_ROOT, "data", "sample_real.pgn")
    out_json = os.path.join(REPO_ROOT, "runs", "offline_exp_e1_entropy.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    
    print("=" * 80)
    print("EXPERIMENT E1: Policy Entropy Regulation & Target Annealing on Real Positions")
    print("=" * 80)
    
    positions = sample_real_positions(pgn_path, target_count=360, seed=42)
    n_pos = len(positions)
    assert n_pos >= 300, f"Insufficient positions: {n_pos} < 300"
    
    # Phase groups
    phase_indices = {
        "opening": [i for i, p in enumerate(positions) if p["phase"] == "opening"],
        "middlegame": [i for i, p in enumerate(positions) if p["phase"] == "middlegame"],
        "endgame": [i for i, p in enumerate(positions) if p["phase"] == "endgame"],
    }
    
    # Base search run under fixed c_scale=0.1
    print("\n[Step 1/4] Running sequential halving baseline (c_scale=0.10) across all positions...")
    searches_baseline: List[Dict[str, Any]] = []
    for i, pos in enumerate(positions):
        s_res = simulate_real_position_search(pos, c_scale=0.10, seed_val=10000 + i)
        searches_baseline.append(s_res)
        
    # Searches for Mechanism C adaptive c_scale
    print("[Step 2/4] Running sequential halving with adaptive c_scale schedule...")
    searches_adaptive: List[Dict[str, Any]] = []
    for i, pos in enumerate(positions):
        cs_val = ADAPTIVE_C_SCALE[pos["phase"]]
        s_res = simulate_real_position_search(pos, c_scale=cs_val, seed_val=10000 + i)
        searches_adaptive.append(s_res)
        
    print("\n[Step 3/4] Evaluating Mechanisms A, B, and C...")
    
    # -------------------------------------------------------------
    # Mechanism A: Target Temperature Softening tau in [0.8, 1.0, 1.25, 1.5]
    # -------------------------------------------------------------
    mech_a_results: Dict[str, Any] = {}
    for tau in TAU_LIST:
        key = f"tau_{tau}"
        pos_records = []
        for i, pos in enumerate(positions):
            s = searches_baseline[i]
            pi_raw = s["pi_prime"]
            pi_soft = soften_distribution(pi_raw, tau)
            
            ent = entropy(pi_soft)
            max_p = float(np.max(pi_soft))
            is_collapsed = bool(max_p > 0.95 or ent < 0.05)
            
            # Top-1 move preservation
            # Does argmax pi_soft match search-selected action?
            target_top1_action = int(s["legal_ids"][np.argmax(pi_soft)])
            preserved = bool(target_top1_action == s["search_action"])
            
            grad_met = compute_gradient_metrics(
                prior_probs=s["prior_probs"],
                target_probs=pi_soft,
                entropy_lambda=0.0,
                search_action=s["search_action"],
                legal_ids=s["legal_ids"],
            )
            
            pos_records.append({
                "phase": pos["phase"],
                "entropy": ent,
                "max_p": max_p,
                "is_collapsed": is_collapsed,
                "top1_preserved": preserved,
                "l2_norm": grad_met["l2_norm"],
                "pull_best": grad_met["pull_best"],
                "cos_sim": grad_met["cos_sim"],
                "is_safe": grad_met["is_safe"],
            })
            
        # Aggregate overall & by phase
        def agg(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {
                "count": len(recs),
                "mean_entropy": float(np.mean([r["entropy"] for r in recs])),
                "std_entropy": float(np.std([r["entropy"] for r in recs])),
                "collapse_rate_pct": float(np.mean([r["is_collapsed"] for r in recs]) * 100),
                "top1_preservation_pct": float(np.mean([r["top1_preserved"] for r in recs]) * 100),
                "mean_max_prob": float(np.mean([r["max_p"] for r in recs])),
                "mean_l2_grad": float(np.mean([r["l2_norm"] for r in recs])),
                "mean_pull_best": float(np.mean([r["pull_best"] for r in recs])),
                "mean_cos_sim": float(np.mean([r["cos_sim"] for r in recs])),
                "all_safe": bool(all(r["is_safe"] for r in recs)),
            }
            
        mech_a_results[key] = {
            "tau": tau,
            "overall": agg(pos_records),
            "by_phase": {ph: agg([r for r in pos_records if r["phase"] == ph]) for ph in ["opening", "middlegame", "endgame"]},
        }
        
    # -------------------------------------------------------------
    # Mechanism B: Loss-level Entropy Bonus lambda in [0.0, 1e-4, 1e-3, 1e-2]
    # -------------------------------------------------------------
    mech_b_results: Dict[str, Any] = {}
    for lam in LAMBDA_LIST:
        key = f"lambda_{lam}"
        pos_records = []
        for i, pos in enumerate(positions):
            s = searches_baseline[i]
            pi_target = s["pi_prime"]
            ent = entropy(pi_target)
            max_p = float(np.max(pi_target))
            is_collapsed = bool(max_p > 0.95 or ent < 0.05)
            
            target_top1_action = int(s["legal_ids"][np.argmax(pi_target)])
            preserved = bool(target_top1_action == s["search_action"])
            
            grad_met = compute_gradient_metrics(
                prior_probs=s["prior_probs"],
                target_probs=pi_target,
                entropy_lambda=lam,
                search_action=s["search_action"],
                legal_ids=s["legal_ids"],
            )
            
            pos_records.append({
                "phase": pos["phase"],
                "entropy": ent,
                "max_p": max_p,
                "is_collapsed": is_collapsed,
                "top1_preserved": preserved,
                "l2_norm": grad_met["l2_norm"],
                "pull_best": grad_met["pull_best"],
                "cos_sim": grad_met["cos_sim"],
                "is_safe": grad_met["is_safe"],
            })
            
        mech_b_results[key] = {
            "lambda": lam,
            "overall": agg(pos_records),
            "by_phase": {ph: agg([r for r in pos_records if r["phase"] == ph]) for ph in ["opening", "middlegame", "endgame"]},
        }
        
    # -------------------------------------------------------------
    # Mechanism C: Adaptive c_scale(t) vs Fixed c_scale=0.10
    # -------------------------------------------------------------
    mech_c_results: Dict[str, Any] = {}
    for label, searches_set in [("fixed_baseline_0.10", searches_baseline), ("adaptive_step_schedule", searches_adaptive)]:
        pos_records = []
        for i, pos in enumerate(positions):
            s = searches_set[i]
            pi_target = s["pi_prime"]
            ent = entropy(pi_target)
            max_p = float(np.max(pi_target))
            is_collapsed = bool(max_p > 0.95 or ent < 0.05)
            
            target_top1_action = int(s["legal_ids"][np.argmax(pi_target)])
            preserved = bool(target_top1_action == s["search_action"])
            
            grad_met = compute_gradient_metrics(
                prior_probs=s["prior_probs"],
                target_probs=pi_target,
                entropy_lambda=0.0,
                search_action=s["search_action"],
                legal_ids=s["legal_ids"],
            )
            
            pos_records.append({
                "phase": pos["phase"],
                "entropy": ent,
                "max_p": max_p,
                "is_collapsed": is_collapsed,
                "top1_preserved": preserved,
                "l2_norm": grad_met["l2_norm"],
                "pull_best": grad_met["pull_best"],
                "cos_sim": grad_met["cos_sim"],
                "is_safe": grad_met["is_safe"],
            })
            
        mech_c_results[label] = {
            "schedule": label,
            "overall": agg(pos_records),
            "by_phase": {ph: agg([r for r in pos_records if r["phase"] == ph]) for ph in ["opening", "middlegame", "endgame"]},
        }
        
    # Save results to JSON
    print(f"\n[Step 4/4] Writing structured results to {out_json}...")
    output_data = {
        "metadata": {
            "experiment": "E1: Policy Entropy Regulation & Target Annealing",
            "source_pgn": "data/sample_real.pgn",
            "total_positions": n_pos,
            "phase_counts": {k: len(v) for k, v in phase_indices.items()},
            "tau_list": TAU_LIST,
            "lambda_list": LAMBDA_LIST,
            "adaptive_c_scale_schedule": ADAPTIVE_C_SCALE,
        },
        "mechanism_a_temperature": mech_a_results,
        "mechanism_b_entropy_bonus": mech_b_results,
        "mechanism_c_adaptive_c_scale": mech_c_results,
    }
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print("Done writing JSON file.")
    
    # -------------------------------------------------------------
    # Output Summary Tables
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print("TABLE 1: MECHANISM A — TARGET TEMPERATURE SOFTENING (c_scale=0.10)")
    print("=" * 95)
    print(f"{'tau':<8} {'Entropy H':<12} {'Max Prob':<12} {'Collapse %':<12} {'Top-1 Pres %':<14} {'L2 Grad':<12} {'Pull Best':<12} {'Cos Sim':<10}")
    print("-" * 95)
    for tau in TAU_LIST:
        m = mech_a_results[f"tau_{tau}"]["overall"]
        print(f"{tau:<8.2f} {m['mean_entropy']:<12.3f} {m['mean_max_prob']:<12.3f} {m['collapse_rate_pct']:<12.1f}% {m['top1_preservation_pct']:<14.1f}% {m['mean_l2_grad']:<12.3f} {m['mean_pull_best']:<12.3f} {m['mean_cos_sim']:<10.3f}")
        
    print("\n" + "=" * 95)
    print("TABLE 2: MECHANISM B — LOSS-LEVEL ENTROPY BONUS (L_soft_CE - lambda * H)")
    print("=" * 95)
    print(f"{'lambda':<10} {'Entropy H':<12} {'Max Prob':<12} {'Collapse %':<12} {'Top-1 Pres %':<14} {'L2 Grad':<12} {'Pull Best':<12} {'Cos Sim':<10}")
    print("-" * 95)
    for lam in LAMBDA_LIST:
        m = mech_b_results[f"lambda_{lam}"]["overall"]
        print(f"{lam:<10.1e} {m['mean_entropy']:<12.3f} {m['mean_max_prob']:<12.3f} {m['collapse_rate_pct']:<12.1f}% {m['top1_preservation_pct']:<14.1f}% {m['mean_l2_grad']:<12.3f} {m['mean_pull_best']:<12.3f} {m['mean_cos_sim']:<10.3f}")
        
    print("\n" + "=" * 95)
    print("TABLE 3: MECHANISM C — ADAPTIVE c_scale(t) SCHEDULE VS FIXED BASELINE")
    print("=" * 95)
    print(f"{'Schedule':<25} {'Phase':<12} {'c_scale':<10} {'Entropy H':<12} {'Collapse %':<12} {'Top-1 Pres %':<14} {'Pull Best':<10}")
    print("-" * 95)
    for label in ["fixed_baseline_0.10", "adaptive_step_schedule"]:
        m_ov = mech_c_results[label]["overall"]
        m_ph = mech_c_results[label]["by_phase"]
        cs_tag = "0.10 (all)" if "fixed" in label else "0.05/0.10/0.20"
        print(f"{label:<25} {'Overall':<12} {cs_tag:<10} {m_ov['mean_entropy']:<12.3f} {m_ov['collapse_rate_pct']:<12.1f}% {m_ov['top1_preservation_pct']:<14.1f}% {m_ov['mean_pull_best']:<10.3f}")
        for ph in ["opening", "middlegame", "endgame"]:
            p_info = m_ph[ph]
            cs_sub = 0.10 if "fixed" in label else ADAPTIVE_C_SCALE[ph]
            print(f"{'':<25} {ph.capitalize():<12} {cs_sub:<10.2f} {p_info['mean_entropy']:<12.3f} {p_info['collapse_rate_pct']:<12.1f}% {p_info['top1_preservation_pct']:<14.1f}% {p_info['mean_pull_best']:<10.3f}")
    print("=" * 95)


if __name__ == "__main__":
    run_experiment_e1()
