#!/usr/bin/env python3
"""Loop 5: Completed-Q Normalization Re-examination (Soft Clamp / Quantile Scaling vs Strict Min-Max).

Evaluates 4 normalization formulations of completed-Q on real positions with simulated noisy tree search:
1. Baseline (Strict Min-Max):
     q_hat = (q - q_min) / (q_max - q_min + eps)
     q_min, q_max from all legal actions' completed-Q.
2. Variant A (Soft Margin Clamping):
     Clamp range to [q_median - delta, q_median + delta] before min-max scaling,
     where delta = max(0.20, 1.5 * IQR(cq)). Outliers beyond the soft margin are clamped.
3. Variant B (Top-k Centered Quantile Scaling):
     Robust quantile-centered scaling using 10th-90th percentiles or robust interquartile spread:
     q_hat = clip((q - q_p10) / (q_p90 - q_p10 + eps), 0.0, 1.0).
4. Variant C (Outlier-resistant Trimmed Min-Max):
     Exclude the single worst outlier from defining q_min if N_visits < 2.
     q_hat = clip((q - q_min_trimmed) / (q_max - q_min_trimmed + eps), 0.0, 1.0).

Evaluates across 300 real chess positions (opening, middlegame, endgame) from data/sample_real.pgn:
- Top-move discrimination ratio: pi'(top1) / pi'(top2)
- Top-move logit gap in pi': log(pi'(top1) / pi'(top2))
- Policy target quality: Entropy H(pi'), KL(pi' || prior), max prob
- Top-1 best move preservation rate
- Outlier resilience under tactical blunders (Q_blunder = -0.9, N=1)
- Gradient stability and norm
- Saves structured results to runs/loop5_q_norm_revisit.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chess
import chess.pgn
import numpy as np

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stateseq.actions import legal_mask
from stateseq.gumbel import (
    C_SCALE,
    C_VISIT,
    EPS,
    M0,
    N_SIMS,
    Node,
    completed_q,
    normalize_q,
    sigma,
    softmax,
)

VARIANTS = ["baseline", "variant_a", "variant_b", "variant_c"]
VARIANT_NAMES = {
    "baseline": "Baseline: Strict Min-Max",
    "variant_a": "Variant A: Soft Margin Clamping",
    "variant_b": "Variant B: Quantile Scaling (p10-p90)",
    "variant_c": "Variant C: Outlier-resistant Trimmed Min-Max",
}


# ---------------- Normalization Formulations ----------------

def normalize_q_baseline(cq: np.ndarray, node: Node | None = None) -> np.ndarray:
    """Baseline: Strict min-max normalization."""
    if cq.size == 0:
        return cq
    q_min = float(np.min(cq))
    q_max = float(np.max(cq))
    span = q_max - q_min + EPS
    return (cq - np.float32(q_min)) / np.float32(span)


def normalize_q_variant_a(cq: np.ndarray, node: Node | None = None) -> np.ndarray:
    """Variant A: Soft Margin Clamping.
    Clamp range to [q_median - delta, q_median + delta] before min-max scaling.
    delta is adaptive: max(0.20, 1.5 * IQR(cq)).
    """
    if cq.size == 0:
        return cq
    q_med = float(np.median(cq))
    q25 = float(np.percentile(cq, 25))
    q75 = float(np.percentile(cq, 75))
    iqr = max(q75 - q25, 0.0)
    delta = max(0.20, 1.5 * iqr)
    
    cq_clamped = np.clip(cq, q_med - delta, q_med + delta)
    q_min = float(np.min(cq_clamped))
    q_max = float(np.max(cq_clamped))
    span = q_max - q_min + EPS
    return (cq_clamped - np.float32(q_min)) / np.float32(span)


def normalize_q_variant_b(cq: np.ndarray, node: Node | None = None) -> np.ndarray:
    """Variant B: Top-k Centered Quantile Scaling.
    Uses 10th and 90th percentiles to avoid extreme tail compression.
    Clipped to [0.0, 1.0].
    """
    if cq.size == 0:
        return cq
    p10 = float(np.percentile(cq, 10))
    p90 = float(np.percentile(cq, 90))
    span = p90 - p10 + EPS
    q_hat = (cq - np.float32(p10)) / np.float32(span)
    return np.clip(q_hat, 0.0, 1.0)


def normalize_q_variant_c(cq: np.ndarray, node: Node | None = None) -> np.ndarray:
    """Variant C: Outlier-resistant Trimmed Min-Max.
    Exclude the single worst outlier from setting q_min if its visits N < 2.
    Clipped to [0.0, 1.0].
    """
    if cq.size == 0:
        return cq
    q_max = float(np.max(cq))
    min_idx = int(np.argmin(cq))
    
    # Check visits of the minimum candidate if node information is available
    if node is not None and node.n.size > min_idx and len(cq) > 2:
        visits_at_min = int(node.n[min_idx])
        if visits_at_min < 2:
            # Exclude this single outlier from computing q_min
            cq_trimmed = np.delete(cq, min_idx)
            q_min = float(np.min(cq_trimmed))
        else:
            q_min = float(np.min(cq))
    else:
        # Fallback if node not passed: trim the single minimum if gap to 2nd lowest > 0.35
        sorted_cq = np.sort(cq)
        if len(sorted_cq) > 2 and (sorted_cq[1] - sorted_cq[0]) > 0.35:
            q_min = float(sorted_cq[1])
        else:
            q_min = float(sorted_cq[0])
            
    span = q_max - q_min + EPS
    q_hat = (cq - np.float32(q_min)) / np.float32(span)
    return np.clip(q_hat, 0.0, 1.0)


def qtransform_custom(
    node: Node,
    norm_fn_name: str,
    c_visit: float = C_VISIT,
    c_scale: float = C_SCALE,
) -> np.ndarray:
    """Computes sigma(q_hat) using the specified normalization method."""
    cq = completed_q(node)
    if cq.size == 0:
        return cq
    
    if norm_fn_name == "baseline":
        q_hat = normalize_q_baseline(cq, node)
    elif norm_fn_name == "variant_a":
        q_hat = normalize_q_variant_a(cq, node)
    elif norm_fn_name == "variant_b":
        q_hat = normalize_q_variant_b(cq, node)
    elif norm_fn_name == "variant_c":
        q_hat = normalize_q_variant_c(cq, node)
    else:
        raise ValueError(f"Unknown normalization variant: {norm_fn_name}")
        
    return sigma(q_hat, node.n_max, c_visit, c_scale)


def compute_pi_prime_custom(
    node: Node,
    norm_fn_name: str,
    c_visit: float = C_VISIT,
    c_scale: float = C_SCALE,
) -> np.ndarray:
    """pi'(a) = softmax(logits + sigma_custom(q_hat))."""
    if node.terminal or node.logits.size == 0:
        return np.zeros(0, np.float32)
    s_vec = qtransform_custom(node, norm_fn_name, c_visit=c_visit, c_scale=c_scale)
    return softmax(node.logits + s_vec)


def calc_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def calc_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


# ---------------- Sampling Real Chess Positions ----------------

def sample_real_positions(pgn_path: str, target_count: int = 300, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample diverse real positions across opening, middlegame, and endgame from PGN."""
    rng = np.random.default_rng(seed)
    games = []
    with open(pgn_path, "r", encoding="utf-8") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            if g.headers.get("Variant", "Standard").lower() != "standard":
                continue
            moves = list(g.mainline_moves())
            if len(moves) >= 6:
                games.append(moves)

    opening_pool = []
    middlegame_pool = []
    endgame_pool = []

    for g_idx, moves in enumerate(games):
        board = chess.Board()
        for ply, mv in enumerate(moves, start=1):
            if ply > 150:
                break
            if not board.is_game_over() and board.legal_moves.count() > 0:
                pos = {
                    "game_idx": g_idx,
                    "ply": ply,
                    "fen": board.fen(),
                    "turn": "white" if board.turn == chess.WHITE else "black",
                    "legal_moves_count": board.legal_moves.count(),
                    "board": board.copy(),
                }
                if ply <= 15:
                    opening_pool.append(pos)
                elif ply <= 45:
                    middlegame_pool.append(pos)
                else:
                    endgame_pool.append(pos)
            board.push(mv)

    per_bucket = target_count // 3
    n_open = min(len(opening_pool), per_bucket)
    n_mid = min(len(middlegame_pool), per_bucket)
    n_end = min(len(endgame_pool), target_count - n_open - n_mid)

    sampled = []
    for pool, n, phase in [(opening_pool, n_open, "opening"), 
                           (middlegame_pool, n_mid, "middlegame"), 
                           (endgame_pool, n_end, "endgame")]:
        idxs = rng.choice(len(pool), size=n, replace=False)
        for i in idxs:
            p = pool[i]
            p["phase"] = phase
            sampled.append(p)

    sampled.sort(key=lambda x: (x["game_idx"], x["ply"]))
    print(f"Sampled {len(sampled)} positions: opening={n_open}, mid={n_mid}, end={n_end}")
    return sampled


# ---------------- Simulated Tree Search Setup ----------------

def simulate_search_state(
    board: chess.Board,
    rng: np.random.Generator,
    has_severe_outlier: bool = False,
) -> Tuple[Node, int, int]:
    """Builds a realistic root Node with simulated sequential halving search state.
    Returns (node, true_best_idx, second_best_idx).
    
    If has_severe_outlier=True, adds an extreme blunder/terminal loss candidate (q = -0.90, N=1).
    """
    mask = legal_mask(board)
    legal_ids = np.flatnonzero(mask)
    num_legal = len(legal_ids)
    
    # 1. Realistic prior logits (entropy ~2.2 - 3.2)
    prior_logits = rng.gumbel(loc=0.0, scale=1.0, size=num_legal).astype(np.float32)
    perm = rng.permutation(num_legal)
    prior_logits[perm] += np.linspace(2.0, 0.0, num_legal, dtype=np.float32)
    
    # 2. Base evaluation and true values
    root_q = float(rng.uniform(-0.25, 0.25))
    true_qs = root_q + rng.normal(loc=0.0, scale=0.08, size=num_legal).astype(np.float32)
    
    # Best and second best separation (e.g. true best has +0.12, 2nd has +0.06)
    order = np.argsort(-true_qs)
    best_idx = int(order[0])
    second_idx = int(order[1]) if num_legal > 1 else best_idx
    true_qs[best_idx] = max(true_qs[best_idx], root_q + 0.12)
    if num_legal > 1:
        true_qs[second_idx] = true_qs[best_idx] - float(rng.uniform(0.04, 0.08))
    
    # 3. Simulate sequential halving visits (64 total visits allocated)
    visits = np.zeros(num_legal, dtype=np.int64)
    q_sums = np.zeros(num_legal, dtype=np.float32)
    
    # Top candidates receive visits (M0=16 down to survivors)
    m = min(M0, num_legal)
    cand_indices = list(order[:m])
    
    # Allocate visits: best receives 28-36 visits, second receives 12-18, others 1-4
    remaining_budget = N_SIMS
    n_best = min(remaining_budget - (m - 1), int(rng.integers(26, 36)))
    visits[best_idx] = n_best
    remaining_budget -= n_best
    
    if num_legal > 1 and remaining_budget > 0:
        n_second = min(remaining_budget - (m - 2), int(rng.integers(12, 18)))
        visits[second_idx] = n_second
        remaining_budget -= n_second
        
    for idx in cand_indices:
        if idx not in (best_idx, second_idx) and remaining_budget > 0:
            alloc = min(remaining_budget, int(rng.integers(1, 4)))
            visits[idx] = alloc
            remaining_budget -= alloc
            
    if remaining_budget > 0:
        visits[best_idx] += remaining_budget
        
    # Populate noisy q_sums
    for idx in range(num_legal):
        if visits[idx] > 0:
            noise = rng.normal(loc=0.0, scale=0.03, size=visits[idx])
            sample_qs = np.clip(true_qs[idx] + noise, -1.0, 1.0)
            q_sums[idx] = float(np.sum(sample_qs))
            
    # If severe outlier is enabled: choose a candidate with N=1 to be a blunder (q = -0.90)
    if has_severe_outlier and num_legal > 2:
        outlier_idx = -1
        # Look for candidate with N=1
        for idx in cand_indices:
            if idx not in (best_idx, second_idx) and visits[idx] == 1:
                outlier_idx = idx
                break
        if outlier_idx == -1:
            # Pick any non-top move and assign N=1
            for idx in range(num_legal):
                if idx not in (best_idx, second_idx):
                    outlier_idx = idx
                    visits[idx] = 1
                    break
        if outlier_idx != -1:
            q_sums[outlier_idx] = -0.90
            true_qs[outlier_idx] = -0.90

    node = Node(
        q=root_q,
        legal=legal_ids.astype(np.int64),
        logits=prior_logits.copy(),
        n=visits.copy(),
        q_sum=q_sums.copy(),
    )
    return node, best_idx, second_idx


# ---------------- Experiment Runner ----------------

def run_loop5_evaluation(
    pgn_path: str,
    target_positions: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    print("=" * 80)
    print("LOOP 5: COMPLETED-Q NORMALIZATION RE-EXAMINATION")
    print(f"Target positions: {target_positions}, Seed: {seed}")
    print("=" * 80)

    positions = sample_real_positions(pgn_path, target_count=target_positions, seed=seed)
    rng = np.random.default_rng(seed)

    # Metrics storage
    # We evaluate two regimes:
    # 1. standard: Realistic search distribution
    # 2. outlier: Severe tactical blunder / terminal loss explored (N=1, Q=-0.90)
    regimes = ["standard", "outlier"]
    metrics: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        reg: {
            v: {
                "discrim_ratio": [],      # pi'(top1) / pi'(top2)
                "discrim_logit_gap": [],  # log(pi'(top1) / pi'(top2))
                "entropy_pi_prime": [],   # H(pi')
                "entropy_prior": [],      # H(pi)
                "kl_divergence": [],      # KL(pi' || pi)
                "max_prob": [],           # max_a pi'(a)
                "top1_preserved": [],     # 1.0 if argmax(pi') == true_best else 0.0
                "grad_norm": [],          # ||prior - pi'||
                "grad_stability": [],     # cosine alignment with ideal direction
                "q_span": [],             # span used for normalization
            }
            for v in VARIANTS
        }
        for reg in regimes
    }

    sample_details = []

    for p_idx, pos in enumerate(positions):
        board = pos["board"]
        if board.legal_moves.count() < 2:
            continue

        pos_record = {
            "index": p_idx,
            "ply": pos["ply"],
            "phase": pos["phase"],
            "legal_moves": board.legal_moves.count(),
            "results": {},
        }

        # Evaluate both standard and outlier regimes
        for reg in regimes:
            has_outlier = (reg == "outlier")
            node, best_idx, second_idx = simulate_search_state(
                board=board,
                rng=rng,
                has_severe_outlier=has_outlier,
            )
            prior_probs = softmax(node.logits)
            prior_ent = calc_entropy(prior_probs)

            ideal_grad = np.zeros_like(prior_probs)
            ideal_grad[best_idx] = -1.0
            ideal_grad = ideal_grad / np.linalg.norm(ideal_grad)

            reg_results = {}

            for v in VARIANTS:
                # Compute pi'
                pi_p = compute_pi_prime_custom(node, v, c_visit=C_VISIT, c_scale=C_SCALE)
                ent_p = calc_entropy(pi_p)
                kl = calc_kl(pi_p, prior_probs)
                max_p = float(np.max(pi_p))

                # Top-move discrimination: top-1 vs top-2
                p_top1 = float(pi_p[best_idx])
                p_top2 = float(pi_p[second_idx]) if second_idx != best_idx else p_top1
                discrim_ratio = float(p_top1 / (p_top2 + EPS))
                discrim_logit_gap = float(math.log(max(p_top1, 1e-12)) - math.log(max(p_top2, 1e-12)))

                # Top-1 preserved
                top1_match = (int(np.argmax(pi_p)) == best_idx)

                # Gradient dynamics
                grad = prior_probs - pi_p
                grad_norm = float(np.linalg.norm(grad))
                grad_normed = grad / (grad_norm + EPS)
                grad_stability = float(np.dot(-grad_normed, -ideal_grad))

                # Normalization Q-span
                cq = completed_q(node)
                span = float(np.max(cq) - np.min(cq))

                m = metrics[reg][v]
                m["discrim_ratio"].append(discrim_ratio)
                m["discrim_logit_gap"].append(discrim_logit_gap)
                m["entropy_pi_prime"].append(ent_p)
                m["entropy_prior"].append(prior_ent)
                m["kl_divergence"].append(kl)
                m["max_prob"].append(max_p)
                m["top1_preserved"].append(1.0 if top1_match else 0.0)
                m["grad_norm"].append(grad_norm)
                m["grad_stability"].append(grad_stability)
                m["q_span"].append(span)

                reg_results[v] = {
                    "discrim_ratio": round(discrim_ratio, 3),
                    "discrim_logit_gap": round(discrim_logit_gap, 4),
                    "entropy": round(ent_p, 4),
                    "kl": round(kl, 4),
                    "max_prob": round(max_p, 4),
                    "top1_match": top1_match,
                    "grad_stability": round(grad_stability, 4),
                }

            pos_record["results"][reg] = reg_results

        if p_idx < 15:
            sample_details.append(pos_record)

        if (p_idx + 1) % 50 == 0 or (p_idx + 1) == len(positions):
            print(f"Evaluated {p_idx + 1}/{len(positions)} positions...")

    # Aggregate summaries
    summary = {}
    for reg in regimes:
        summary[reg] = {}
        for v in VARIANTS:
            m = metrics[reg][v]
            summary[reg][v] = {
                "variant_name": VARIANT_NAMES[v],
                "mean_discrim_ratio": float(np.mean(m["discrim_ratio"])),
                "p50_discrim_ratio": float(np.median(m["discrim_ratio"])),
                "p90_discrim_ratio": float(np.percentile(m["discrim_ratio"], 90)),
                "mean_discrim_logit_gap": float(np.mean(m["discrim_logit_gap"])),
                "p50_discrim_logit_gap": float(np.median(m["discrim_logit_gap"])),
                "mean_entropy": float(np.mean(m["entropy_pi_prime"])),
                "mean_kl": float(np.mean(m["kl_divergence"])),
                "mean_max_prob": float(np.mean(m["max_prob"])),
                "top1_preservation_pct": float(np.mean(m["top1_preserved"]) * 100),
                "mean_grad_norm": float(np.mean(m["grad_norm"])),
                "mean_grad_stability": float(np.mean(m["grad_stability"])),
            }

    # Cross-regime comparison: Outlier impact / resilience
    # Outlier degradation ratio: discrim_ratio(outlier) / discrim_ratio(standard)
    outlier_resilience = {}
    for v in VARIANTS:
        std_gap = summary["standard"][v]["mean_discrim_logit_gap"]
        out_gap = summary["outlier"][v]["mean_discrim_logit_gap"]
        gap_retention_pct = (out_gap / (std_gap + EPS)) * 100
        
        std_ratio = summary["standard"][v]["mean_discrim_ratio"]
        out_ratio = summary["outlier"][v]["mean_discrim_ratio"]
        ratio_retention_pct = (out_ratio / (std_ratio + EPS)) * 100

        top1_std = summary["standard"][v]["top1_preservation_pct"]
        top1_out = summary["outlier"][v]["top1_preservation_pct"]

        outlier_resilience[v] = {
            "variant_name": VARIANT_NAMES[v],
            "logit_gap_standard": float(std_gap),
            "logit_gap_outlier": float(out_gap),
            "logit_gap_retention_pct": float(gap_retention_pct),
            "discrim_ratio_standard": float(std_ratio),
            "discrim_ratio_outlier": float(out_ratio),
            "ratio_retention_pct": float(ratio_retention_pct),
            "top1_preserved_standard": float(top1_std),
            "top1_preserved_outlier": float(top1_out),
            "top1_drop": float(top1_std - top1_out),
        }

    output_data = {
        "metadata": {
            "task": "Loop 5: Completed-Q Normalization Re-examination",
            "n_positions": len(positions),
            "c_visit": C_VISIT,
            "c_scale": C_SCALE,
            "n_sims": N_SIMS,
            "m0": M0,
            "variants": VARIANT_NAMES,
            "regimes": {
                "standard": "Realistic tree search distribution without catastrophic outliers",
                "outlier": "Realistic tree search with one catastrophic blunder/terminal loss candidate (N=1, Q=-0.90)",
            },
        },
        "summary": summary,
        "outlier_resilience": outlier_resilience,
        "sample_records": sample_details,
    }

    return output_data


def print_summary_table(data: Dict[str, Any]) -> None:
    summary = data["summary"]
    resilience = data["outlier_resilience"]

    print("\n" + "=" * 105)
    print("LOOP 5: COMPLETED-Q NORMALIZATION RE-EXAMINATION (300 REAL POSITIONS)")
    print("=" * 105)

    print("\n--- 1. STANDARD REGIME (Normal Search Distributions) ---")
    header_std = (
        f"{'Variant':<36} | {'Top-1/2 Ratio':<14} | {'Logit Gap':<10} | {'Mean H(pi\')':<11} | "
        f"{'Mean KL':<8} | {'Top-1 %':<8} | {'Grad Stab':<9}"
    )
    print(header_std)
    print("-" * 105)
    for v in VARIANTS:
        s = summary["standard"][v]
        print(
            f"{VARIANT_NAMES[v]:<36} | {s['mean_discrim_ratio']:<14.3f} | {s['mean_discrim_logit_gap']:<10.4f} | "
            f"{s['mean_entropy']:<11.4f} | {s['mean_kl']:<8.4f} | {s['top1_preservation_pct']:<7.1f}% | "
            f"{s['mean_grad_stability']:<9.4f}"
        )
    print("-" * 105)

    print("\n--- 2. OUTLIER / BLUNDER REGIME (Catastrophic Blunder Q=-0.90, N=1 Present) ---")
    header_out = (
        f"{'Variant':<36} | {'Top-1/2 Ratio':<14} | {'Logit Gap':<10} | {'Mean H(pi\')':<11} | "
        f"{'Mean KL':<8} | {'Top-1 %':<8} | {'Grad Stab':<9}"
    )
    print(header_out)
    print("-" * 105)
    for v in VARIANTS:
        s = summary["outlier"][v]
        print(
            f"{VARIANT_NAMES[v]:<36} | {s['mean_discrim_ratio']:<14.3f} | {s['mean_discrim_logit_gap']:<10.4f} | "
            f"{s['mean_entropy']:<11.4f} | {s['mean_kl']:<8.4f} | {s['top1_preservation_pct']:<7.1f}% | "
            f"{s['mean_grad_stability']:<9.4f}"
        )
    print("-" * 105)

    print("\n--- 3. OUTLIER RESILIENCE & DISCRIMINATION RETENTION ---")
    res_header = (
        f"{'Variant':<36} | {'Std LogitGap':<12} | {'Out LogitGap':<12} | {'Gap Retention':<14} | "
        f"{'Std Top-1%':<10} | {'Out Top-1%':<10} | {'Top-1 Drop':<10}"
    )
    print(res_header)
    print("-" * 105)
    for v in VARIANTS:
        r = resilience[v]
        print(
            f"{VARIANT_NAMES[v]:<36} | {r['logit_gap_standard']:<12.4f} | {r['logit_gap_outlier']:<12.4f} | "
            f"{r['logit_gap_retention_pct']:<13.1f}% | {r['top1_preserved_standard']:<9.1f}% | "
            f"{r['top1_preserved_outlier']:<9.1f}% | {r['top1_drop']:<9.1f}%"
        )
    print("=" * 105 + "\n")


def main():
    pgn_path = os.path.join(REPO_ROOT, "data", "sample_real.pgn")
    if not os.path.exists(pgn_path):
        raise FileNotFoundError(f"PGN dataset not found at {pgn_path}")

    out_dir = os.path.join(REPO_ROOT, "runs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "loop5_q_norm_revisit.json")

    results = run_loop5_evaluation(pgn_path=pgn_path, target_positions=300, seed=42)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results successfully saved to {out_path}")
    print_summary_table(results)


if __name__ == "__main__":
    main()
