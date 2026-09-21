"""Loop 1: Search Guidance Scale Revisitation (sigma(q) Damping vs Linear Scale).

Evaluates 3 mathematical formulations of sigma(q) on real positions across diverse legal move counts:
- Formulation 1 (Linear baseline):
    sigma_1(q) = (c_visit + max_b N(b)) * c_scale * q_hat
    with c_scale=0.1, c_visit=50.
- Formulation 2 (Sublinear saturation damping):
    sigma_2(q) = (c_visit + sqrt(max_b N(b)) * sqrt(c_visit)) * c_scale * q_hat
- Formulation 3 (Dynamic Visit ratio damping):
    sigma_3(q) = (c_visit + max_b N(b) * (N(a) / (1 + sum N))) * c_scale * q_hat

Evaluates across 300+ real chess positions from opening, middlegame, and tactical endgames:
- Delta sigma span (max sigma - min sigma)
- Output target entropy H(pi')
- Gradient signal stability and Top-1 best move preservation
- Deep tactical branch behavior (N=64 simulation test)
- Saves structured results to runs/loop1_sigma_scaling.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import chess
import chess.pgn
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stateseq.actions import legal_mask, NUM_ACTIONS
from stateseq.gumbel import (
    C_VISIT,
    C_SCALE,
    EPS,
    M0,
    N_SIMS,
    Node,
    completed_q,
    normalize_q,
    order_halving,
    pi_prime,
    policy_probs,
    qtransform_completed,
    softmax,
)

FORMULATIONS = ["formulation_1", "formulation_2", "formulation_3"]


# ---------------- Mathematical Formulations ----------------

def sigma_f1(q_hat: np.ndarray, n_arr: np.ndarray, c_visit: float = 50.0, c_scale: float = 0.1) -> np.ndarray:
    """Formulation 1: Linear baseline.
    sigma_1(q) = (c_visit + max_b N(b)) * c_scale * q_hat
    """
    n_max = float(np.max(n_arr)) if len(n_arr) > 0 else 0.0
    scale = (c_visit + n_max) * c_scale
    return scale * np.asarray(q_hat, dtype=np.float32)


def sigma_f2(q_hat: np.ndarray, n_arr: np.ndarray, c_visit: float = 50.0, c_scale: float = 0.1) -> np.ndarray:
    """Formulation 2: Sublinear saturation damping.
    sigma_2(q) = (c_visit + sqrt(max_b N(b)) * sqrt(c_visit)) * c_scale * q_hat
    """
    n_max = float(np.max(n_arr)) if len(n_arr) > 0 else 0.0
    scale = (c_visit + math.sqrt(n_max) * math.sqrt(c_visit)) * c_scale
    return scale * np.asarray(q_hat, dtype=np.float32)


def sigma_f3(q_hat: np.ndarray, n_arr: np.ndarray, c_visit: float = 50.0, c_scale: float = 0.1) -> np.ndarray:
    """Formulation 3: Dynamic Visit ratio damping.
    sigma_3(q_a) = (c_visit + max_b N(b) * (N(a) / (1 + sum_b N(b)))) * c_scale * q_hat_a
    """
    n_max = float(np.max(n_arr)) if len(n_arr) > 0 else 0.0
    n_tot = float(np.sum(n_arr)) if len(n_arr) > 0 else 0.0
    ratios = np.asarray(n_arr, dtype=np.float32) / (1.0 + n_tot)
    scales = (c_visit + n_max * ratios) * c_scale
    return scales * np.asarray(q_hat, dtype=np.float32)


def compute_sigma_for_node(node: Node, form_name: str, c_visit: float = 50.0, c_scale: float = 0.1) -> np.ndarray:
    """Computes sigma(q_hat) using the specified formulation for all legal moves of the node."""
    cq = completed_q(node)
    if cq.size == 0:
        return cq
    q_hat = normalize_q(cq, float(cq.min()), float(cq.max()))
    n_arr = node.n if node.n.size else np.zeros(len(node.legal), dtype=np.int64)
    
    if form_name == "formulation_1":
        return sigma_f1(q_hat, n_arr, c_visit=c_visit, c_scale=c_scale)
    elif form_name == "formulation_2":
        return sigma_f2(q_hat, n_arr, c_visit=c_visit, c_scale=c_scale)
    elif form_name == "formulation_3":
        return sigma_f3(q_hat, n_arr, c_visit=c_visit, c_scale=c_scale)
    else:
        raise ValueError(f"Unknown formulation: {form_name}")


def compute_pi_prime(node: Node, form_name: str, c_visit: float = 50.0, c_scale: float = 0.1) -> np.ndarray:
    """Computes pi'(a) = softmax(logits + sigma_form(q_hat))."""
    if node.terminal or node.logits.size == 0:
        return np.zeros(0, np.float32)
    s_vec = compute_sigma_for_node(node, form_name, c_visit=c_visit, c_scale=c_scale)
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

def sample_real_positions(pgn_path: str, target_count: int = 330, seed: int = 42) -> List[Dict[str, Any]]:
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


# ---------------- Order Halving Search with Custom Selection ----------------

def run_search_for_formulation(
    board: chess.Board,
    legal_ids: np.ndarray,
    raw_logits: np.ndarray,
    root_q: float,
    child_qs: np.ndarray,
    form_name: str,
    n_sims: int = N_SIMS,
    m0: int = M0,
    seed: int = 0,
) -> Tuple[Node, dict]:
    """Runs sequential halving search where non-root selection and halving elimination
    use the candidate formulation of sigma(q_hat).
    """
    root = Node(
        q=float(root_q),
        legal=legal_ids.astype(np.int64),
        logits=raw_logits.copy(),
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

    rng = np.random.default_rng(seed)

    # Gumbel Top-m candidate selection
    from stateseq.gumbel import gumbel_topm, _n_rounds, _Candidate

    cands = gumbel_topm(root, m0=m0, rng=rng, g=1.0)
    m = len(cands)
    rounds = _n_rounds(m)
    surv = [_Candidate(action=a, noise=ns) for a, ns in cands]

    base, rem = divmod(n_sims, rounds)
    budget_per_round = [base + (1 if i < rem else 0) for i in range(rounds)]

    tree = [root]

    def _simulate(node: Node) -> float:
        if node.is_terminal:
            return float(node.q)
        # Select action using candidate formulation
        s_node = compute_sigma_for_node(node, form_name, c_visit=C_VISIT, c_scale=C_SCALE)
        pi_imp = softmax(node.logits + s_node)
        if node.n.size == 0:
            a = int(node.legal[np.argmax(pi_imp)])
        else:
            frac = node.n.astype(np.float32) / np.float32(1 + node.n_total)
            a = int(node.legal[np.argmax(pi_imp - frac)])

        edge_idx = int(np.flatnonzero(node.legal == a)[0])
        key = int(a)
        child = node.children.get(key)
        if child is None:
            child = expand_fn(node, a)
            node.children[key] = child
            tree.append(child)
            val = -float(child.q)
        else:
            val = -_simulate(child)
        node.record_child(edge_idx, val)
        return val

    def do_sim_root(c: _Candidate) -> None:
        if c.child is None:
            c.child = expand_fn(root, c.action)
            tree.append(c.child)
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
        # Elimination
        s_root = compute_sigma_for_node(root, form_name, c_visit=C_VISIT, c_scale=C_SCALE)
        s_map = {int(a): float(x) for a, x in zip(root.legal, s_root)}
        l_root = {int(a): float(x) for a, x in zip(root.legal, root.logits)}
        surv.sort(
            key=lambda cand: cand.noise + l_root[cand.action] + s_map[cand.action],
            reverse=True,
        )
        surv = surv[: max(1, len(surv) // 2)]

    return root, {"survivor": surv[0].action if surv else None}


# ---------------- Deep Tactical Branching Stress Test (N=64) ----------------

def evaluate_deep_tactical_branches() -> Dict[str, Any]:
    """Evaluates N=64 deep tactical branch behavior across formulations:
    - Over-amplification check: Does max_b N(b) = 64 blow up logits?
    - Compare scale amplification factor at N=0, N=16, N=32, N=64.
    """
    n_values = [0, 4, 8, 16, 32, 48, 64]
    c_visit = 50.0
    c_scale = 0.1

    scale_factors = {}
    for form in FORMULATIONS:
        factors = []
        for n in n_values:
            n_arr = np.array([n, 0, 0, 0], dtype=np.int64)
            q_hat = np.array([1.0, 0.5, 0.0, 0.0], dtype=np.float32)
            if form == "formulation_1":
                s = sigma_f1(q_hat, n_arr, c_visit, c_scale)
            elif form == "formulation_2":
                s = sigma_f2(q_hat, n_arr, c_visit, c_scale)
            elif form == "formulation_3":
                s = sigma_f3(q_hat, n_arr, c_visit, c_scale)
            factors.append({
                "N": n,
                "scale_factor": float((s[0] - s[2])), # delta sigma for q_hat span 1.0
                "s_best": float(s[0]),
                "s_worst": float(s[2]),
            })
        scale_factors[form] = factors

    # Scenario: Tactical sharp trap where N=64 concentrated on best move vs suboptimal moves
    # Logits prior uniform, delta Q = 1.0 (best Q = 0.8, blunder Q = -0.2)
    tactical_comparison = {}
    q_hat_span = 1.0
    for form in FORMULATIONS:
        # At N_max = 64
        n_arr = np.array([64, 0, 0, 0], dtype=np.int64)
        q_hat = np.array([1.0, 0.3, 0.1, 0.0], dtype=np.float32)
        logits = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if form == "formulation_1":
            s = sigma_f1(q_hat, n_arr, c_visit, c_scale)
        elif form == "formulation_2":
            s = sigma_f2(q_hat, n_arr, c_visit, c_scale)
        elif form == "formulation_3":
            s = sigma_f3(q_hat, n_arr, c_visit, c_scale)
        probs = softmax(logits + s)
        ent = calc_entropy(probs)
        tactical_comparison[form] = {
            "delta_sigma_at_64": float(s.max() - s.min()),
            "best_move_prob": float(probs[0]),
            "entropy": float(ent),
            "effective_logit_boost": float(s[0] - s[3]),
            "prevents_over_amplification": bool(float(s.max() - s.min()) <= 10.0),
        }

    return {
        "scale_factors_vs_N": scale_factors,
        "tactical_saturation_test": tactical_comparison,
    }


# ---------------- Main Evaluation Pipeline ----------------

def run_loop1_evaluation(
    pgn_path: str,
    target_positions: int = 330,
    seed: int = 42,
) -> Dict[str, Any]:
    print("=== Step 1: Sampling Real Positions ===")
    positions = sample_real_positions(pgn_path, target_count=target_positions, seed=seed)

    print(f"=== Step 2: Running Evaluation across {len(positions)} positions ===")
    rng = np.random.default_rng(seed)

    # Storage for detailed position records
    pos_records = []
    # Metrics accumulator
    metrics_per_form = {
        f: {
            "delta_sigma": [],
            "entropy_pi_prime": [],
            "entropy_prior": [],
            "kl_pi_prior": [],
            "max_p": [],
            "top1_preserved": [], # did pi' top-1 agree with best simulated Q?
            "grad_norm": [],      # L2 norm of gradient (softmax(logits) - pi')
            "grad_stability": [], # cosine sim of grad with true best move direction
            "by_phase": {"opening": [], "middlegame": [], "endgame": []},
            "by_legal_bucket": {"low_le20": [], "mid_21_40": [], "high_gt40": []},
        }
        for f in FORMULATIONS
    }

    for p_idx, pos in enumerate(positions):
        board = pos["board"]
        mask = legal_mask(board)
        legal_ids = np.flatnonzero(mask)
        num_legal = len(legal_ids)
        if num_legal == 0:
            continue

        phase = pos["phase"]
        if num_legal <= 20:
            legal_bucket = "low_le20"
        elif num_legal <= 40:
            legal_bucket = "mid_21_40"
        else:
            legal_bucket = "high_gt40"

        # Realistic prior logits: smooth distribution with entropy ~2.0 - 3.2
        raw_prior_logits = rng.gumbel(loc=0.0, scale=1.0, size=num_legal).astype(np.float32)
        perm = rng.permutation(num_legal)
        raw_prior_logits[perm] += np.linspace(2.2, 0.0, num_legal, dtype=np.float32)
        prior_probs = softmax(raw_prior_logits)
        prior_ent = calc_entropy(prior_probs)

        # Realistic child evaluations Q in [-1, 1]
        root_v = float(rng.uniform(-0.35, 0.35))
        child_qs = root_v - rng.exponential(scale=0.25, size=num_legal).astype(np.float32)
        # Guarantee true best move
        true_best_local_idx = int(perm[0]) if num_legal > 0 else 0
        child_qs[true_best_local_idx] = max(child_qs[true_best_local_idx], root_v + rng.uniform(0.08, 0.25))
        child_qs = np.clip(child_qs, -1.0, 1.0)
        best_child_action = int(legal_ids[true_best_local_idx])

        pos_eval_entry = {
            "ply": pos["ply"],
            "phase": phase,
            "legal_moves": num_legal,
            "legal_bucket": legal_bucket,
            "prior_entropy": prior_ent,
            "formulations": {},
        }

        # Run each formulation on identical position, prior, and random seed
        for f_idx, form in enumerate(FORMULATIONS):
            search_seed = seed + p_idx * 10 + f_idx
            root_node, s_info = run_search_for_formulation(
                board=board,
                legal_ids=legal_ids,
                raw_logits=raw_prior_logits,
                root_q=root_v,
                child_qs=child_qs,
                form_name=form,
                n_sims=N_SIMS,
                m0=M0,
                seed=search_seed,
            )

            # Target pi'
            pi_p = compute_pi_prime(root_node, form, c_visit=C_VISIT, c_scale=C_SCALE)
            ent_p = calc_entropy(pi_p)
            kl = calc_kl(pi_p, prior_probs)
            max_p = float(np.max(pi_p))

            # Delta sigma span
            s_vec = compute_sigma_for_node(root_node, form, c_visit=C_VISIT, c_scale=C_SCALE)
            d_sigma = float(s_vec.max() - s_vec.min()) if len(s_vec) > 0 else 0.0

            # Top-1 best move preservation: did pi' highest prob match best child Q?
            pi_p_top1_local_idx = int(np.argmax(pi_p))
            top1_match = (pi_p_top1_local_idx == true_best_local_idx)

            # Gradient signal stability:
            # Grad w.r.t logits: grad = softmax(logits) - pi' = prior_probs - pi_p
            # Gradient pulls network toward pi'.
            grad = prior_probs - pi_p
            grad_norm = float(np.linalg.norm(grad))
            
            # Target direction: -1 on true best move, >0 on others
            # Directional agreement: dot product with optimal step vector
            ideal_grad = np.zeros_like(grad)
            ideal_grad[true_best_local_idx] = -1.0
            ideal_grad = ideal_grad / np.linalg.norm(ideal_grad)
            grad_normalized = grad / (grad_norm + EPS)
            grad_stability = float(np.dot(-grad_normalized, -ideal_grad)) # alignment with pushing best move up

            # Record
            metrics_per_form[form]["delta_sigma"].append(d_sigma)
            metrics_per_form[form]["entropy_pi_prime"].append(ent_p)
            metrics_per_form[form]["entropy_prior"].append(prior_ent)
            metrics_per_form[form]["kl_pi_prior"].append(kl)
            metrics_per_form[form]["max_p"].append(max_p)
            metrics_per_form[form]["top1_preserved"].append(1.0 if top1_match else 0.0)
            metrics_per_form[form]["grad_norm"].append(grad_norm)
            metrics_per_form[form]["grad_stability"].append(grad_stability)

            # Groupings
            sample_summary = {
                "delta_sigma": d_sigma,
                "entropy": ent_p,
                "kl": kl,
                "max_p": max_p,
                "top1_preserved": top1_match,
                "grad_norm": grad_norm,
                "grad_stability": grad_stability,
            }
            metrics_per_form[form]["by_phase"][phase].append(sample_summary)
            metrics_per_form[form]["by_legal_bucket"][legal_bucket].append(sample_summary)

            pos_eval_entry["formulations"][form] = {
                "delta_sigma": round(d_sigma, 4),
                "entropy": round(ent_p, 4),
                "kl": round(kl, 4),
                "max_p": round(max_p, 4),
                "top1_match": bool(top1_match),
                "grad_norm": round(grad_norm, 4),
                "grad_stability": round(grad_stability, 4),
            }

        pos_records.append(pos_eval_entry)

        if (p_idx + 1) % 50 == 0 or (p_idx + 1) == len(positions):
            print(f"Processed {p_idx + 1}/{len(positions)} positions...")

    print("=== Step 3: Deep Tactical Saturation Test ===")
    deep_tactical_results = evaluate_deep_tactical_branches()

    # Aggregate summaries
    summary = {}
    for form in FORMULATIONS:
        m = metrics_per_form[form]
        d_sig = np.array(m["delta_sigma"])
        ent = np.array(m["entropy_pi_prime"])
        kl = np.array(m["kl_pi_prior"])
        max_p = np.array(m["max_p"])
        top1 = np.array(m["top1_preserved"])
        gnorm = np.array(m["grad_norm"])
        gstabi = np.array(m["grad_stability"])

        # Phase breakdowns
        phase_stats = {}
        for ph in ["opening", "middlegame", "endgame"]:
            sub = m["by_phase"][ph]
            if sub:
                phase_stats[ph] = {
                    "count": len(sub),
                    "mean_delta_sigma": float(np.mean([s["delta_sigma"] for s in sub])),
                    "mean_entropy": float(np.mean([s["entropy"] for s in sub])),
                    "mean_kl": float(np.mean([s["kl"] for s in sub])),
                    "top1_preservation_pct": float(np.mean([1.0 if s["top1_preserved"] else 0.0 for s in sub]) * 100),
                }

        # Legal bucket breakdowns
        bucket_stats = {}
        for b in ["low_le20", "mid_21_40", "high_gt40"]:
            sub = m["by_legal_bucket"][b]
            if sub:
                bucket_stats[b] = {
                    "count": len(sub),
                    "mean_delta_sigma": float(np.mean([s["delta_sigma"] for s in sub])),
                    "mean_entropy": float(np.mean([s["entropy"] for s in sub])),
                    "mean_kl": float(np.mean([s["kl"] for s in sub])),
                    "top1_preservation_pct": float(np.mean([1.0 if s["top1_preserved"] else 0.0 for s in sub]) * 100),
                }

        summary[form] = {
            "mean_delta_sigma": float(np.mean(d_sig)),
            "std_delta_sigma": float(np.std(d_sig)),
            "p50_delta_sigma": float(np.median(d_sig)),
            "p90_delta_sigma": float(np.percentile(d_sig, 90)),
            "max_delta_sigma": float(np.max(d_sig)),

            "mean_entropy": float(np.mean(ent)),
            "std_entropy": float(np.std(ent)),
            "p50_entropy": float(np.median(ent)),

            "mean_kl": float(np.mean(kl)),
            "p50_kl": float(np.median(kl)),
            "p90_kl": float(np.percentile(kl, 90)),

            "mean_max_p": float(np.mean(max_p)),
            "top1_preservation_pct": float(np.mean(top1) * 100),

            "mean_grad_norm": float(np.mean(gnorm)),
            "std_grad_norm": float(np.std(gnorm)),
            "mean_grad_stability": float(np.mean(gstabi)),

            "by_phase": phase_stats,
            "by_legal_bucket": bucket_stats,
        }

    output_data = {
        "metadata": {
            "task": "Loop 1: Search Guidance Scale Revisitation (sigma(q) Damping vs Linear Scale)",
            "n_positions": len(positions),
            "c_visit": C_VISIT,
            "c_scale": C_SCALE,
            "n_sims": N_SIMS,
            "m0": M0,
            "formulations": {
                "formulation_1": "Linear baseline: (c_visit + max_b N(b)) * c_scale * q_hat",
                "formulation_2": "Sublinear saturation damping: (c_visit + sqrt(max_b N(b)) * sqrt(c_visit)) * c_scale * q_hat",
                "formulation_3": "Dynamic visit ratio damping: (c_visit + max_b N(b) * (N(a) / (1 + sum N))) * c_scale * q_hat",
            },
        },
        "summary": summary,
        "deep_tactical_branches": deep_tactical_results,
        "sample_records": pos_records[:20], # Include 20 detailed sample positions
    }

    return output_data


def print_summary_table(summary: Dict[str, Any], deep_tactical: Dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print("LOOP 1: SEARCH GUIDANCE SCALE REVISITATION - SUMMARY METRICS TABLE")
    print("=" * 90)
    header = (
        f"{'Formulation':<28} | {'Mean Delta_sigma':<16} | {'p90 Delta_sigma':<15} | {'Mean H(pi\')':<11} | "
        f"{'Mean KL':<8} | {'Top-1 %':<8} | {'Grad Stab':<9}"
    )
    print(header)
    print("-" * 90)
    labels = {
        "formulation_1": "1. Linear Baseline",
        "formulation_2": "2. Sublinear Saturation",
        "formulation_3": "3. Dynamic Visit Ratio",
    }
    for form in FORMULATIONS:
        s = summary[form]
        print(
            f"{labels[form]:<28} | {s['mean_delta_sigma']:<16.4f} | {s['p90_delta_sigma']:<15.4f} | "
            f"{s['mean_entropy']:<11.4f} | {s['mean_kl']:<8.4f} | {s['top1_preservation_pct']:<7.1f}% | "
            f"{s['mean_grad_stability']:<9.4f}"
        )
    print("=" * 90)

    print("\nDEEP TACTICAL SATURATION TEST (N=64):")
    print("-" * 90)
    tact_header = f"{'Formulation':<28} | {'Delta_sigma at N=64':<19} | {'Top Move P':<11} | {'Entropy':<8} | {'Prevents Over-Amp'}"
    print(tact_header)
    print("-" * 90)
    for form in FORMULATIONS:
        t = deep_tactical["tactical_saturation_test"][form]
        print(
            f"{labels[form]:<28} | {t['delta_sigma_at_64']:<19.4f} | {t['best_move_prob']:<11.4f} | "
            f"{t['entropy']:<8.4f} | {str(t['prevents_over_amplification']):<15}"
        )
    print("=" * 90 + "\n")


def main():
    pgn_path = os.path.join(REPO_ROOT, "data", "sample_real.pgn")
    if not os.path.exists(pgn_path):
        raise FileNotFoundError(f"PGN dataset not found at {pgn_path}")

    out_dir = os.path.join(REPO_ROOT, "runs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "loop1_sigma_scaling.json")

    results = run_loop1_evaluation(pgn_path=pgn_path, target_positions=330, seed=42)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results successfully saved to {out_path}")
    print_summary_table(results["summary"], results["deep_tactical_branches"])


if __name__ == "__main__":
    main()
