"""Experiment 2.1: Gumbel pi' Target Distribution & Gradient Response Analysis.

Quantitative measurement of:
1. Impact of c_scale across [0.05, 0.1, 0.2, 0.5, 1.0].
2. Representative tactical / strategic scenarios:
   - Scenario A (Clear Advantage / Sharp Tactic): One dominant move with Q=0.8, others around -0.5. Prior policy moderately uniform or slightly favoring wrong move.
   - Scenario B (Subtle Edge / Positional Squeeze): Best move Q=0.25, 2nd best Q=0.20, others Q=0.0. Policy logits minor differences.
   - Scenario C (Equal / Dead Drawn Endgame): All legal moves Q between -0.02 and +0.02.
   - Scenario D (Blunder Avoidance): Disastrous blunder (Q=-0.9), 3 solid moves (Q=0.1). Prior strongly favors the blunder.
3. Search simulation with n_sims=64, m0=16, sequential halving rounds, using stateseq.gumbel directly:
   - Output pi'(a) for each scenario and each c_scale.
   - Entropy of raw prior policy vs entropy of improved policy pi'.
   - KL divergence D_KL(pi' || pi_prior).
   - Effective temperature / logits boost range: Delta sigma = max sigma - min sigma.
   - Backward gradient pull on network logits: for CE loss L = - sum pi'(a) log pi_theta(a),
     grad w.r.t logits z is grad_z L = pi_theta - pi'. Compute gradient vector magnitude (L2 norm)
     and directional agreement with true best move (grad_z L component or cosine similarity).
4. Output structured tabular results and save summary JSON runs/offline_exp_gumbel.json.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stateseq.gumbel import (
    C_VISIT,
    M0,
    N_SIMS,
    Node,
    completed_q,
    export_pi_prime,
    order_halving,
    policy_probs,
    qtransform_completed,
    softmax,
)

C_SCALES = [0.05, 0.1, 0.2, 0.5, 1.0]


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) = sum p * log(p / q)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


@dataclass
class ScenarioSpec:
    name: str
    description: str
    num_moves: int
    best_move_idx: int
    prior_logits: np.ndarray  # shape (num_moves,)
    q_evals: np.ndarray       # shape (num_moves,) root-perspective true child values
    root_q: float             # root prior value


def build_scenarios() -> list[ScenarioSpec]:
    scenarios = []

    # Scenario A: Clear Advantage / Sharp Tactic
    # 16 legal moves. Move 0 is winning tactic (Q=+0.8). Moves 1..15 are bad (Q=-0.5).
    # Prior policy slightly favors a wrong move (e.g. move 1 has logit +0.5, move 0 has logit 0.0, others 0.0).
    n_a = 16
    logits_a = np.zeros(n_a, dtype=np.float32)
    logits_a[1] = 0.5  # human blunder / tempting move
    logits_a[0] = 0.0  # true sharp tactic
    q_a = np.full(n_a, -0.5, dtype=np.float32)
    q_a[0] = 0.8
    scenarios.append(ScenarioSpec(
        name="Scenario A",
        description="Sharp Tactic (1 dominant move Q=0.8, others Q=-0.5; prior favors wrong move)",
        num_moves=n_a,
        best_move_idx=0,
        prior_logits=logits_a,
        q_evals=q_a,
        root_q=0.0,
    ))

    # Scenario B: Subtle Edge / Positional Squeeze
    # 20 legal moves. Move 0 is best (Q=0.25), Move 1 is second best (Q=0.20), others Q=0.0.
    # Prior logits have minor differences (~0.5 to 1.0), e.g. move 0: 0.8, move 1: 1.0, others 0.0 ~ 0.5.
    n_b = 20
    logits_b = np.linspace(0.0, 0.5, n_b, dtype=np.float32)
    logits_b[0] = 0.8
    logits_b[1] = 1.0  # prior slightly prefers second best
    q_b = np.zeros(n_b, dtype=np.float32)
    q_b[0] = 0.25
    q_b[1] = 0.20
    scenarios.append(ScenarioSpec(
        name="Scenario B",
        description="Positional Squeeze (Best Q=0.25, 2nd Q=0.20, others Q=0.0; prior minor diffs)",
        num_moves=n_b,
        best_move_idx=0,
        prior_logits=logits_b,
        q_evals=q_b,
        root_q=0.1,
    ))

    # Scenario C: Equal / Dead Drawn Endgame
    # 12 legal moves. All Q between -0.02 and +0.02. Move 0 has +0.015, Move 1 has +0.010, Move 2 has 0.0, etc.
    # Prior logits roughly uniform with tiny jitter.
    n_c = 12
    rng = np.random.default_rng(42)
    logits_c = rng.uniform(-0.1, 0.1, n_c).astype(np.float32)
    q_c = np.linspace(0.015, -0.015, n_c, dtype=np.float32)
    scenarios.append(ScenarioSpec(
        name="Scenario C",
        description="Dead Drawn Endgame (All Q in [-0.02, +0.02]; prior roughly uniform)",
        num_moves=n_c,
        best_move_idx=0,
        prior_logits=logits_c,
        q_evals=q_c,
        root_q=0.0,
    ))

    # Scenario D: Blunder Avoidance
    # 10 legal moves. Move 0 is a disastrous blunder (Q = -0.9).
    # Moves 1, 2, 3 are solid moves (Q = 0.1). Other moves are mediocre (Q = -0.2).
    # Prior policy strongly favors the blunder move 0 (e.g. tempting trap, logit +2.5 vs 0.0).
    n_d = 10
    logits_d = np.zeros(n_d, dtype=np.float32)
    logits_d[0] = 2.5  # human trap: strongly favored by prior!
    logits_d[1] = 0.2
    logits_d[2] = 0.0
    logits_d[3] = -0.1
    q_d = np.full(n_d, -0.2, dtype=np.float32)
    q_d[0] = -0.9  # blunder!
    q_d[1] = 0.1   # solid
    q_d[2] = 0.1   # solid
    q_d[3] = 0.1   # solid
    scenarios.append(ScenarioSpec(
        name="Scenario D",
        description="Blunder Avoidance (Blunder Q=-0.9 with high prior; solid moves Q=0.1)",
        num_moves=n_d,
        best_move_idx=1,  # one of the solid moves
        prior_logits=logits_d,
        q_evals=q_d,
        root_q=-0.1,
    ))

    return scenarios


def make_expand_fn(spec: ScenarioSpec) -> Callable[[Node, int], Node]:
    """Build expand function where 1-ply simulation returns the child with child.q = -spec.q_evals[action].
    Note: stateseq.gumbel simulates: val = -child.q. Thus child.q = -root_perspective_q.
    Subsequent moves in child (if deep) return neutral values so search is consistent.
    """
    def expand(node: Node, action: int) -> Node:
        # If expanding from root:
        if node.depth == 0:
            root_q_for_move = spec.q_evals[action]
            # Child's perspective value is -root_q_for_move
            child_q = -float(root_q_for_move)
            # Create a 1-move terminal-ish or 1-move child
            # To allow deeper sims if requested, child can have a neutral legal move
            return Node(
                legal=np.array([999], dtype=np.int64),
                logits=np.array([0.0], dtype=np.float32),
                q=child_q,
                depth=1,
                action=action,
                path=(action,),
                terminal=True,  # terminal child makes child_value = -child.q directly
            )
        else:
            # deeper
            return Node(
                legal=np.array([], dtype=np.int64),
                logits=np.array([], dtype=np.float32),
                q=0.0,
                depth=node.depth + 1,
                action=action,
                path=node.path + (action,),
                terminal=True,
            )

    return expand


def run_experiment(seed: int = 12345):
    scenarios = build_scenarios()
    results = {}

    print("=" * 95)
    print("EXPERIMENT 2.1: Gumbel pi' Target Distribution & Gradient Response Analysis")
    print("=" * 95)

    for spec in scenarios:
        results[spec.name] = {
            "description": spec.description,
            "num_moves": spec.num_moves,
            "best_move_idx": spec.best_move_idx,
            "scales": {},
        }

        prior_probs = softmax(spec.prior_logits)
        prior_ent = entropy(prior_probs)

        print(f"\n[{spec.name}] {spec.description}")
        print(f"  Legal Moves: {spec.num_moves} | Prior Policy Entropy: {prior_ent:.4f}")
        print(f"  Prior Prob(Best Move {spec.best_move_idx}): {prior_probs[spec.best_move_idx]:.4f} | "
              f"Prior Prob Max: {prior_probs.max():.4f} (move {np.argmax(prior_probs)})")
        print("-" * 95)
        print(f"{'c_scale':>7} | {'pi_best':>8} | {'pi_blund':>8} | {'Ent(pi\')':>8} | {'KL(pi\'||pi)':>11} | "
              f"{'Delta_sigma':>11} | {'|grad_z|':>8} | {'grad_best':>9} | {'cos_sim':>7} | {'chosen':>6}")
        print("-" * 95)

        for c_scale in C_SCALES:
            # Run sequential halving with fixed seed (g=0 or paired Gumbel)
            # In selfplay generation, g=1.0 is used; for deterministic evaluation g=0.0 or fixed seed is useful.
            # We use g=1.0 with fixed seed to simulate actual selfplay generation dynamics,
            # and repeat across multiple seeds to average or report representative run.
            
            # For exact reproducibility of single representative run:
            root = Node(
                legal=np.arange(spec.num_moves, dtype=np.int64),
                logits=spec.prior_logits.copy(),
                q=spec.root_q,
            )
            expand_fn = make_expand_fn(spec)
            search_out = order_halving(
                root=root,
                expand=expand_fn,
                n_sims=N_SIMS,
                m0=M0,
                g=0.0,  # evaluate target dynamics under clean search
                seed=seed,
                c_visit=C_VISIT,
                c_scale=c_scale,
            )

            # Export pi_prime
            legal_ids, pi_prime_probs = export_pi_prime(root, c_visit=C_VISIT, c_scale=c_scale)

            # Metrics
            pi_prime_ent = entropy(pi_prime_probs)
            kl = kl_divergence(pi_prime_probs, prior_probs)

            # Delta sigma = max(sigma) - min(sigma) on completed Q
            sigma_vec = qtransform_completed(root, c_visit=C_VISIT, c_scale=c_scale)
            delta_sigma = float(sigma_vec.max() - sigma_vec.min()) if sigma_vec.size > 0 else 0.0

            # Gradient calculation:
            # Cross entropy loss: L = - sum_a pi'(a) * log pi_theta(a)
            # where pi_theta = softmax(z).
            # dL / dz = pi_theta - pi'.
            # If current network output is z = spec.prior_logits, then pi_theta = prior_probs.
            grad_z = prior_probs - pi_prime_probs  # shape (num_moves,)
            grad_norm = float(np.linalg.norm(grad_z))

            # Directional agreement with true best move:
            # To push logit towards best move, we want grad_z[best] < 0 (i.e. -dL/dz > 0).
            # The parameter update is z <- z - lr * grad_z = z + lr * (pi' - pi_theta).
            # So the pull on the best move is (pi'[best] - pi_theta[best]) = -grad_z[best].
            pull_best = float(-grad_z[spec.best_move_idx])

            # Ideal gradient vector would be (pi_prior - 1_best)
            ideal_pull = np.zeros_like(grad_z)
            ideal_pull[spec.best_move_idx] = 1.0
            ideal_grad = prior_probs - ideal_pull
            cos_sim = float(np.dot(grad_z, ideal_grad) / (np.linalg.norm(grad_z) * np.linalg.norm(ideal_grad) + 1e-12))

            chosen_action = search_out["action"]
            prob_best = float(pi_prime_probs[spec.best_move_idx])
            blunder_idx = 0 if spec.name == "Scenario D" else -1
            prob_blunder = float(pi_prime_probs[blunder_idx]) if blunder_idx >= 0 else 0.0

            blunder_str = f"{prob_blunder:8.4f}" if blunder_idx >= 0 else "   N/A   "

            print(f"{c_scale:7.2f} | {prob_best:8.4f} | {blunder_str} | {pi_prime_ent:8.4f} | {kl:11.4f} | "
                  f"{delta_sigma:11.4f} | {grad_norm:8.4f} | {pull_best:+9.4f} | {cos_sim:7.4f} | {chosen_action:6d}")

            results[spec.name]["scales"][str(c_scale)] = {
                "c_scale": c_scale,
                "chosen_action": int(chosen_action),
                "prob_best": prob_best,
                "prob_blunder": prob_blunder if blunder_idx >= 0 else None,
                "pi_prime_entropy": pi_prime_ent,
                "prior_entropy": prior_ent,
                "kl_div": kl,
                "delta_sigma": delta_sigma,
                "grad_norm": grad_norm,
                "pull_best": pull_best,
                "cos_sim_ideal": cos_sim,
                "n_visits": [int(x) for x in root.n],
                "completed_q": [float(x) for x in completed_q(root)],
                "pi_prime": [float(x) for x in pi_prime_probs],
            }

    # Save to runs/offline_exp_gumbel.json
    os.makedirs("runs", exist_ok=True)
    out_path = os.path.join("runs", "offline_exp_gumbel.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nExperiment complete. Summary saved to {out_path}.")


if __name__ == "__main__":
    run_experiment()
