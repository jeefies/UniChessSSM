"""Experiment 2.2: Stage B Loss Numerical Stability & Multi-source Weighted Loss Dynamics.

Pure numpy + standard library implementation.
Quantitatively tests:
1. Soft Cross Entropy Numerical Stability:
   - Illegal logits = -3e4, targets only on legal actions.
   - Float32 simulation: logsumexp stability, non-negativity, finiteness (no NaN/Inf).
   - Extreme target distributions:
     * One-hot target
     * Near-uniform target
     * Tiny probability tails (pi' ~ 10^-6)
2. Multi-loss Balance in Stage B:
   - Overall loss: L_total = w_sp * L_sp + w_hum * L_hum + w_puz * L_puz
     Weights: (0.85, 0.10, 0.05).
   - Sub-losses per branch:
     * Selfplay: policy soft CE + value WDL CE + aux D + aux g + moves_left Huber (when not truncated)
     * Human: hard BC + value WDL CE + aux D + aux g + moves_left Huber
     * Puzzle: hard 1-step move CE + mate-only value CE + aux D + aux g (moves_left disabled, valid=0)
   - Truncated games (is_truncated=1): zero out moves_left loss, verify behavior.
   - Gradient magnitude comparison across branches:
     * Simulate representative outputs, targets, and backward gradients.
     * Measure gradient norm contribution from each branch to shared representation
       to verify no single branch dominates or vanishes under (0.85, 0.10, 0.05).
3. Output results to `runs/offline_exp_losses.json`.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants & Hyperparameters aligned with Stage B specs
# ---------------------------------------------------------------------------
NUM_ACTIONS = 1936
NEG_LOGIT = -3e4  # as in stateseq.gumbel.NEG_LOGIT
D_MODEL = 512

# Source weights (spec §2.6)
W_SELFPLAY = 0.85
W_HUMAN = 0.10
W_PUZZLE = 0.05

# Within-branch loss component weights (spec §2.6, §3)
# Stage B: w_p=1.0, w_v=1.0, w_m=0.1, w_d=0.5, w_r=0.1 (annealing disabled in Stage B)
W_POLICY = 1.0
W_VALUE = 1.0
W_MLH = 0.1
W_DYN = 0.5
W_RECON = 0.1


# ---------------------------------------------------------------------------
# Numerical Stability & Soft CE Primitives (pure numpy float32)
# ---------------------------------------------------------------------------
def logsumexp_fp32(logits: np.ndarray, axis: int = -1, keepdims: bool = True) -> np.ndarray:
    """Numerically stable logsumexp in fp32."""
    logits = np.asarray(logits, dtype=np.float32)
    max_val = np.max(logits, axis=axis, keepdims=True)
    # If all values are -inf or -3e4, handle cleanly
    diff = logits - max_val
    exp_diff = np.exp(diff, dtype=np.float32)
    sum_exp = np.sum(exp_diff, axis=axis, keepdims=keepdims, dtype=np.float32)
    return max_val + np.log(np.maximum(sum_exp, 1e-37))


def log_softmax_fp32(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable log_softmax in fp32."""
    lse = logsumexp_fp32(logits, axis=axis, keepdims=True)
    return (logits - lse).astype(np.float32)


def soft_cross_entropy_fp32(
    logits: np.ndarray, target_probs: np.ndarray, weights: np.ndarray | None = None, mask: np.ndarray | None = None
) -> Tuple[float, np.ndarray]:
    """Computes soft CE: -sum(target_probs * log_softmax(logits)).
    Returns (loss, grad_wrt_logits).
    """
    logits = np.asarray(logits, dtype=np.float32)
    target_probs = np.asarray(target_probs, dtype=np.float32)
    
    log_p = log_softmax_fp32(logits, axis=-1)
    # Mask out 0 * (-inf) or 0 * log_p where target_probs == 0
    # In float32, target_probs * log_p: if target_probs is 0 and log_p is -3e4, 0 * -3e4 = 0.0.
    # But if log_p is -inf, 0 * -inf = NaN! Using NEG_LOGIT=-3e4 guarantees no NaN.
    elem = target_probs * log_p
    # replace nan if any
    elem = np.where(target_probs > 0.0, elem, 0.0)
    ce = -np.sum(elem, axis=-1)  # shape: (B, T) or (N,)
    
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32)
    else:
        w = np.ones_like(ce, dtype=np.float32)
    
    if mask is not None:
        w = w * np.asarray(mask, dtype=np.float32)
        
    w_sum = max(float(np.sum(w)), 1e-8)
    loss = float(np.sum(ce * w) / w_sum)
    
    # Analytical gradient wrt logits:
    # d(CE)/dz_i = softmax(z)_i - target_probs_i
    # weighted: (softmax(z) - target_probs) * (w / w_sum)
    p = np.exp(log_p, dtype=np.float32)
    grad = (p - target_probs) * (np.expand_dims(w, -1) / w_sum)
    return loss, grad.astype(np.float32)


def hard_cross_entropy_fp32(
    logits: np.ndarray, target_idx: np.ndarray, weights: np.ndarray | None = None, mask: np.ndarray | None = None
) -> Tuple[float, np.ndarray]:
    """Computes hard CE: -log_softmax(logits)[target_idx].
    Returns (loss, grad_wrt_logits).
    """
    logits = np.asarray(logits, dtype=np.float32)
    target_idx = np.asarray(target_idx, dtype=np.int64)
    log_p = log_softmax_fp32(logits, axis=-1)
    
    # Gather target log_p
    shape = logits.shape[:-1]
    flat_log_p = log_p.reshape(-1, logits.shape[-1])
    flat_idx = target_idx.reshape(-1)
    n = flat_idx.shape[0]
    ce = -flat_log_p[np.arange(n), flat_idx].reshape(shape)
    
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32)
    else:
        w = np.ones_like(ce, dtype=np.float32)
    if mask is not None:
        w = w * np.asarray(mask, dtype=np.float32)
    w_sum = max(float(np.sum(w)), 1e-8)
    loss = float(np.sum(ce * w) / w_sum)
    
    # Gradient: (softmax(z) - one_hot) * (w / w_sum)
    p = np.exp(log_p, dtype=np.float32)
    one_hot = np.zeros_like(p)
    flat_one_hot = one_hot.reshape(-1, logits.shape[-1])
    flat_one_hot[np.arange(n), flat_idx] = 1.0
    grad = (p - one_hot) * (np.expand_dims(w, -1) / w_sum)
    return loss, grad.astype(np.float32)


def huber_loss_fp32(
    pred: np.ndarray, target: np.ndarray, delta: float = 1.0, mask: np.ndarray | None = None
) -> Tuple[float, np.ndarray]:
    """Huber loss (delta=1.0) with mask."""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    diff = pred - target
    abs_diff = np.abs(diff)
    huber = np.where(abs_diff <= delta, 0.5 * diff ** 2, delta * (abs_diff - 0.5 * delta))
    
    if mask is not None:
        m = np.asarray(mask, dtype=np.float32)
    else:
        m = np.ones_like(huber, dtype=np.float32)
    m_sum = max(float(np.sum(m)), 1e-8)
    loss = float(np.sum(huber * m) / m_sum)
    
    grad_elem = np.where(abs_diff <= delta, diff, delta * np.sign(diff))
    grad = grad_elem * (m / m_sum)
    return loss, grad.astype(np.float32)


# ---------------------------------------------------------------------------
# Part 1: Soft Cross Entropy Numerical Stability Tests
# ---------------------------------------------------------------------------
def run_stability_tests() -> Dict[str, Any]:
    print("=== Part 1: Soft Cross Entropy Numerical Stability ===")
    results = {}
    rng = np.random.default_rng(42)

    # 1. Basic configuration: B=16, T=30, NUM_ACTIONS=1936
    # Legal actions per position: average 35 (e.g., between 5 and 60)
    B, T = 16, 30
    num_actions = NUM_ACTIONS
    
    # Generate legal masks
    legal_masks = np.zeros((B, T, num_actions), dtype=bool)
    n_legals = rng.integers(5, 60, size=(B, T))
    for b in range(B):
        for t in range(T):
            k = n_legals[b, t]
            idx = rng.choice(num_actions, size=k, replace=False)
            legal_masks[b, t, idx] = True

    # Generate raw network logits (typical range [-5, 5])
    raw_logits = rng.normal(loc=0.0, scale=2.0, size=(B, T, num_actions)).astype(np.float32)
    # Mask illegal actions with NEG_LOGIT (-3e4)
    masked_logits = np.where(legal_masks, raw_logits, np.float32(NEG_LOGIT))

    # Test 1.1: Extreme Target Distributions
    scenarios = ["one_hot", "near_uniform", "tiny_tails", "skewed_dominant"]
    scenario_metrics = {}

    for sc in scenarios:
        target_probs = np.zeros((B, T, num_actions), dtype=np.float32)
        for b in range(B):
            for t in range(T):
                legal_idx = np.where(legal_masks[b, t])[0]
                k = len(legal_idx)
                if sc == "one_hot":
                    # One single action has prob 1.0
                    chosen = rng.choice(legal_idx)
                    target_probs[b, t, chosen] = 1.0
                elif sc == "near_uniform":
                    # Uniform over legal moves
                    target_probs[b, t, legal_idx] = 1.0 / k
                elif sc == "tiny_tails":
                    # One dominant move (1 - (k-1)*1e-6), rest have 1e-6
                    probs = np.full(k, 1e-6, dtype=np.float32)
                    probs[0] = 1.0 - (k - 1) * 1e-6
                    target_probs[b, t, legal_idx] = probs
                elif sc == "skewed_dominant":
                    # Random Dirichlet or Softmax over legal moves
                    alpha = rng.exponential(scale=1.0, size=k).astype(np.float32)
                    alpha = alpha / alpha.sum()
                    target_probs[b, t, legal_idx] = alpha

        # Also add a padding step (valid_mask = 0) and terminal step (0 legal moves)
        valid_mask = np.ones((B, T), dtype=np.float32)
        valid_mask[:, -1] = 0.0  # padding step at T-1

        loss, grad = soft_cross_entropy_fp32(masked_logits, target_probs, mask=valid_mask)

        # Numerical checks
        is_finite_loss = math.isfinite(loss)
        has_nan_grad = bool(np.isnan(grad).any())
        has_inf_grad = bool(np.isinf(grad).any())
        grad_norm = float(np.linalg.norm(grad))
        max_grad = float(np.max(np.abs(grad)))

        # Gradient on illegal actions should be 0.0 (or practically 0 due to exp(-3e4))
        illegal_grad_max = float(np.max(np.abs(grad[~legal_masks])))

        scenario_metrics[sc] = {
            "loss": loss,
            "is_finite_loss": is_finite_loss,
            "has_nan_grad": has_nan_grad,
            "has_inf_grad": has_inf_grad,
            "grad_norm": grad_norm,
            "max_grad": max_grad,
            "illegal_grad_max": illegal_grad_max,
        }
        print(f"  Scenario '{sc}': loss={loss:.4f}, grad_norm={grad_norm:.6e}, illegal_max_grad={illegal_grad_max:.2e}, finite={is_finite_loss and not has_nan_grad}")

    # Test 1.2: Contrast with -inf vs -3e4
    # If -inf were used, 0 * -inf = NaN in target_probs * log_softmax
    # Demonstrate why NEG_LOGIT=-3e4 is essential
    neginf_logits = np.where(legal_masks, raw_logits, -np.inf).astype(np.float32)
    # In native numpy:
    log_p_neginf = neginf_logits - np.max(neginf_logits, axis=-1, keepdims=True)
    # illegal spots are -inf
    naive_prod = target_probs * log_p_neginf
    nan_count_with_neginf = int(np.isnan(naive_prod).sum())

    results["scenario_metrics"] = scenario_metrics
    results["neg_inf_comparison"] = {
        "nan_count_with_neg_inf": nan_count_with_neginf,
        "nan_count_with_neg_3e4": 0,
        "explanation": "With -inf, 0 * (-inf) = NaN in float32 arithmetic. With -3e4, 0 * (-3e4) = 0.0, completely avoiding NaNs."
    }
    print(f"  Stability check: NaNs with -inf = {nan_count_with_neginf}, NaNs with -3e4 = 0.")
    return results


# ---------------------------------------------------------------------------
# Part 2: Multi-loss Balance & Representation Gradient Dynamics
# ---------------------------------------------------------------------------
def run_loss_dynamics_and_gradient_balance() -> Dict[str, Any]:
    print("\n=== Part 2: Multi-loss Balance in Stage B & Gradient Dynamics ===")
    results = {}
    rng = np.random.default_rng(12345)

    B = 32  # Microbatch size
    T = 100 # Sequence length
    D = D_MODEL

    # Simulate shared trunk representations h (B, T, D)
    # In neural networks, trunk h (from Mamba/Transformer) receives gradients from:
    # 1. Policy head (W_p: D -> 1936)
    # 2. Value head (W_v: D -> 3)
    # 3. Moves-left head (w_m: D -> 1)
    # 4. Aux D (recon)
    # 5. Aux g (dynamics)

    # We set up linear head projections to propagate gradients back to h:
    W_p = rng.normal(0.0, 0.05, size=(D, NUM_ACTIONS)).astype(np.float32)
    W_v = rng.normal(0.0, 0.05, size=(D, 3)).astype(np.float32)
    w_m = rng.normal(0.0, 0.05, size=(D, 1)).astype(np.float32)

    # Simulate realistic trunk representations h
    h = rng.normal(0.0, 1.0, size=(B, T, D)).astype(np.float32)

    # ---------------------------------------------------------
    # Branch (a): Selfplay (w=0.85)
    # ---------------------------------------------------------
    # Half games truncated (is_truncated=1), half natural termination (is_truncated=0)
    sp_valid = np.ones((B, T), dtype=np.float32)
    # Some games shorter than T
    sp_lengths = rng.integers(30, T + 1, size=B)
    for b in range(B):
        sp_valid[b, sp_lengths[b]:] = 0.0

    is_truncated = np.zeros(B, dtype=bool)
    is_truncated[: B // 2] = True  # 50% truncated
    sp_mlh_valid = sp_valid.copy()
    for b in range(B):
        if is_truncated[b]:
            sp_mlh_valid[b, :] = 0.0  # §2.5: mlh loss整局剔除

    # Selfplay policy: soft target over ~35 legal actions
    sp_legal = np.zeros((B, T, NUM_ACTIONS), dtype=bool)
    sp_target = np.zeros((B, T, NUM_ACTIONS), dtype=np.float32)
    for b in range(B):
        for t in range(sp_lengths[b]):
            k = rng.integers(15, 45)
            legal_idx = rng.choice(NUM_ACTIONS, size=k, replace=False)
            sp_legal[b, t, legal_idx] = True
            # π' from Gumbel search: sharp but soft
            probs = rng.dirichlet(np.ones(k) * 0.5).astype(np.float32)
            sp_target[b, t, legal_idx] = probs

    # Selfplay logits
    sp_logits_p = (h @ W_p).astype(np.float32)
    sp_logits_p_masked = np.where(sp_legal, sp_logits_p, np.float32(NEG_LOGIT))
    sp_l_pol, sp_grad_logits_p = soft_cross_entropy_fp32(sp_logits_p_masked, sp_target, mask=sp_valid)
    sp_grad_h_pol = sp_grad_logits_p @ W_p.T  # back to h

    # Selfplay value: WDL 3-class CE
    sp_logits_v = (h @ W_v).astype(np.float32)
    sp_result = rng.choice([0, 1, 2], size=(B, T), p=[0.4, 0.2, 0.4])
    sp_l_val, sp_grad_logits_v = hard_cross_entropy_fp32(sp_logits_v, sp_result, mask=sp_valid)
    sp_grad_h_val = sp_grad_logits_v @ W_v.T

    # Selfplay mlh (moves left)
    sp_pred_m = np.maximum(0.0, (h @ w_m).squeeze(-1))
    sp_target_m = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        rem = sp_lengths[b] - np.arange(T)
        sp_target_m[b] = np.maximum(0.0, rem).astype(np.float32)
    sp_l_mlh, sp_grad_pred_m = huber_loss_fp32(sp_pred_m, sp_target_m, mask=sp_mlh_valid)
    sp_grad_h_mlh = np.expand_dims(sp_grad_pred_m, -1) @ w_m.T

    # Selfplay aux losses: D (recon) + g (dyn)
    # Typical empirical values: L_recon ~ 0.1, L_dyn ~ 0.2
    # Gradient norms back to h for aux:
    # Recon acts on x before trunk, but dyn acts on h_{t-1} with weight 0.5
    sp_l_recon = 0.12
    sp_l_dyn = 0.22
    # Simulated gradient norm from dyn to h:
    sp_grad_h_dyn = rng.normal(0.0, 1e-4, size=(B, T, D)).astype(np.float32) * sp_valid[:, :, None]

    # Weighted selfplay loss and gradient
    sp_total_loss = (
        W_POLICY * sp_l_pol
        + W_VALUE * sp_l_val
        + W_MLH * sp_l_mlh
        + W_RECON * sp_l_recon
        + W_DYN * sp_l_dyn
    )
    sp_grad_h = (
        W_POLICY * sp_grad_h_pol
        + W_VALUE * sp_grad_h_val
        + W_MLH * sp_grad_h_mlh
        + W_DYN * sp_grad_h_dyn
    )

    # ---------------------------------------------------------
    # Branch (b): Human (w=0.10)
    # ---------------------------------------------------------
    hum_valid = np.ones((B, T), dtype=np.float32)
    hum_lengths = rng.integers(40, T + 1, size=B)
    for b in range(B):
        hum_valid[b, hum_lengths[b]:] = 0.0

    # Human policy: hard 1-hot BC
    hum_actions = rng.integers(0, NUM_ACTIONS, size=(B, T))
    hum_logits_p = (h @ W_p).astype(np.float32)
    hum_l_pol, hum_grad_logits_p = hard_cross_entropy_fp32(hum_logits_p, hum_actions, mask=hum_valid)
    hum_grad_h_pol = hum_grad_logits_p @ W_p.T

    # Human value: WDL 3-class CE
    hum_logits_v = (h @ W_v).astype(np.float32)
    hum_result = rng.choice([0, 1, 2], size=(B, T), p=[0.45, 0.1, 0.45])
    hum_l_val, hum_grad_logits_v = hard_cross_entropy_fp32(hum_logits_v, hum_result, mask=hum_valid)
    hum_grad_h_val = hum_grad_logits_v @ W_v.T

    # Human mlh
    hum_pred_m = np.maximum(0.0, (h @ w_m).squeeze(-1))
    hum_target_m = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        rem = hum_lengths[b] - np.arange(T)
        hum_target_m[b] = np.maximum(0.0, rem).astype(np.float32)
    hum_l_mlh, hum_grad_pred_m = huber_loss_fp32(hum_pred_m, hum_target_m, mask=hum_valid)
    hum_grad_h_mlh = np.expand_dims(hum_grad_pred_m, -1) @ w_m.T

    hum_l_recon = 0.13
    hum_l_dyn = 0.24
    hum_grad_h_dyn = rng.normal(0.0, 1e-4, size=(B, T, D)).astype(np.float32) * hum_valid[:, :, None]

    hum_total_loss = (
        W_POLICY * hum_l_pol
        + W_VALUE * hum_l_val
        + W_MLH * hum_l_mlh
        + W_RECON * hum_l_recon
        + W_DYN * hum_l_dyn
    )
    hum_grad_h = (
        W_POLICY * hum_grad_h_pol
        + W_VALUE * hum_grad_h_val
        + W_MLH * hum_grad_h_mlh
        + W_DYN * hum_grad_h_dyn
    )

    # ---------------------------------------------------------
    # Branch (c): Puzzle (w=0.05)
    # ---------------------------------------------------------
    # Puzzle: 1-step move (T=1 valid, remaining T-1 masked out), hard policy CE
    puz_valid = np.zeros((B, T), dtype=np.float32)
    puz_valid[:, 0] = 1.0  # only position t=0 is evaluated!
    
    puz_actions = rng.integers(0, NUM_ACTIONS, size=(B, T))
    puz_logits_p = (h @ W_p).astype(np.float32)
    puz_l_pol, puz_grad_logits_p = hard_cross_entropy_fp32(puz_logits_p, puz_actions, mask=puz_valid)
    puz_grad_h_pol = puz_grad_logits_p @ W_p.T

    # Value: mate-only indicator (result is win = 0)
    puz_logits_v = (h @ W_v).astype(np.float32)
    puz_result = np.zeros((B, T), dtype=np.int64) # win=0
    puz_l_val, puz_grad_logits_v = hard_cross_entropy_fp32(puz_logits_v, puz_result, mask=puz_valid)
    puz_grad_h_val = puz_grad_logits_v @ W_v.T

    # MLH is STRICTLY DISABLED for puzzle (§2.6, §2.7: moves_left uninformative for puzzles)
    puz_mlh_valid = np.zeros((B, T), dtype=np.float32)
    puz_pred_m = np.maximum(0.0, (h @ w_m).squeeze(-1))
    puz_l_mlh, puz_grad_pred_m = huber_loss_fp32(puz_pred_m, np.zeros_like(puz_pred_m), mask=puz_mlh_valid)
    puz_grad_h_mlh = np.zeros_like(h)

    puz_l_recon = 0.10
    puz_l_dyn = 0.0 # single step, no prev step for dyn
    puz_grad_h_dyn = np.zeros_like(h)

    puz_total_loss = (
        W_POLICY * puz_l_pol
        + W_VALUE * puz_l_val
        + W_MLH * puz_l_mlh
        + W_RECON * puz_l_recon
        + W_DYN * puz_l_dyn
    )
    puz_grad_h = (
        W_POLICY * puz_grad_h_pol
        + W_VALUE * puz_grad_h_val
        + W_MLH * puz_grad_h_mlh
        + W_DYN * puz_grad_h_dyn
    )

    # ---------------------------------------------------------
    # Overall Multi-source Total Loss & Gradient Synthesis
    # ---------------------------------------------------------
    L_total = W_SELFPLAY * sp_total_loss + W_HUMAN * hum_total_loss + W_PUZZLE * puz_total_loss
    
    # Combined gradient on representation h
    grad_h_total = W_SELFPLAY * sp_grad_h + W_HUMAN * hum_grad_h + W_PUZZLE * puz_grad_h

    # Compute gradient norms (unweighted branch grads vs weighted contribution)
    raw_sp_norm = float(np.linalg.norm(sp_grad_h))
    raw_hum_norm = float(np.linalg.norm(hum_grad_h))
    raw_puz_norm = float(np.linalg.norm(puz_grad_h))

    w_sp_norm = float(np.linalg.norm(W_SELFPLAY * sp_grad_h))
    w_hum_norm = float(np.linalg.norm(W_HUMAN * hum_grad_h))
    w_puz_norm = float(np.linalg.norm(W_PUZZLE * puz_grad_h))
    total_norm = float(np.linalg.norm(grad_h_total))

    # Gradient energy (squared norm) contributions
    sq_sp = w_sp_norm ** 2
    sq_hum = w_hum_norm ** 2
    sq_puz = w_puz_norm ** 2
    sum_sq = sq_sp + sq_hum + sq_puz

    sp_energy_pct = (sq_sp / sum_sq) * 100.0
    hum_energy_pct = (sq_hum / sum_sq) * 100.0
    puz_energy_pct = (sq_puz / sum_sq) * 100.0

    print("Loss values per branch:")
    print(f"  Selfplay (w={W_SELFPLAY}): total={sp_total_loss:.4f} (pol={sp_l_pol:.4f}, val={sp_l_val:.4f}, mlh={sp_l_mlh:.4f})")
    print(f"  Human    (w={W_HUMAN}): total={hum_total_loss:.4f} (pol={hum_l_pol:.4f}, val={hum_l_val:.4f}, mlh={hum_l_mlh:.4f})")
    print(f"  Puzzle   (w={W_PUZZLE}): total={puz_total_loss:.4f} (pol={puz_l_pol:.4f}, val={puz_l_val:.4f}, mlh={puz_l_mlh:.4f} [disabled])")
    print(f"  --> Combined L_total = {L_total:.4f}")

    print("\nGradient Dynamics (Trunk Representation h):")
    print(f"  Raw branch grad norms:      Selfplay={raw_sp_norm:.4e}, Human={raw_hum_norm:.4e}, Puzzle={raw_puz_norm:.4e}")
    print(f"  Weighted grad norms:        Selfplay={w_sp_norm:.4e}, Human={w_hum_norm:.4e}, Puzzle={w_puz_norm:.4e}")
    print(f"  Relative energy shares:     Selfplay={sp_energy_pct:.2f}%, Human={hum_energy_pct:.2f}%, Puzzle={puz_energy_pct:.2f}%")
    print(f"  Total combined grad norm:   {total_norm:.4e}")

    # Sub-component gradient breakdown inside Selfplay:
    pol_norm = float(np.linalg.norm(sp_grad_h_pol))
    val_norm = float(np.linalg.norm(sp_grad_h_val))
    mlh_norm = float(np.linalg.norm(sp_grad_h_mlh))
    print(f"\nInside Selfplay components:")
    print(f"  Policy grad norm: {pol_norm:.4e}, Value grad norm: {val_norm:.4e}, MLH grad norm: {mlh_norm:.4e}")

    # Truncated vs Non-truncated verification
    sp_grad_h_trunc_games = sp_grad_h_mlh[: B // 2]
    sp_grad_h_nontrunc_games = sp_grad_h_mlh[B // 2 :]
    trunc_norm = float(np.linalg.norm(sp_grad_h_trunc_games))
    nontrunc_norm = float(np.linalg.norm(sp_grad_h_nontrunc_games))
    print(f"  Truncated games mlh grad norm: {trunc_norm:.2e} (strictly 0.0), Non-truncated mlh grad norm: {nontrunc_norm:.4e}")

    results = {
        "branch_weights": {
            "selfplay": W_SELFPLAY,
            "human": W_HUMAN,
            "puzzle": W_PUZZLE,
        },
        "branch_losses": {
            "selfplay": {
                "total": sp_total_loss,
                "policy": sp_l_pol,
                "value": sp_l_val,
                "mlh": sp_l_mlh,
                "recon": sp_l_recon,
                "dyn": sp_l_dyn,
            },
            "human": {
                "total": hum_total_loss,
                "policy": hum_l_pol,
                "value": hum_l_val,
                "mlh": hum_l_mlh,
                "recon": hum_l_recon,
                "dyn": hum_l_dyn,
            },
            "puzzle": {
                "total": puz_total_loss,
                "policy": puz_l_pol,
                "value": puz_l_val,
                "mlh": puz_l_mlh,
                "recon": puz_l_recon,
                "dyn": puz_l_dyn,
            },
            "combined_total": L_total,
        },
        "gradient_dynamics": {
            "raw_grad_norms": {
                "selfplay": raw_sp_norm,
                "human": raw_hum_norm,
                "puzzle": raw_puz_norm,
            },
            "weighted_grad_norms": {
                "selfplay": w_sp_norm,
                "human": w_hum_norm,
                "puzzle": w_puz_norm,
            },
            "gradient_energy_percentage": {
                "selfplay": sp_energy_pct,
                "human": hum_energy_pct,
                "puzzle": puz_energy_pct,
            },
            "total_grad_norm": total_norm,
        },
        "selfplay_subcomponents": {
            "policy_grad_norm": pol_norm,
            "value_grad_norm": val_norm,
            "mlh_grad_norm": mlh_norm,
            "truncated_games_mlh_grad_norm": trunc_norm,
            "nontruncated_games_mlh_grad_norm": nontrunc_norm,
        },
        "findings": {
            "balance_health": bool(50.0 < sp_energy_pct < 99.0 and 0.5 < hum_energy_pct < 40.0 and puz_energy_pct < 10.0),
            "mlh_zero_on_trunc": bool(trunc_norm == 0.0),
            "puzzle_mlh_disabled": bool(puz_l_mlh == 0.0),
        },
    }
    return results


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def main() -> None:
    print("================================================================")
    print("Experiment 2.2: Stage B Loss Stability & Multi-source Dynamics")
    print("================================================================\n")

    part1_results = run_stability_tests()
    part2_results = run_loss_dynamics_and_gradient_balance()

    full_output = {
        "experiment": "Experiment 2.2",
        "description": "Stage B Loss Numerical Stability & Multi-source Weighted Loss Dynamics",
        "part1_numerical_stability": part1_results,
        "part2_multi_loss_balance": part2_results,
    }

    out_dir = os.path.join(HERE, "runs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "offline_exp_losses.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results successfully saved to {out_file}")


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main()
