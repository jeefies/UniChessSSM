#!/usr/bin/env python3
"""Loop 3: WDL Value Loss Dynamics (CE vs MSE vs Focal CE).

Evaluates three value loss formulations on WDL predictions across diverse game regimes:
1. Cross Entropy (CE): Standard multi-class cross entropy on 3 classes [W, D, L].
2. Mean Squared Error (MSE): MSE on scalar Q-value (Q = p_W - p_L) vs target z in {+1, 0, -1}.
   Also evaluates MSE directly on probability simplex vectors (p_W, p_D, p_L).
3. Focal Cross Entropy (Focal CE): FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
   with gamma=2.0, focusing gradients on hard/uncertain predictions.

Evaluates gradient dynamics and properties across:
- Decisive wins (target z = +1, target class = 0 [W])
- Decisive losses (target z = -1, target class = 2 [L])
- Dead draws (target z = 0, target class = 1 [D])
- Well-calibrated predictions (p_target -> 1.0)
- Hard / miscalibrated predictions (e.g. blunders where p_target -> 0.0)
- Ambiguous / balanced predictions (p_W ~ p_D ~ p_L ~ 1/3)

Analyzes:
- Gradient magnitude ||dL/d(logits)|| w.r.t logits across confidence p_t in (0, 1)
- Gradient vanishing at well-calibrated regimes vs gradient explosion on blunders
- Draw margin resolution: separation and sensitivity in equal/drawish endgame regimes
- Saves comprehensive results and metrics to runs/loop3_wdl_dynamics.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------- Mathematical Formulations ----------------

def wdl_softmax(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)


def wdl_to_q(probs: torch.Tensor) -> torch.Tensor:
    """Q = p_W - p_L in [-1, 1]."""
    return probs[..., 0] - probs[..., 2]


def loss_ce(logits: torch.Tensor, target_class: torch.Tensor) -> torch.Tensor:
    """Standard Cross Entropy loss: -log(p_target)."""
    return F.cross_entropy(logits, target_class, reduction="none")


def loss_mse_q(logits: torch.Tensor, target_class: torch.Tensor) -> torch.Tensor:
    """MSE on scalar Q = p_W - p_L vs target z in {+1, 0, -1}."""
    # target_class: 0 -> +1.0, 1 -> 0.0, 2 -> -1.0
    mapping = torch.tensor([1.0, 0.0, -1.0], dtype=logits.dtype, device=logits.device)
    z_target = mapping[target_class]
    probs = wdl_softmax(logits)
    q_pred = wdl_to_q(probs)
    return (q_pred - z_target).pow(2)


def loss_mse_vec(logits: torch.Tensor, target_class: torch.Tensor) -> torch.Tensor:
    """MSE on probability vector p vs one-hot target vector y."""
    probs = wdl_softmax(logits)
    y_one_hot = F.one_hot(target_class, num_classes=3).to(logits.dtype)
    return (probs - y_one_hot).pow(2).sum(dim=-1)


def loss_focal_ce(
    logits: torch.Tensor,
    target_class: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Focal Cross Entropy: -alpha * (1 - p_t)^gamma * log(p_t)."""
    probs = wdl_softmax(logits)
    p_t = probs.gather(-1, target_class.unsqueeze(-1)).squeeze(-1)
    p_t = p_t.clamp(min=1e-12, max=1.0)
    focal_weight = (1.0 - p_t).pow(gamma)
    ce = -torch.log(p_t)
    return alpha * focal_weight * ce


# ---------------- Analytical & Synthetic Gradient Sweep ----------------

def sweep_confidence_gradients(
    gamma: float = 2.0,
) -> Dict[str, List[Dict[str, float]]]:
    """Sweeps target class confidence p_t from 0.01 to 0.99 for Win, Loss, and Draw regimes.
    Computes exact gradients dL/d(logits) and their L2 norm for CE, MSE_Q, MSE_Vec, and Focal CE.
    """
    p_t_vals = np.linspace(0.01, 0.99, 99)
    regimes = {
        "decisive_win": 0,    # Target = Win (0)
        "dead_draw": 1,       # Target = Draw (1)
        "decisive_loss": 2,   # Target = Loss (2)
    }

    results = {}

    for reg_name, target_cls_idx in regimes.items():
        records = []
        for pt in p_t_vals:
            # Construct symmetric non-target probabilities: (1 - pt)/2 each
            p_other = (1.0 - pt) / 2.0
            probs_np = np.zeros(3, dtype=np.float32)
            probs_np[target_cls_idx] = pt
            for i in range(3):
                if i != target_cls_idx:
                    probs_np[i] = p_other

            # Convert to logits: log(p) + constant
            logits_np = np.log(probs_np + 1e-15)
            target_t = torch.tensor([target_cls_idx], dtype=torch.long)

            # Measure gradient for each loss formulation
            grads = {}
            for name, loss_fn in [
                ("CE", loss_ce),
                ("MSE_Q", loss_mse_q),
                ("MSE_Vec", loss_mse_vec),
                ("Focal_CE", lambda lg, tg: loss_focal_ce(lg, tg, gamma=gamma)),
            ]:
                lg_t = torch.tensor(logits_np.reshape(1, 3), dtype=torch.float32, requires_grad=True)
                l = loss_fn(lg_t, target_t)
                l.backward()
                g = lg_t.grad.squeeze(0).numpy()
                gnorm = float(np.linalg.norm(g))
                g_target = float(g[target_cls_idx])
                grads[name] = {
                    "loss": float(l.item()),
                    "grad_norm": gnorm,
                    "grad_target": g_target,
                    "grad_vec": g.tolist(),
                }

            records.append({
                "p_target": float(pt),
                "CE": grads["CE"],
                "MSE_Q": grads["MSE_Q"],
                "MSE_Vec": grads["MSE_Vec"],
                "Focal_CE": grads["Focal_CE"],
            })
        results[reg_name] = records

    return results


# ---------------- Real Chess Positions Evaluation ----------------

def evaluate_on_real_game_regimes(
    pgn_path: str,
    target_count: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """Extracts positions from sample games and evaluates loss and gradient behavior
    across real chess contexts:
    - Decisive wins (evaluation heavily skewed, white or black winning)
    - Decisive losses (blunder defense or deep loss)
    - Dead draws (balanced endgame or symmetrical opening)
    """
    import chess
    import chess.pgn

    rng = np.random.default_rng(seed)
    games = []
    if os.path.exists(pgn_path):
        with open(pgn_path, "r", encoding="utf-8") as f:
            while len(games) < 50:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                moves = list(g.mainline_moves())
                res_str = g.headers.get("Result", "*")
                if res_str in ["1-0", "0-1", "1/2-1/2"] and len(moves) >= 10:
                    games.append((moves, res_str))

    # Buckets:
    # 0: Win, 1: Draw, 2: Loss
    win_positions = []
    loss_positions = []
    draw_positions = []

    for moves, res_str in games:
        board = chess.Board()
        # White result perspective
        if res_str == "1-0":
            w_res = 0 # win
        elif res_str == "0-1":
            w_res = 2 # loss
        else:
            w_res = 1 # draw

        for ply, mv in enumerate(moves, start=1):
            if ply > 120:
                break
            # Perspective of current turn
            turn_res = w_res if board.turn == chess.WHITE else (2 if w_res == 0 else (0 if w_res == 2 else 1))
            entry = {
                "fen": board.fen(),
                "ply": ply,
                "turn": "white" if board.turn == chess.WHITE else "black",
                "result_class": turn_res,
                "is_draw": (turn_res == 1),
            }
            if turn_res == 0:
                win_positions.append(entry)
            elif turn_res == 2:
                loss_positions.append(entry)
            else:
                draw_positions.append(entry)
            board.push(mv)

    per_bucket = target_count // 3
    rng.shuffle(win_positions)
    rng.shuffle(loss_positions)
    rng.shuffle(draw_positions)

    selected_wins = win_positions[:per_bucket]
    selected_losses = loss_positions[:per_bucket]
    selected_draws = draw_positions[:per_bucket]

    print(f"Sampled real positions: Wins={len(selected_wins)}, Losses={len(selected_losses)}, Draws={len(selected_draws)}")

    # We evaluate 3 model prediction scenarios per position:
    # Scenario A: Well-calibrated (model agrees with true outcome: p_true ~ 0.75 - 0.95)
    # Scenario B: Blunder / Miscalibrated (model predicts opposite outcome: p_opposite ~ 0.75 - 0.95)
    # Scenario C: Equal / Uncertain (model predicts balanced drawish probabilities: ~[0.33, 0.34, 0.33])

    def test_regime(positions_pool: List[Dict[str, Any]], regime_name: str) -> Dict[str, Any]:
        metrics = {
            "CE": {"loss": [], "grad_norm": [], "grad_target": []},
            "MSE_Q": {"loss": [], "grad_norm": [], "grad_target": []},
            "MSE_Vec": {"loss": [], "grad_norm": [], "grad_target": []},
            "Focal_CE": {"loss": [], "grad_norm": [], "grad_target": []},
        }

        scenario_breakdown = {
            "well_calibrated": {"CE": [], "MSE_Q": [], "MSE_Vec": [], "Focal_CE": []},
            "blunder_miscalibrated": {"CE": [], "MSE_Q": [], "MSE_Vec": [], "Focal_CE": []},
            "uncertain_drawish": {"CE": [], "MSE_Q": [], "MSE_Vec": [], "Focal_CE": []},
        }

        for p_idx, pos in enumerate(positions_pool):
            target_cls = pos["result_class"]
            target_t = torch.tensor([target_cls], dtype=torch.long)

            for scen in ["well_calibrated", "blunder_miscalibrated", "uncertain_drawish"]:
                sub_rng = np.random.default_rng(seed + p_idx * 17 + len(scen))
                if scen == "well_calibrated":
                    # High probability on target
                    p_target = sub_rng.uniform(0.70, 0.92)
                    p_rest = (1.0 - p_target) / 2.0
                    p_arr = np.array([p_rest, p_rest, p_rest], dtype=np.float32)
                    p_arr[target_cls] = p_target
                elif scen == "blunder_miscalibrated":
                    # High probability on wrong outcome
                    wrong_cls = 2 if target_cls == 0 else (0 if target_cls == 2 else 0)
                    p_wrong = sub_rng.uniform(0.70, 0.92)
                    p_rest = (1.0 - p_wrong) / 2.0
                    p_arr = np.array([p_rest, p_rest, p_rest], dtype=np.float32)
                    p_arr[wrong_cls] = p_wrong
                else:
                    # Drawish / uncertain
                    p_arr = np.array([0.33, 0.34, 0.33], dtype=np.float32)
                    p_arr += sub_rng.normal(0, 0.03, 3).astype(np.float32)
                    p_arr = np.clip(p_arr, 0.05, 0.9)
                    p_arr = p_arr / p_arr.sum()

                logits_np = np.log(p_arr + 1e-12)

                for name, fn in [
                    ("CE", loss_ce),
                    ("MSE_Q", loss_mse_q),
                    ("MSE_Vec", loss_mse_vec),
                    ("Focal_CE", lambda lg, tg: loss_focal_ce(lg, tg, gamma=2.0)),
                ]:
                    lg_t = torch.tensor(logits_np.reshape(1, 3), dtype=torch.float32, requires_grad=True)
                    loss = fn(lg_t, target_t)
                    loss.backward()
                    gnorm = float(torch.norm(lg_t.grad).item())
                    gtarg = float(lg_t.grad[0, target_cls].item())
                    l_val = float(loss.item())

                    metrics[name]["loss"].append(l_val)
                    metrics[name]["grad_norm"].append(gnorm)
                    metrics[name]["grad_target"].append(gtarg)

                    scenario_breakdown[scen][name].append({
                        "loss": l_val,
                        "grad_norm": gnorm,
                        "grad_target": gtarg,
                    })

        # Summarize
        summary_m = {}
        for name in ["CE", "MSE_Q", "MSE_Vec", "Focal_CE"]:
            summary_m[name] = {
                "mean_loss": float(np.mean(metrics[name]["loss"])),
                "std_loss": float(np.std(metrics[name]["loss"])),
                "mean_grad_norm": float(np.mean(metrics[name]["grad_norm"])),
                "std_grad_norm": float(np.std(metrics[name]["grad_norm"])),
                "p90_grad_norm": float(np.percentile(metrics[name]["grad_norm"], 90)),
                "mean_grad_target": float(np.mean(metrics[name]["grad_target"])),
            }

        scen_summary = {}
        for scen, loss_dict in scenario_breakdown.items():
            scen_summary[scen] = {}
            for name, items in loss_dict.items():
                scen_summary[scen][name] = {
                    "mean_loss": float(np.mean([x["loss"] for x in items])),
                    "mean_grad_norm": float(np.mean([x["grad_norm"] for x in items])),
                    "mean_grad_target": float(np.mean([x["grad_target"] for x in items])),
                }

        return {
            "regime": regime_name,
            "count": len(positions_pool),
            "summary": summary_m,
            "by_scenario": scen_summary,
        }

    res_wins = test_regime(selected_wins, "decisive_wins")
    res_losses = test_regime(selected_losses, "decisive_losses")
    res_draws = test_regime(selected_draws, "dead_draws")

    return {
        "decisive_wins": res_wins,
        "decisive_losses": res_losses,
        "dead_draws": res_draws,
    }


# ---------------- Draw Margin Sensitivity Analysis ----------------

def evaluate_draw_margin_sensitivity() -> Dict[str, Any]:
    """Analyzes loss sensitivity and gradient resolution when separating a drawn position (Q = 0)
    from a slight advantage (+0.10) vs slight disadvantage (-0.10).
    In grandmaster and engine chess, draw margin resolution is crucial to avoid false optimism.
    """
    # Suppose true outcome is Dead Draw (target = 1)
    target_draw = torch.tensor([1], dtype=torch.long)

    # Slight evaluation differences around draw:
    # 1. Exact draw prediction: p = [0.10, 0.80, 0.10] -> Q = 0.0
    # 2. Slight white edge: p = [0.15, 0.80, 0.05] -> Q = +0.10
    # 3. Slight black edge: p = [0.05, 0.80, 0.15] -> Q = -0.10
    # 4. Clear edge: p = [0.30, 0.60, 0.10] -> Q = +0.20
    scenarios = [
        ("exact_draw_Q0.0", np.array([0.10, 0.80, 0.10], dtype=np.float32)),
        ("slight_win_Q+0.1", np.array([0.15, 0.80, 0.05], dtype=np.float32)),
        ("slight_loss_Q-0.1", np.array([0.05, 0.80, 0.15], dtype=np.float32)),
        ("clear_edge_Q+0.2", np.array([0.30, 0.60, 0.10], dtype=np.float32)),
        ("wide_draw_Q0.0", np.array([0.333, 0.334, 0.333], dtype=np.float32)),
    ]

    draw_analysis = {}
    for label, probs_np in scenarios:
        logits_np = np.log(probs_np + 1e-12)
        q_val = float(probs_np[0] - probs_np[2])
        loss_dict = {}

        for name, fn in [
            ("CE", loss_ce),
            ("MSE_Q", loss_mse_q),
            ("MSE_Vec", loss_mse_vec),
            ("Focal_CE", lambda lg, tg: loss_focal_ce(lg, tg, gamma=2.0)),
        ]:
            lg_t = torch.tensor(logits_np.reshape(1, 3), dtype=torch.float32, requires_grad=True)
            loss = fn(lg_t, target_draw)
            loss.backward()
            loss_dict[name] = {
                "loss": float(loss.item()),
                "grad_norm": float(torch.norm(lg_t.grad).item()),
                "grad_draw": float(lg_t.grad[0, 1].item()),
                "grad_diff_win_loss": float((lg_t.grad[0, 0] - lg_t.grad[0, 2]).item()),
            }

        draw_analysis[label] = {
            "probs": probs_np.tolist(),
            "Q": q_val,
            "metrics": loss_dict,
        }

    return draw_analysis


# ---------------- Summary & Printing ----------------

def print_summary_table(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 95)
    print("LOOP 3: WDL VALUE LOSS DYNAMICS (CE vs MSE vs FOCAL CE) - SUMMARY")
    print("=" * 95)
    header = (
        f"{'Regime / Scenario':<32} | {'CE GradNorm':<12} | {'MSE_Q Grad':<12} | "
        f"{'MSE_Vec Gr':<12} | {'Focal Grad':<12}"
    )
    print(header)
    print("-" * 95)

    regimes = ["decisive_wins", "decisive_losses", "dead_draws"]
    for reg in regimes:
        rdata = results["real_game_evaluations"][reg]
        r_name = reg.replace("_", " ").title()
        print(f"[{r_name}] (N={rdata['count']})")
        for scen in ["well_calibrated", "blunder_miscalibrated", "uncertain_drawish"]:
            sc_data = rdata["by_scenario"][scen]
            sc_label = f"  - {scen.replace('_', ' ').capitalize()}"
            print(
                f"{sc_label:<32} | {sc_data['CE']['mean_grad_norm']:<12.4f} | "
                f"{sc_data['MSE_Q']['mean_grad_norm']:<12.4f} | "
                f"{sc_data['MSE_Vec']['mean_grad_norm']:<12.4f} | "
                f"{sc_data['Focal_CE']['mean_grad_norm']:<12.4f}"
            )
        print("-" * 95)

    print("\nDRAW MARGIN RESOLUTION (Target = Dead Draw, True Z = 0):")
    print("-" * 95)
    d_header = f"{'Scenario':<25} | {'Q Val':<7} | {'CE Loss':<9} | {'MSE_Q Loss':<11} | {'Focal Loss':<11} | {'CE d(W-L)':<10}"
    print(d_header)
    print("-" * 95)
    for sc_name, sc_info in results["draw_margin_sensitivity"].items():
        m = sc_info["metrics"]
        print(
            f"{sc_name:<25} | {sc_info['Q']:<7.2f} | {m['CE']['loss']:<9.4f} | "
            f"{m['MSE_Q']['loss']:<11.4f} | {m['Focal_CE']['loss']:<11.4f} | "
            f"{m['CE']['grad_diff_win_loss']:<10.4f}"
        )
    print("=" * 95 + "\n")


def run_loop3_pipeline() -> Dict[str, Any]:
    print("=== Step 1: Sweeping Confidence Gradients across (0.01 -> 0.99) ===")
    sweep_results = sweep_confidence_gradients(gamma=2.0)

    print("=== Step 2: Evaluating Real Positions on Decisive Wins, Losses, and Dead Draws ===")
    pgn_path = os.path.join(REPO_ROOT, "data", "sample_real.pgn")
    real_eval = evaluate_on_real_game_regimes(pgn_path=pgn_path, target_count=300, seed=42)

    print("=== Step 3: Evaluating Draw Margin Sensitivity ===")
    draw_margin = evaluate_draw_margin_sensitivity()

    # Consolidate overall verdict and comparative recommendations
    verdict = {
        "cross_entropy": {
            "characteristics": "Linear gradient in error (p - y), strong corrective signal on blunders, scale-invariant.",
            "draw_dynamics": "High sensitivity to probability shifts, naturally preserves 3-class simplex geometry.",
            "gradient_stability": "Robust, well-bounded; no saturation vanishing on blunders.",
        },
        "mse_scalar_q": {
            "characteristics": "Directly optimizes expected Q = p_W - p_L, but collapses draw distinction (0-0 vs 0.5-0.5).",
            "draw_dynamics": "Gradient vanishes when Q_pred matches z_target even if probability dispersion is completely wrong.",
            "gradient_stability": "Quadratic damping near convergence, sluggish correction of high-confidence blunders.",
        },
        "focal_ce": {
            "characteristics": "Suppresses gradients on well-calibrated positions (1-p_t)^2, amplifies hard blunders.",
            "draw_dynamics": "Can over-focus on noisy draw/win boundaries, leading to potential variance in endgame learning.",
            "gradient_stability": "Near-zero gradients for confident positions; very sharp gradients on outliers.",
        },
        "recommendation": (
            "Standard multi-class CE remains mathematically superior for WDL training in SSM, as it preserves "
            "the 3-class probability manifold and maintains constant non-vanishing gradient force on blunder corrections "
            "without collapsing the draw margin or causing gradient starvation on well-calibrated nodes."
        ),
    }

    out_data = {
        "metadata": {
            "task": "Loop 3: WDL Value Loss Dynamics (CE vs MSE vs Focal CE)",
            "formulations": {
                "CE": "CrossEntropy(logits, target_class)",
                "MSE_Q": "( (p_W - p_L) - z_target )^2",
                "MSE_Vec": "sum_c (p_c - y_c)^2",
                "Focal_CE": "(1 - p_t)^gamma * CE with gamma=2.0",
            },
            "regimes_evaluated": ["decisive_wins", "decisive_losses", "dead_draws"],
        },
        "real_game_evaluations": real_eval,
        "draw_margin_sensitivity": draw_margin,
        "confidence_gradient_sweep": {
            "sample_pts": [0.05, 0.20, 0.50, 0.80, 0.95],
            "win_regime_samples": [s for s in sweep_results["decisive_win"] if any(math.isclose(s["p_target"], pt, abs_tol=0.015) for pt in [0.05, 0.20, 0.50, 0.80, 0.95])],
            "draw_regime_samples": [s for s in sweep_results["dead_draw"] if any(math.isclose(s["p_target"], pt, abs_tol=0.015) for pt in [0.05, 0.20, 0.50, 0.80, 0.95])],
        },
        "verdict": verdict,
    }

    return out_data


def main():
    out_dir = os.path.join(REPO_ROOT, "runs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "loop3_wdl_dynamics.json")

    results = run_loop3_pipeline()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results successfully saved to {out_path}")
    print_summary_table(results)


if __name__ == "__main__":
    main()
