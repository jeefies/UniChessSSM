#!/usr/bin/env python3
"""Loop 10: Arena Paired Wald SPRT Early Stopping Simulation.

Investigates sequential probability ratio testing (SPRT) for champion vs challenger
evaluation in the 400-game match pipeline:
- Baseline: Run all 400 games unconditionally.
- SPRT hypotheses:
    H0: winrate <= 0.50 (challenger is equal or inferior, do not promote)
    H1: winrate >= 0.55 (challenger meets the +55% promotion gate)
    Parameters: alpha = 0.05 (Type I error, false promotion),
                beta  = 0.05 (Type II error, false rejection).
- Wald boundaries:
    A = log((1 - beta) / alpha) = log(0.95 / 0.05) = log(19) ~ 2.9444 (accept H1)
    B = log(beta / (1 - alpha)) = log(0.05 / 0.95) = -log(19) ~ -2.9444 (reject H1 / accept H0)

Protocol refinements:
1. Paired Openings:
   - Games are played in pairs (Challenger plays White in game 2k, Black in game 2k+1 from the same opening).
   - Paired game score outcome:
       * Both win: +2 pts (Challenger +1.0)
       * Win + Draw: +1.5 pts (Challenger +0.75)
       * Win + Loss or Draw + Draw: +1.0 pts (Challenger +0.50)
       * Draw + Loss: +0.5 pts (Challenger +0.25)
       * Both lose: 0.0 pts (Challenger 0.0)
   - Early stopping decisions are evaluated ONLY at opening-pair boundaries (even game count N = 2, 4, ..., 400).
2. Fail-Fast Asymmetry (Hard Promotion Rule):
   - To promote to Champion, Challenger MUST complete all 400 games and achieve score >= 55% (220.0 / 400).
   - SPRT Early stopping is applied ONLY to reject inferior candidates (LLR <= B) early, saving GPU compute.
   - Minimum games before early stopping check: N_min = 64 games (32 opening pairs).
   - Maximum games: N_max = 400 games (200 opening pairs).

Evaluates across candidate strength profiles (simulated 10,000 matches each):
- Profile 1: Heavily Inferior candidate (true winrate = 35%, e.g. round2 level)
- Profile 2: Moderately Inferior candidate (true winrate = 45%)
- Profile 3: Equal candidate (true winrate = 50%)
- Profile 4: Borderline / Weak Superior candidate (true winrate = 53%)
- Profile 5: Genuine Qualifying candidate (true winrate = 57%)
- Profile 6: Dominant candidate (true winrate = 65%)

Tracks:
- Average games played (compute saving % vs 400 games)
- Early rejection rate (%)
- False rejection rate (rejecting true >= 55% candidate)
- False promotion rate (promoting < 55% candidate)
- Saves structured results to runs/loop10_arena_sprt.json and prints summary table.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------- Mathematical SPRT Scaffold ----------------

def compute_llr_bernoulli(wins: float, losses: float, draws: float, p0: float = 0.50, p1: float = 0.55) -> float:
    """Computes Log-Likelihood Ratio for game score.
    Using standard trinomial to binomial reduction or score-based normal approximation:
    In chess arena matches, score S = wins + 0.5 * draws, total N = wins + losses + draws.
    Using Wald's SPRT on effective binomial / scoring Bernoulli variable:
    LLR = S * log(p1 / p0) + (N - S) * log((1 - p1) / (1 - p0)).
    """
    s = wins + 0.5 * draws
    n = wins + losses + draws
    if n <= 0:
        return 0.0
    
    # Clamp bounds to avoid log(0)
    p0 = min(max(p0, 1e-6), 1.0 - 1e-6)
    p1 = min(max(p1, 1e-6), 1.0 - 1e-6)
    
    term1 = s * math.log(p1 / p0)
    term2 = (n - s) * math.log((1.0 - p1) / (1.0 - p0))
    return term1 + term2


@dataclass
class MatchResult:
    total_games: int
    challenger_wins: int
    challenger_losses: int
    draws: int
    challenger_score: float
    promoted: bool
    early_stopped: bool
    stop_reason: str  # "sprt_reject", "completed_promoted", "completed_rejected"


def simulate_paired_batch_vectorized(
    n_matches: int,
    true_winrate: float,
    draw_rate: float = 0.35,
    white_advantage: float = 0.05,
    min_games: int = 64,
    max_games: int = 400,
    alpha: float = 0.05,
    beta: float = 0.05,
    p0: float = 0.50,
    p1: float = 0.55,
    rng: np.random.Generator | None = None,
) -> List[MatchResult]:
    """Vectorized simulation of n_matches paired matches for maximum speed."""
    if rng is None:
        rng = np.random.default_rng()

    bound_b = math.log(beta / (1.0 - alpha))  # ~ -2.944 (reject H1)
    log_p1_p0 = math.log(p1 / p0)
    log_1p1_1p0 = math.log((1.0 - p1) / (1.0 - p0))

    # White / Black winrates for Challenger
    c_white_score = min(0.98, max(0.02, true_winrate + white_advantage))
    c_black_score = min(0.98, max(0.02, true_winrate - white_advantage))

    w_win_p = max(0.01, c_white_score - 0.5 * draw_rate)
    w_loss_p = max(0.01, (1.0 - c_white_score) - 0.5 * draw_rate)
    w_draw_p = max(0.01, 1.0 - w_win_p - w_loss_p)
    w_probs = np.array([w_win_p, w_draw_p, w_loss_p])
    w_probs = w_probs / np.sum(w_probs)

    b_win_p = max(0.01, c_black_score - 0.5 * draw_rate)
    b_loss_p = max(0.01, (1.0 - c_black_score) - 0.5 * draw_rate)
    b_draw_p = max(0.01, 1.0 - b_win_p - b_loss_p)
    b_probs = np.array([b_win_p, b_draw_p, b_loss_p])
    b_probs = b_probs / np.sum(b_probs)

    num_pairs = max_games // 2
    # Pre-generate all outcomes: shape (num_pairs, n_matches)
    # 0 = win (score 1.0), 1 = draw (score 0.5), 2 = loss (score 0.0)
    w_out = rng.choice([0, 1, 2], size=(num_pairs, n_matches), p=w_probs)
    b_out = rng.choice([0, 1, 2], size=(num_pairs, n_matches), p=b_probs)

    # Score per game
    score_map = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    win_map = np.array([1, 0, 0], dtype=np.int32)
    loss_map = np.array([0, 0, 1], dtype=np.int32)
    draw_map = np.array([0, 1, 0], dtype=np.int32)

    w_score = score_map[w_out]
    b_score = score_map[b_out]
    pair_score = w_score + b_score

    w_wins = win_map[w_out]
    b_wins = win_map[b_out]
    pair_wins = w_wins + b_wins

    w_losses = loss_map[w_out]
    b_losses = loss_map[b_out]
    pair_losses = w_losses + b_losses

    w_draws = draw_map[w_out]
    b_draws = draw_map[b_out]
    pair_draws = w_draws + b_draws

    cum_score = np.cumsum(pair_score, axis=0) # (num_pairs, n_matches)
    cum_wins = np.cumsum(pair_wins, axis=0)
    cum_losses = np.cumsum(pair_losses, axis=0)
    cum_draws = np.cumsum(pair_draws, axis=0)

    results: List[MatchResult] = []

    # For each match, find earliest stopping pair
    pair_counts = np.arange(1, num_pairs + 1) * 2 # games count: 2, 4, ..., 400

    for m in range(n_matches):
        stopped = False
        for p_idx in range(num_pairs):
            current_n = pair_counts[p_idx]
            if current_n < min_games:
                continue
            
            s = cum_score[p_idx, m]
            llr = s * log_p1_p0 + (current_n - s) * log_1p1_1p0

            if current_n < max_games and llr <= bound_b:
                results.append(MatchResult(
                    total_games=int(current_n),
                    challenger_wins=int(cum_wins[p_idx, m]),
                    challenger_losses=int(cum_losses[p_idx, m]),
                    draws=int(cum_draws[p_idx, m]),
                    challenger_score=float(s),
                    promoted=False,
                    early_stopped=True,
                    stop_reason="sprt_reject",
                ))
                stopped = True
                break
        
        if not stopped:
            final_s = cum_score[-1, m]
            promoted = (final_s / max_games) >= 0.55
            results.append(MatchResult(
                total_games=max_games,
                challenger_wins=int(cum_wins[-1, m]),
                challenger_losses=int(cum_losses[-1, m]),
                draws=int(cum_draws[-1, m]),
                challenger_score=float(final_s),
                promoted=bool(promoted),
                early_stopped=False,
                stop_reason="completed_promoted" if promoted else "completed_rejected",
            ))

    return results


# ---------------- Profile Evaluation Runner ----------------

PROFILES = [
    {"name": "Heavily Inferior (round2 level)", "true_winrate": 0.35, "desc": "W/L ratio ~ 0.35"},
    {"name": "Moderately Inferior", "true_winrate": 0.45, "desc": "W/L ratio ~ 0.45"},
    {"name": "Equal Candidate", "true_winrate": 0.50, "desc": "Challenger == Champion"},
    {"name": "Borderline Superior", "true_winrate": 0.53, "desc": "Below 55% gate"},
    {"name": "Genuine Qualifying Candidate", "true_winrate": 0.57, "desc": "Meets 55% gate"},
    {"name": "Dominant Candidate", "true_winrate": 0.65, "desc": "Clearly superior (+110 Elo)"},
]


def run_loop10_simulation(n_sim_matches: int = 10000, seed: int = 2026) -> Dict[str, Any]:
    print(f">>> Running Arena Paired Wald SPRT Simulation ({n_sim_matches} matches per profile)...")
    rng = np.random.default_rng(seed)

    results: Dict[str, Any] = {
        "meta": {
            "num_sim_matches_per_profile": n_sim_matches,
            "min_games_before_stop": 64,
            "max_games": 400,
            "promotion_threshold": 0.55,
            "h0_threshold": 0.50,
            "h1_threshold": 0.55,
            "alpha": 0.05,
            "beta": 0.05,
        },
        "profiles": {},
    }

    for p in PROFILES:
        name = p["name"]
        tw = p["true_winrate"]
        print(f"  -> Simulating: {name} (true winrate {tw*100:.1f}%)...")

        # 1. Run SPRT-assisted matches
        sprt_matches = simulate_paired_batch_vectorized(
            n_matches=n_sim_matches,
            true_winrate=tw,
            draw_rate=0.35,
            white_advantage=0.05,
            min_games=64,
            max_games=400,
            alpha=0.05,
            beta=0.05,
            p0=0.50,
            p1=0.55,
            rng=rng,
        )

        # 2. Run Ground Truth fixed 400-game baseline (same distribution)
        gt_matches = simulate_paired_batch_vectorized(
            n_matches=n_sim_matches,
            true_winrate=tw,
            draw_rate=0.35,
            white_advantage=0.05,
            min_games=400,  # disable early stop
            max_games=400,
            alpha=0.05,
            beta=0.05,
            p0=0.50,
            p1=0.55,
            rng=rng,
        )
        gt_promotions = sum(1 for m in gt_matches if m.promoted)

        # Calculate metrics
        games_played = [m.total_games for m in sprt_matches]
        mean_games = float(np.mean(games_played))
        median_games = float(np.median(games_played))
        p90_games = float(np.percentile(games_played, 90))
        compute_saving_pct = float((1.0 - mean_games / 400.0) * 100.0)

        early_stopped_count = sum(1 for m in sprt_matches if m.early_stopped)
        early_stop_pct = float(early_stopped_count / n_sim_matches * 100.0)

        sprt_promoted_count = sum(1 for m in sprt_matches if m.promoted)
        sprt_promo_pct = float(sprt_promoted_count / n_sim_matches * 100.0)
        gt_promo_pct = float(gt_promotions / n_sim_matches * 100.0)

        # False negative (wrongly rejected candidate that would qualify)
        # and false positive (promoted candidate that is inferior)
        if tw >= 0.55:
            fn_rate_pct = float(early_stopped_count / n_sim_matches * 100.0)
            fp_rate_pct = 0.0
        else:
            fn_rate_pct = 0.0
            fp_rate_pct = sprt_promo_pct

        results["profiles"][name] = {
            "true_winrate": tw,
            "mean_games": round(mean_games, 1),
            "median_games": round(median_games, 1),
            "p90_games": round(p90_games, 1),
            "compute_saving_pct": round(compute_saving_pct, 1),
            "early_rejection_pct": round(early_stop_pct, 1),
            "sprt_promotion_pct": round(sprt_promo_pct, 2),
            "gt_promotion_pct": round(gt_promo_pct, 2),
            "false_negative_pct": round(fn_rate_pct, 2),
            "false_positive_pct": round(fp_rate_pct, 2),
        }

    # Save to runs/loop10_arena_sprt.json
    out_dir = Path(REPO_ROOT) / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "loop10_arena_sprt.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved Loop 10 arena SPRT audit results to: {out_path}\n")
    print_summary_table(results)
    return results


def print_summary_table(results: Dict[str, Any]) -> None:
    print("=" * 115)
    print("LOOP 10: ARENA PAIRED WALD SPRT EARLY STOPPING SIMULATION (400 GAMES MATCH)")
    print("=" * 115)
    print(f"{'Candidate Profile':<35} | {'True Win%':<9} | {'Mean Games':<10} | {'Compute Saved':<13} | {'Early Stop%':<11} | {'Promo%':<7} | {'FN / FP'}")
    print("-" * 115)
    for name, data in results["profiles"].items():
        tw = f"{data['true_winrate']*100:.1f}%"
        mg = f"{data['mean_games']:.1f}"
        cs = f"{data['compute_saving_pct']:.1f}%"
        es = f"{data['early_rejection_pct']:.1f}%"
        promo = f"{data['sprt_promotion_pct']:.1f}%"
        if data['true_winrate'] >= 0.55:
            err = f"FN: {data['false_negative_pct']:.2f}%"
        else:
            err = f"FP: {data['false_positive_pct']:.2f}%"
        print(f"{name:<35} | {tw:<9} | {mg:<10} | {cs:<13} | {es:<11} | {promo:<7} | {err}")
    print("=" * 115)


if __name__ == "__main__":
    run_loop10_simulation()
