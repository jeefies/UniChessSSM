#!/usr/bin/env python3
"""Loop 9: Replay Buffer Cross-Generation Weight Decay & Quality Sampling Simulation.

Investigates replay buffer dynamics across 10 generations of selfplay:
- Buffer capacity: 10 generations x 25k games/gen = 250k games sliding window.
- In each generation g (0 to 9), selfplay data is produced by that generation's model.
- Model skill / tactical quality progressively improves from Gen 0 (B0 coldstart, noisy, high blunder rate)
  to Gen 9 (advanced, refined tactics, low blunder rate).

Compares 3 Sampling Strategies:
1. Baseline: Uniform across generations (w_g = 1/10 = 0.10 for all g in [0, 9]).
2. Geometric Decay:
   - beta = 0.75: w_g proportional to beta^(9 - g) (strong recency bias).
   - beta = 0.85: w_g proportional to beta^(9 - g) (moderate recency bias).
   - Normalized over active generations in buffer.
3. Quality-Weighted Sampling:
   - Evaluates games on:
     * Decisive outcomes (1-0 / 0-1 vs drawn / shuffle games).
     * Blunder score (inferred from sudden evaluation drops / game quality metrics).
   - Prioritizes decisive and clean games over repetitive / blunder-laden draws.
   - Combines recency floor (alpha=0.20 uniform floor) with quality score:
     w_i proportional to (1 - alpha) * quality_i + alpha * (1 / N).

Evaluates:
- Policy target drift: E_{x ~ sampled} [ KL(pi'_sampled || pi_latest) ]
- Mean target entropy and KL divergence to latest generation's teacher policy
- Learning sample efficiency: simulated progress per 1,000 gradient steps
- Retention of endgame / tactical edge cases (preventing catastrophic forgetting on old-generation tactical motifs)
- Saves structured audit results to runs/loop9_replay_buffer.json and prints summary table.
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

from stateseq.actions import NUM_ACTIONS
from stateseq.gumbel import softmax


# ---------------- Synthetic 10-Generation Buffer Simulation ----------------

@dataclass
class GenerationMeta:
    gen_id: int
    elo: float
    blunder_rate: float  # probability of high-variance blunder move per game
    decisive_rate: float # rate of decisive results (win/loss) vs draw
    policy_quality: float # scaling parameter for policy sharpness / tactical accuracy
    mean_entropy: float


@dataclass
class GameSample:
    game_id: int
    gen_id: int
    is_decisive: bool
    blunder_count: int
    game_length: int
    quality_score: float
    # Representative position features for KL / drift evaluation
    policy_logits: np.ndarray  # (NUM_ACTIONS,)
    pi_prime: np.ndarray       # (NUM_ACTIONS,)


def generate_synthetic_generations(num_gens: int = 10, seed: int = 42) -> List[GenerationMeta]:
    """Generates ground truth evolution parameters across 10 generations.
    Gen 0: Cold-start B0 (~1500 Elo, high blunder rate 0.45, high noise).
    Gen 9: Mature champion (~2500 Elo, blunder rate 0.05, sharp tactical accuracy).
    """
    rng = np.random.default_rng(seed)
    gens = []
    base_elo = 1500.0
    elo_step = (2500.0 - 1500.0) / (num_gens - 1)
    
    for g in range(num_gens):
        elo = base_elo + g * elo_step + rng.normal(0, 15.0)
        blunder_rate = float(np.clip(0.48 - g * 0.045 + rng.normal(0, 0.01), 0.04, 0.50))
        decisive_rate = float(np.clip(0.68 - g * 0.025 + rng.normal(0, 0.01), 0.40, 0.75))
        policy_quality = 1.0 + g * 0.35  # sharpness multiplier
        mean_entropy = float(np.clip(2.60 - g * 0.08, 1.80, 2.70))
        gens.append(GenerationMeta(
            gen_id=g,
            elo=elo,
            blunder_rate=blunder_rate,
            decisive_rate=decisive_rate,
            policy_quality=policy_quality,
            mean_entropy=mean_entropy,
        ))
    return gens


def build_generation_sample_pool(
    gens: List[GenerationMeta],
    samples_per_gen: int = 500,
    seed: int = 1234,
) -> List[GameSample]:
    """Builds a pool of sampled game positions across 10 generations.
    Each sample has ground-truth policy logits and pi' targets reflecting its generation's level.
    """
    rng = np.random.default_rng(seed)
    all_samples: List[GameSample] = []
    game_counter = 0

    # Fixed common action indices for synthetic chess evaluation (e.g. 64 plausible moves)
    n_active_moves = 40

    for g_meta in gens:
        g = g_meta.gen_id
        for _ in range(samples_per_gen):
            game_counter += 1
            is_decisive = bool(rng.random() < g_meta.decisive_rate)
            # Poisson blunders per game
            blunders = int(rng.poisson(g_meta.blunder_rate * 3.0))
            game_len = int(rng.normal(75.0, 18.0))
            game_len = max(20, min(250, game_len))

            # Quality metric: higher for decisive games with fewer blunders
            # Normalized in [0.1, 1.0]
            decisive_bonus = 0.35 if is_decisive else 0.0
            blunder_penalty = min(0.6, blunders * 0.18)
            length_penalty = 0.15 if game_len > 180 else 0.0
            raw_q = 0.50 + decisive_bonus - blunder_penalty - length_penalty
            quality = float(np.clip(raw_q + rng.normal(0, 0.05), 0.05, 1.0))

            # Synthetic policy logits: centered around true best moves with generation quality
            base_logits = np.full(NUM_ACTIONS, -3e4, dtype=np.float32)
            active_indices = np.arange(n_active_moves)
            
            # The true optimal move is index 0; good moves are 1..3; blunder is 30..39
            # In higher generations, mass concentrates on index 0..2
            true_signal = np.zeros(n_active_moves, dtype=np.float32)
            true_signal[0] = 3.0 * g_meta.policy_quality
            true_signal[1:4] = 1.5 * g_meta.policy_quality
            
            # Noise decreases with generation
            noise_scale = max(0.2, 2.5 - g * 0.22)
            noise = rng.gumbel(loc=0.0, scale=noise_scale, size=n_active_moves).astype(np.float32)
            
            if blunders > 0 and rng.random() < 0.5:
                # Add high noise on a blunder move
                blunder_idx = rng.integers(10, n_active_moves)
                noise[blunder_idx] += 4.5
                
            active_logits = true_signal + noise
            base_logits[active_indices] = active_logits

            # Compute pi_prime target via softmax with Gumbel search sharpening
            p_prime = np.zeros(NUM_ACTIONS, dtype=np.float32)
            sub_probs = softmax(active_logits * 1.2)
            p_prime[active_indices] = sub_probs

            all_samples.append(GameSample(
                game_id=game_counter,
                gen_id=g,
                is_decisive=is_decisive,
                blunder_count=blunders,
                game_length=game_len,
                quality_score=quality,
                policy_logits=base_logits,
                pi_prime=p_prime,
            ))

    return all_samples


# ---------------- Sampling Strategy Evaluator ----------------

def compute_sampling_probabilities(
    samples: List[GameSample],
    strategy: str,
    beta: float = 0.85,
    alpha_floor: float = 0.15,
) -> np.ndarray:
    """Computes normalized sampling probability for each game sample in the buffer."""
    n = len(samples)
    gen_ids = np.array([s.gen_id for s in samples], dtype=np.int32)
    latest_gen = 9

    if strategy == "uniform":
        # Strategy 1: Uniform across generations (each generation has equal total probability 1/10)
        # Each sample within generation g has weight 1 / (10 * n_g)
        unique_gens, counts = np.unique(gen_ids, return_counts=True)
        gen_to_prob = {g: 1.0 / (len(unique_gens) * cnt) for g, cnt in zip(unique_gens, counts)}
        weights = np.array([gen_to_prob[g] for g in gen_ids], dtype=np.float64)
        return weights / np.sum(weights)

    elif strategy == "geometric":
        # Strategy 2: Geometric recency decay w_g proportional to beta^(latest_gen - g)
        unique_gens, counts = np.unique(gen_ids, return_counts=True)
        raw_gen_w = {g: (beta ** (latest_gen - g)) for g in unique_gens}
        total_gen_w = sum(raw_gen_w.values())
        gen_to_prob = {g: (raw_gen_w[g] / total_gen_w) / cnt for g, cnt in zip(unique_gens, counts)}
        weights = np.array([gen_to_prob[g] for g in gen_ids], dtype=np.float64)
        return weights / np.sum(weights)

    elif strategy == "quality":
        # Strategy 3: Quality-weighted sampling with decisive bias and recency floor
        # Base quality score
        quality_scores = np.array([s.quality_score for s in samples], dtype=np.float64)
        # Recency scaling: mild recency preference (beta=0.90) combined with quality
        recency = np.array([0.90 ** (latest_gen - s.gen_id) for s in samples], dtype=np.float64)
        combined_score = quality_scores * recency
        
        # Floor anchor to avoid complete starvation of old / complex games
        uniform_floor = 1.0 / n
        p_raw = combined_score / np.sum(combined_score)
        p_final = (1.0 - alpha_floor) * p_raw + alpha_floor * uniform_floor
        return p_final / np.sum(p_final)

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


def evaluate_sampling_dynamics(
    samples: List[GameSample],
    probs: np.ndarray,
    reference_latest_target: np.ndarray,
    n_eval_draws: int = 15000,
    seed: int = 999,
) -> Dict[str, Any]:
    """Evaluates policy drift, target KL divergence, blunder dilution, and sample efficiency."""
    rng = np.random.default_rng(seed)
    drawn_indices = rng.choice(len(samples), size=n_eval_draws, p=probs, replace=True)
    drawn_samples = [samples[i] for i in drawn_indices]

    # 1. Generation representation distribution
    gen_counts = np.zeros(10, dtype=np.int32)
    for s in drawn_samples:
        gen_counts[s.gen_id] += 1
    gen_dist = (gen_counts / n_eval_draws).tolist()

    # 2. Quality & Blunder Exposure
    decisive_frac = float(np.mean([1.0 if s.is_decisive else 0.0 for s in drawn_samples]))
    mean_blunders_per_game = float(np.mean([s.blunder_count for s in drawn_samples]))
    blunder_game_pct = float(np.mean([1.0 if s.blunder_count > 0 else 0.0 for s in drawn_samples]) * 100.0)
    mean_quality = float(np.mean([s.quality_score for s in drawn_samples]))

    # 3. Policy Target Drift & KL Divergence to latest champion target
    # Active subset evaluated (indices 0..39)
    eps = 1e-9
    kl_divs = []
    entropies = []
    top1_match_count = 0

    latest_top1 = int(np.argmax(reference_latest_target))

    for s in drawn_samples:
        p = s.pi_prime[:40]
        p = p / (np.sum(p) + eps)
        q = reference_latest_target[:40]
        q = q / (np.sum(q) + eps)

        # KL(p || q) where p is the sampled training target, q is the latest teacher
        kl = float(np.sum(p * np.log((p + eps) / (q + eps))))
        kl_divs.append(max(0.0, kl))

        # Entropy H(p)
        h = float(-np.sum(p * np.log(p + eps)))
        entropies.append(h)

        if int(np.argmax(p)) == latest_top1:
            top1_match_count += 1

    mean_kl = float(np.mean(kl_divs))
    p95_kl = float(np.percentile(kl_divs, 95))
    mean_entropy = float(np.mean(entropies))
    top1_agreement_pct = float(top1_match_count / n_eval_draws * 100.0)

    # 4. Learning Efficiency Simulation (Simulated gradient alignment score)
    # Alignment: cosine similarity of gradient pulling towards latest truth vs noise
    # Higher quality & lower KL yields higher gradient signal-to-noise ratio (SNR)
    gradient_snr = float(1.0 / (1.0 + mean_kl) * (1.0 - mean_blunders_per_game * 0.15) * 10.0)
    effective_sample_multiplier = float(gradient_snr / 4.8)  # normalized relative to baseline

    # 5. Old-generation Edge Case Coverage (% of drawn samples from Gen 0..2 to prevent catastrophic forgetting)
    old_gen_coverage_pct = float(sum(gen_dist[:3]) * 100.0)
    recent_gen_coverage_pct = float(sum(gen_dist[7:]) * 100.0)

    return {
        "generation_distribution": [round(x, 4) for x in gen_dist],
        "old_gen_coverage_pct": round(old_gen_coverage_pct, 2),
        "recent_gen_coverage_pct": round(recent_gen_coverage_pct, 2),
        "decisive_rate": round(decisive_frac, 4),
        "mean_blunders_per_game": round(mean_blunders_per_game, 3),
        "blunder_game_pct": round(blunder_game_pct, 2),
        "mean_quality_score": round(mean_quality, 4),
        "mean_kl_to_latest": round(mean_kl, 4),
        "p95_kl_to_latest": round(p95_kl, 4),
        "mean_target_entropy": round(mean_entropy, 4),
        "top1_agreement_pct": round(top1_agreement_pct, 2),
        "gradient_snr": round(gradient_snr, 3),
        "effective_sample_multiplier": round(effective_sample_multiplier, 3),
    }


# ---------------- Main Orchestration ----------------

def run_loop9_simulation() -> Dict[str, Any]:
    print(">>> Generating synthetic 10-generation replay buffer (250k virtual pool)...")
    gens = generate_synthetic_generations(num_gens=10, seed=42)
    samples = build_generation_sample_pool(gens, samples_per_gen=500, seed=1234)

    # Reference latest teacher target: average pi' of Gen 9
    gen9_samples = [s for s in samples if s.gen_id == 9]
    reference_latest = np.mean([s.pi_prime for s in gen9_samples], axis=0)
    reference_latest = reference_latest / np.sum(reference_latest)

    strategies = [
        ("uniform", "Uniform (10% per Gen Baseline)", {}),
        ("geometric_beta075", "Geometric Decay (beta=0.75)", {"beta": 0.75}),
        ("geometric_beta085", "Geometric Decay (beta=0.85)", {"beta": 0.85}),
        ("quality_weighted", "Quality-Weighted (Decisive + Low-Blunder + Anchor)", {"alpha_floor": 0.15}),
    ]

    results: Dict[str, Any] = {
        "meta": {
            "num_generations": 10,
            "buffer_capacity_games": 250000,
            "games_per_gen": 25000,
            "pool_size_evaluated": len(samples),
            "eval_draws": 15000,
        },
        "strategies": {},
    }

    for key, name, kwargs in strategies:
        strat_type = "geometric" if "geometric" in key else ("uniform" if key == "uniform" else "quality")
        probs = compute_sampling_probabilities(samples, strategy=strat_type, **kwargs)
        eval_metrics = evaluate_sampling_dynamics(
            samples, probs, reference_latest_target=reference_latest, n_eval_draws=15000, seed=2026
        )
        eval_metrics["strategy_name"] = name
        results["strategies"][key] = eval_metrics

    # Save to runs/loop9_replay_buffer.json
    out_dir = Path(REPO_ROOT) / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "loop9_replay_buffer.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved Loop 9 replay buffer audit results to: {out_path}\n")
    print_summary_table(results)
    return results


def print_summary_table(results: Dict[str, Any]) -> None:
    print("=" * 115)
    print("LOOP 9: REPLAY BUFFER CROSS-GENERATION WEIGHT DECAY & QUALITY SAMPLING SIMULATION")
    print("=" * 115)
    print(f"{'Sampling Strategy':<38} | {'Recent(7-9)':<11} | {'Old(0-2)':<9} | {'Blunder%':<9} | {'KL to Gen9':<10} | {'Top-1 Agr%':<10} | {'SNR Mult'}")
    print("-" * 115)
    for key, data in results["strategies"].items():
        name = data["strategy_name"]
        recent = f"{data['recent_gen_coverage_pct']:.1f}%"
        old = f"{data['old_gen_coverage_pct']:.1f}%"
        blunder = f"{data['blunder_game_pct']:.1f}%"
        kl = f"{data['mean_kl_to_latest']:.3f}"
        top1 = f"{data['top1_agreement_pct']:.1f}%"
        eff = f"{data['effective_sample_multiplier']:.2f}x"
        print(f"{name:<38} | {recent:<11} | {old:<9} | {blunder:<9} | {kl:<10} | {top1:<10} | {eff}")
    print("=" * 115)


if __name__ == "__main__":
    run_loop9_simulation()
