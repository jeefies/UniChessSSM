"""V3 Shard DataLoader Throughput, Microbatch Padding Waste & Prefetch Benchmark.

Direction 4 Implementation:
Evaluates V3 Shard DataLoader dynamics under different microbatch sizes B in [4, 8, 16, 32]
and sequence batching strategies:
  - Strategy A (Naive Fixed T=300): pad every microbatch to T=300.
  - Strategy B (Dynamic Batch Max T_batch <= 300): pad only to max(T_i) in the batch.
  - Strategy C (Length-sorted Dynamic Batching, as used in train/stage_b2.py chunking):
    group games of similar lengths, pad to max(T_i) of the chunk.

For each strategy and batch size, measures across 100 iterations:
  - Padding waste ratio: Waste = 1 - (sum real_plies) / (B * T_padded).
  - Data loading & collate throughput: games/sec and plies/sec.
  - Batch preparation latency: ms per batch (mean, p50, p90, p99).
  - Memory footprint per batch: MB (numpy + tensor memory allocated for batch).

Saves results to runs/dataloader_benchmark.json and prints a structured summary table.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

# Enable torch import if not natively in unichess env
try:
    import torch
except ModuleNotFoundError:
    zhiseek_site = r"C:\Users\jeefy\.conda\envs\ZhiSeek\Lib\site-packages"
    if os.path.exists(zhiseek_site) and zhiseek_site not in sys.path:
        sys.path.append(zhiseek_site)
    import torch

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataclasses import dataclass
from stateseq.actions import NUM_ACTIONS
from stateseq.adapter import standardize_elo
from stateseq.data.dataset import replay_game
from stateseq.data.dataset_selfplay import B2_T_MAX
from stateseq.data.gshards import V3ShardReader, validate_v3_pipol
from stateseq.features import FEATURE_DIM


@dataclass
class TrainBatch:
    """整序列训练批（数据加载及基准专用结构，避免导入依赖 mamba_ssm 的 model.py）"""
    features: torch.Tensor      # (B, T, 785)
    actions: torch.Tensor       # (B, T) int64
    legal_mask: torch.Tensor    # (B, T, 1936) bool
    results: torch.Tensor       # (B, T) int64
    moves_left: torch.Tensor    # (B, T) float
    elo_weight: torch.Tensor    # (B,) float
    tc_bucket: torch.Tensor     # (B,) int64
    elo_std: torch.Tensor       # (B,) float
    color: torch.Tensor         # (B, T) int64


def build_game_item(reader: V3ShardReader, index: int, t_max: int = B2_T_MAX):
    """Load and replay a single game from V3ShardReader."""
    g = reader.game(index)
    meta = g["meta"]
    data = replay_game(
        g["actions"],
        meta,
        t_max=t_max,
        pipol_actions=g["pipol_actions"],
        pipol_probs=g["pipol_probs"],
    )
    tc = int(meta["tc_bucket"])
    is_truncated = bool(meta["is_truncated"])
    elo_mean = float(meta["elo_mean"])
    return index, data, tc, is_truncated, elo_mean


def collate_microbatch(
    items: list[tuple[int, dict, int, bool, float]],
    target_t: int | None = None,
) -> dict[str, Any]:
    """Collate items into TrainBatch and soft target with specified target_t.
    
    If target_t is None, use max(len(actions)) in items.
    """
    items = sorted(items, key=lambda it: -len(it[1]["actions"]))
    b = len(items)
    max_real_len = max(len(it[1]["actions"]) for it in items)
    t = target_t if target_t is not None else max_real_len

    def pad_stack(key: str, dtype, shape_tail: tuple = ()) -> torch.Tensor:
        out = np.zeros((b, t) + shape_tail, dtype=dtype)
        for i, (_, d, *_rest) in enumerate(items):
            n = min(len(d["actions"]), t)
            out[i, :n] = d[key][:n]
        return torch.from_numpy(out)

    features = pad_stack("features", np.float32, (FEATURE_DIM,))
    legal = pad_stack("legal_mask", np.bool_, (NUM_ACTIONS,))
    actions = pad_stack("actions", np.int64)
    results = pad_stack("results", np.int64)
    moves_left = pad_stack("moves_left", np.float32)
    color = pad_stack("color", np.int64)
    real_lens = np.asarray([len(it[1]["actions"]) for it in items])
    valid = torch.from_numpy(np.arange(t)[None, :] < real_lens[:, None])

    elo_w = torch.ones(b, dtype=torch.float32)
    elo_arr = np.array([it[4] for it in items], dtype=np.float64)
    elo_std = torch.tensor(standardize_elo(elo_arr).astype(np.float32))
    tc = torch.tensor([it[2] for it in items], dtype=torch.long)

    soft_target = np.zeros((b, t, NUM_ACTIONS), dtype=np.float32)
    for i, (_, d, *_rest) in enumerate(items):
        for j, (acts, probs) in enumerate(zip(d["pipol_actions"], d["pipol_probs"])):
            if j >= t:
                break
            if len(acts):
                soft_target[i, j, acts] = probs

    is_trunc = torch.tensor([it[3] for it in items], dtype=torch.bool)
    mlh_valid = valid & ~is_trunc.unsqueeze(1)

    batch_obj = TrainBatch(
        features=features,
        actions=actions,
        legal_mask=legal,
        results=results,
        moves_left=moves_left,
        elo_weight=elo_w,
        tc_bucket=tc,
        elo_std=elo_std,
        color=color,
    )

    return {
        "batch": batch_obj,
        "valid": valid,
        "mlh_valid": mlh_valid,
        "policy_soft_target": torch.from_numpy(soft_target),
        "b": b,
        "t": t,
        "real_lens": real_lens,
    }


def compute_batch_memory_mb(batch_dict: dict[str, Any]) -> float:
    """Calculate total tensor memory footprint in Megabytes."""
    total_bytes = 0
    tb = batch_dict["batch"]
    for field in (
        "features", "actions", "legal_mask", "results",
        "moves_left", "elo_weight", "tc_bucket", "elo_std", "color"
    ):
        t = getattr(tb, field)
        if isinstance(t, torch.Tensor):
            total_bytes += t.element_size() * t.nelement()

    for k in ("valid", "mlh_valid", "policy_soft_target"):
        t = batch_dict.get(k)
        if isinstance(t, torch.Tensor):
            total_bytes += t.element_size() * t.nelement()

    return float(total_bytes / (1024 * 1024))


def run_benchmark():
    shard_dir = os.path.join(REPO_ROOT, "data", "shards_real_v3")
    print(f"Loading V3 reader from {shard_dir}...")
    reader = V3ShardReader(shard_dir)
    n_games = len(reader.meta_all)
    lengths = reader.meta_all["n_plies"].astype(np.int64)
    print(f"Loaded {n_games} games. Min len: {lengths.min()}, Max: {lengths.max()}, Mean: {lengths.mean():.2f}")

    # Preload all raw replayed game items into memory to isolate batching/collating/prefetch logic
    # and avoid filesystem jitter
    print("Pre-replaying 100 games into memory cache...")
    t0 = time.perf_counter()
    game_cache = [build_game_item(reader, i, t_max=B2_T_MAX) for i in range(n_games)]
    preload_dur = time.perf_counter() - t0
    print(f"Pre-replayed {n_games} games in {preload_dur:.3f}s ({n_games / preload_dur:.1f} games/s)")

    batch_sizes = [4, 8, 16, 32]
    strategies = [
        ("Strategy A (Naive Fixed T=300)", "A_fixed_300"),
        ("Strategy B (Dynamic Batch Max T_batch <= 300)", "B_dynamic_max"),
        ("Strategy C (Length-sorted Dynamic Batching)", "C_sorted_chunk"),
    ]

    num_iters = 100
    results = {
        "dataset_info": {
            "shard_dir": shard_dir,
            "num_games": n_games,
            "min_plies": int(lengths.min()),
            "max_plies": int(lengths.max()),
            "mean_plies": float(lengths.mean()),
        },
        "configs": [],
    }

    print("\n" + "=" * 105)
    print(f"{'Strategy':<38} | {'B':<3} | {'Waste (%)':<9} | {'T_pad':<6} | {'Prep (ms)':<9} | {'Throughput (g/s)':<16} | {'Throughput (p/s)':<16} | {'Mem (MB)':<8}")
    print("-" * 105)

    rng = np.random.default_rng(20260921)

    for strat_label, strat_key in strategies:
        for b in batch_sizes:
            latencies_ms = []
            waste_ratios = []
            t_padded_list = []
            mem_mb_list = []
            total_real_plies = 0
            total_games_proc = 0

            # Prepare batches for 100 iterations
            # In Strategy A & B: random sample of size b from 0..n_games-1
            # In Strategy C: length-sorted chunks of size b
            t_start_all = time.perf_counter()

            for it in range(num_iters):
                t_prep_start = time.perf_counter()

                if strat_key in ("A_fixed_300", "B_dynamic_max"):
                    indices = rng.choice(n_games, size=b, replace=True)
                    items = [game_cache[idx] for idx in indices]
                    target_t = 300 if strat_key == "A_fixed_300" else None
                elif strat_key == "C_sorted_chunk":
                    # Length-sorted dynamic chunking
                    # Pick an anchor game or sorted slice
                    sort_order = np.argsort(lengths, kind="stable")
                    # Choose a random window of size b in the sorted order
                    start_idx = rng.integers(0, max(1, n_games - b + 1))
                    chunk_indices = sort_order[start_idx : start_idx + b]
                    if len(chunk_indices) < b:
                        # Wrap around if needed
                        wrap = sort_order[: b - len(chunk_indices)]
                        chunk_indices = np.concatenate([chunk_indices, wrap])
                    items = [game_cache[idx] for idx in chunk_indices]
                    target_t = None

                batch_res = collate_microbatch(items, target_t=target_t)
                t_prep_end = time.perf_counter()

                prep_ms = (t_prep_end - t_prep_start) * 1000.0
                latencies_ms.append(prep_ms)

                real_plies = int(batch_res["real_lens"].sum())
                t_padded = int(batch_res["t"])
                t_padded_list.append(t_padded)
                total_real_plies += real_plies
                total_games_proc += b

                capacity = b * t_padded
                waste = 1.0 - (real_plies / capacity)
                waste_ratios.append(waste)

                mem_mb = compute_batch_memory_mb(batch_res)
                mem_mb_list.append(mem_mb)

            total_elapsed = time.perf_counter() - t_start_all
            games_per_sec = total_games_proc / total_elapsed
            plies_per_sec = total_real_plies / total_elapsed

            mean_waste = float(np.mean(waste_ratios) * 100.0)
            mean_t_pad = float(np.mean(t_padded_list))
            mean_lat = float(np.mean(latencies_ms))
            p50_lat = float(np.percentile(latencies_ms, 50))
            p90_lat = float(np.percentile(latencies_ms, 90))
            p99_lat = float(np.percentile(latencies_ms, 99))
            mean_mem = float(np.mean(mem_mb_list))

            res_entry = {
                "strategy": strat_key,
                "strategy_label": strat_label,
                "microbatch": b,
                "iterations": num_iters,
                "padding_waste_pct": round(mean_waste, 2),
                "avg_t_padded": round(mean_t_pad, 1),
                "prep_latency_ms": {
                    "mean": round(mean_lat, 2),
                    "p50": round(p50_lat, 2),
                    "p90": round(p90_lat, 2),
                    "p99": round(p99_lat, 2),
                },
                "throughput": {
                    "games_per_sec": round(games_per_sec, 2),
                    "plies_per_sec": round(plies_per_sec, 2),
                },
                "memory_footprint_mb": round(mean_mem, 2),
            }
            results["configs"].append(res_entry)

            print(
                f"{strat_label:<38} | {b:<3} | {mean_waste:>8.2f}% | {mean_t_pad:>6.1f} | {mean_lat:>7.2f} ms | "
                f"{games_per_sec:>16.1f} | {plies_per_sec:>16.1f} | {mean_mem:>7.2f}"
            )

    print("=" * 105)

    # Save to runs/dataloader_benchmark.json
    out_dir = os.path.join(REPO_ROOT, "runs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "dataloader_benchmark.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved benchmark results to: {out_file}")
    print(f"File size: {os.path.getsize(out_file)} bytes")


if __name__ == "__main__":
    run_benchmark()
