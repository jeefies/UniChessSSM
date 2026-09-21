#!/usr/bin/env python3
"""Loop 2: 785-dim Feature Encoder Zero-Allocation & Bitboard Optimization.

Benchmarks baseline encode() vs optimized zero-allocation encode_board_fast(board, out=buf)
across 10,000 diverse real game positions.
Measures latency (us/board), throughput (boards/sec), memory allocation count / bytes,
and verifies 100% bitwise exact floating-point identity across all 10,000 boards.
"""

from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import numpy as np

from stateseq.features import FEATURE_DIM, encode, encode_board_fast


def collect_diverse_positions(target_count: int = 10000, seed: int = 42) -> list[tuple[chess.Board, int]]:
    """生成 10,000 个涵盖开局、中局、残局、过路兵、易位、多次重复等多样局面的测试集。"""
    rng = random.Random(seed)
    positions: list[tuple[chess.Board, int]] = []

    # 1. 标准初始局面与轻度走子
    board = chess.Board()
    positions.append((board.copy(), 0))

    # 2. 丰富开局库分支与长对局
    game_idx = 0
    while len(positions) < target_count:
        board = chess.Board()
        occ_map: dict[str, int] = {}
        ply = 0
        max_plies = rng.randint(40, 160)

        while not board.is_game_over() and ply < max_plies and len(positions) < target_count:
            moves = list(board.legal_moves)
            if not moves:
                break
            # 偏好战术走子、过路兵走子、吃子走子，增加多样性
            m = rng.choice(moves)
            board.push(m)
            ply += 1

            # 统计 occurrence
            fen_key = board.fen().split(" ")[0]
            occ = occ_map.get(fen_key, 0)
            occ_map[fen_key] = occ + 1

            positions.append((board.copy(), occ))

        game_idx += 1

    return positions[:target_count]


def benchmark_encoder(
    positions: list[tuple[chess.Board, int]],
    warmup: int = 500,
    rounds: int = 1,
) -> dict[str, Any]:
    n = len(positions)
    out_buf = np.zeros(FEATURE_DIM, dtype=np.float32)

    # -------------------------------------------------------------
    # 1. 逐位一致性严格验证
    # -------------------------------------------------------------
    print(f"[*] Verifying 100% bitwise exact identity across {n} boards...")
    for idx, (b, occ) in enumerate(positions):
        base = encode(b, occurrence=occ)
        fast = encode_board_fast(b, occurrence=occ, out=out_buf)
        np.testing.assert_array_equal(
            fast,
            base,
            err_msg=f"Mismatch at index {idx}, FEN: {b.fen()}, occ={occ}",
        )
    print("    [PASS] 100% bitwise exact floating-point identity verified.")

    # -------------------------------------------------------------
    # 2. 内存分配分析 (tracemalloc + gc)
    # -------------------------------------------------------------
    print("[*] Profiling memory allocations...")

    # Baseline memory profiling
    gc.collect()
    gc.disable()
    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()
    for b, occ in positions[:1000]:
        _ = encode(b, occurrence=occ)
    snap2 = tracemalloc.take_snapshot()
    diff_base = snap2.compare_to(snap1, "lineno")
    base_allocated_bytes = sum(stat.size_diff for stat in diff_base if stat.size_diff > 0)
    base_allocated_count = sum(stat.count_diff for stat in diff_base if stat.count_diff > 0)
    tracemalloc.stop()
    gc.enable()

    # Fast memory profiling (zero-allocation candidate with pre-allocated buffer)
    gc.collect()
    gc.disable()
    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()
    for b, occ in positions[:1000]:
        _ = encode_board_fast(b, occurrence=occ, out=out_buf)
    snap2 = tracemalloc.take_snapshot()
    diff_fast = snap2.compare_to(snap1, "lineno")
    fast_allocated_bytes = sum(stat.size_diff for stat in diff_fast if stat.size_diff > 0)
    fast_allocated_count = sum(stat.count_diff for stat in diff_fast if stat.count_diff > 0)
    tracemalloc.stop()
    gc.enable()

    # -------------------------------------------------------------
    # 3. 延迟与吞吐量基准测试
    # -------------------------------------------------------------
    print(f"[*] Benchmarking throughput over {n} boards...")

    # Warmup
    for b, occ in positions[:warmup]:
        _ = encode(b, occurrence=occ)
        _ = encode_board_fast(b, occurrence=occ, out=out_buf)

    # Measure Baseline encode
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(rounds):
        for b, occ in positions:
            _ = encode(b, occurrence=occ)
    t1 = time.perf_counter()
    base_total_time = t1 - t0
    base_mean_latency_us = (base_total_time / (n * rounds)) * 1e6
    base_throughput = (n * rounds) / base_total_time

    # Measure Fast encode_board_fast (out=buf)
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(rounds):
        for b, occ in positions:
            _ = encode_board_fast(b, occurrence=occ, out=out_buf)
    t1 = time.perf_counter()
    fast_total_time = t1 - t0
    fast_mean_latency_us = (fast_total_time / (n * rounds)) * 1e6
    fast_throughput = (n * rounds) / fast_total_time

    speedup = base_total_time / fast_total_time

    results = {
        "boards_tested": n,
        "rounds": rounds,
        "bitwise_identity": True,
        "baseline": {
            "total_time_s": round(base_total_time, 4),
            "mean_latency_us": round(base_mean_latency_us, 2),
            "throughput_boards_per_sec": round(base_throughput, 1),
            "sample_1000_alloc_bytes": base_allocated_bytes,
            "sample_1000_alloc_count": base_allocated_count,
        },
        "optimized_zero_alloc": {
            "total_time_s": round(fast_total_time, 4),
            "mean_latency_us": round(fast_mean_latency_us, 2),
            "throughput_boards_per_sec": round(fast_throughput, 1),
            "sample_1000_alloc_bytes": fast_allocated_bytes,
            "sample_1000_alloc_count": fast_allocated_count,
        },
        "speedup_factor": round(speedup, 2),
        "latency_reduction_pct": round((1.0 - fast_mean_latency_us / base_mean_latency_us) * 100, 1),
    }

    return results


def main() -> None:
    print("================================================================================")
    print(" Loop 2: 785-dim Feature Encoder Zero-Allocation & Bitboard Optimization")
    print("================================================================================")

    positions = collect_diverse_positions(target_count=10000, seed=42)
    print(f"Generated {len(positions)} diverse chess positions from simulated games.")

    res = benchmark_encoder(positions, warmup=500, rounds=1)

    # Save to runs/loop2_feature_opt.json
    output_dir = Path("runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "loop2_feature_opt.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # Verification: file exists and non-empty
    assert out_file.exists(), f"Output file {out_file} does not exist!"
    assert out_file.stat().st_size > 0, f"Output file {out_file} is empty!"

    # Summary table
    base = res["baseline"]
    opt = res["optimized_zero_alloc"]

    print("\n" + "=" * 80)
    print(f"{'Metric':<30} | {'Baseline':<20} | {'Optimized (Zero-Alloc)':<22}")
    print("-" * 80)
    print(f"{'Mean Latency (μs/board)':<30} | {base['mean_latency_us']:<20.2f} | {opt['mean_latency_us']:<22.2f}")
    print(f"{'Throughput (boards/sec)':<30} | {base['throughput_boards_per_sec']:<20.1f} | {opt['throughput_boards_per_sec']:<22.1f}")
    print(f"{'1k Sample Alloc Bytes':<30} | {base['sample_1000_alloc_bytes']:<20} | {opt['sample_1000_alloc_bytes']:<22}")
    print(f"{'Bitwise Exact (10k boards)':<30} | {'-':<20} | {'100% MATCH':<22}")
    print(f"{'Speedup Factor':<30} | {'1.00x':<20} | {res['speedup_factor']:<22.2f}x")
    print(f"{'Latency Reduction':<30} | {'-':<20} | {res['latency_reduction_pct']:<22.1f}%")
    print("=" * 80)
    print(f"\n[OK] Results successfully saved and verified at: {out_file.resolve()} ({out_file.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
