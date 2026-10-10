"""并发配置横向基准测试脚本 (Concurrency Benchmark)

对比不同 Worker 数量与单 Worker 并发组合下的自对弈吞吐表现：
- 6 Workers x 24 Concurrency (144 in-flight, 当前基线)
- 4 Workers x 36 Concurrency (144 in-flight, 用户提议)
- 8 Workers x 20 Concurrency (160 in-flight, 压榨 20 核 CPU)
- 8 Workers x 24 Concurrency (192 in-flight, 超高吞吐)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

IMPORT_ROOT = os.environ.get("UNICHESS_IMPORT_ROOT", str(Path(__file__).resolve().parents[2]))
if IMPORT_ROOT not in sys.path:
    sys.path.insert(0, IMPORT_ROOT)


def run_benchmark(workers: int, concurrency: int, total_games: int, ckpt: str, opp_ckpt: str, python_bin: str) -> dict:
    out_dir = f"/tmp/bench_w{workers}_c{concurrency}"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    script_path = os.path.join(IMPORT_ROOT, "SSM/tools/run_gpu_server_selfplay.py")
    cmd = [
        python_bin,
        "-u",
        script_path,
        "--ckpt", ckpt,
        "--opp-ckpt", opp_ckpt,
        "--out", out_dir,
        "--total-games", str(total_games),
        "--first-game", "0",
        "--workers", str(workers),
        "--concurrency", str(concurrency),
        "--simulations", "64",
        "--m0", "16",
        "--c-scale", "0.02",
        "--adaptive-sims",
        "--deep-sims", "256",
        "--p-explore", "0.15",
        "--precision", "tf32",
    ]

    print(f"\n{'=' * 65}")
    print(f"【测试开始】Workers={workers}, Concurrency={concurrency} (总在途对局={workers * concurrency}, 目标={total_games}局)")
    print(f"{'=' * 65}")

    t0 = time.perf_counter()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=IMPORT_ROOT)
    t1 = time.perf_counter()
    elapsed = t1 - t0

    if res.returncode != 0:
        print(f"[错误] 退出码 {res.returncode}:\n{res.stdout}")
        return {"error": res.stdout, "workers": workers, "concurrency": concurrency}

    # 统计步数
    games_file = os.path.join(out_dir, ".games.jsonl")
    total_plies = 0
    actual_games = 0
    if os.path.exists(games_file):
        with open(games_file, "r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    g = json.loads(line)
                    total_plies += g.get("plies", 0)
                    actual_games += 1

    gps = actual_games / elapsed if elapsed > 0 else 0
    pps = total_plies / elapsed if elapsed > 0 else 0

    print(f"[完成] 耗时: {elapsed:.2f}s | 完成局数: {actual_games} | 累计步数: {total_plies}")
    print(f"       吞吐: {gps:.3f} 局/秒 ({gps * 60:.1f} 局/分) | {pps:.1f} 步/秒")

    # 清理
    shutil.rmtree(out_dir, ignore_errors=True)

    return {
        "workers": workers,
        "concurrency": concurrency,
        "in_flight": workers * concurrency,
        "games": actual_games,
        "plies": total_plies,
        "elapsed_sec": elapsed,
        "games_per_sec": gps,
        "games_per_min": gps * 60,
        "plies_per_sec": pps,
    }


def main():
    parser = argparse.ArgumentParser(description="Concurrency Benchmark for SSM Selfplay")
    parser.add_argument("--games", type=int, default=72, help="每组测试局数（建议能被 4, 6, 8 整除，如 72）")
    parser.add_argument("--ckpt", default=os.path.join(IMPORT_ROOT, "SSM/runs/champion.pt"))
    parser.add_argument("--opp-ckpt", default=os.path.join(IMPORT_ROOT, "SSM/runs/champion_gen3.pt"))
    args = parser.parse_args()

    python_bin = sys.executable

    configs = [
        (6, 24),  # 方案 A：当前默认 (144)
        (4, 36),  # 方案 B：用户提议 (144)
        (8, 20),  # 方案 C：多进程分摊 (160)
        (8, 24),  # 方案 D：激进高并发 (192)
    ]

    results = []
    for w, c in configs:
        r = run_benchmark(w, c, args.games, args.ckpt, args.opp_ckpt, python_bin)
        results.append(r)
        time.sleep(2)  # 给 GPU 和 shm 2秒释放间歇

    print("\n" + "=" * 75)
    print(f"【并发配置横向对照测试最终战报 (每组 {args.games} 局)】")
    print("=" * 75)
    header = f"| 方案 | Workers | Concurrency | 在途总数 | 耗时(秒) | 速率(局/分) | 步数/秒 | 相对基线 |"
    sep = f"|---|---|---|---|---|---|---|---|"
    print(header)
    print(sep)

    baseline_gpm = results[0].get("games_per_min", 1.0)
    for i, r in enumerate(results):
        if "error" in r:
            print(f"| 方案 {chr(65+i)} | {r['workers']} | {r['concurrency']} | ERROR | - | - | - | 失败 |")
            continue
        ratio = (r["games_per_min"] / baseline_gpm - 1.0) * 100
        ratio_str = f"{ratio:+.1f}%" if i > 0 else "基线"
        line = f"| 方案 {chr(65+i)} | {r['workers']} | {r['concurrency']} | {r['in_flight']} | {r['elapsed_sec']:.1f}s | **{r['games_per_min']:.1f}** | {r['plies_per_sec']:.1f} | **{ratio_str}** |"
        print(line)


if __name__ == "__main__":
    main()
