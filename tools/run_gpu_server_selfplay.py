"""SSM 多 Worker + GpuServer 自对弈并发生成脚本。

使用 P4 共享 GPU 服务架构：
1. 启动 GpuServer（/dev/shm 共享内存 + FIFO，CUDA Graph 批不变模式），独占 GPU；
2. 启动 N 个独立的纯 CPU Worker 进程，每个 Worker 跑一个独立 OS 进程（完全绕开 GIL）；
3. 各 Worker 通过 IPC 向 GpuServer 发送 EvalRequest，GpuServer 聚合拼大批；
4. 每个 Worker 负责 500 局，生成 1 个标准分片；
5. 全部 Worker 完成后，主脚本聚合切片到主目录，更新 manifest.json 与 .games.jsonl。
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

# 确保导入根在 sys.path
IMPORT_ROOT = os.environ.get("UNICHESS_IMPORT_ROOT", str(Path(__file__).resolve().parents[2]))
if IMPORT_ROOT not in sys.path:
    sys.path.insert(0, IMPORT_ROOT)

from SSM.infer.gpu_server import GpuServer


def main():
    parser = argparse.ArgumentParser(description="SSM GpuServer Multi-Worker Selfplay")
    parser.add_argument("--ckpt", default=os.path.join(IMPORT_ROOT, "SSM/runs/champion.pt"))
    parser.add_argument("--out", default=os.path.join(IMPORT_ROOT, "SSM/runs/stage_b_gen_5000_cs002"))
    parser.add_argument("--first-game", type=int, default=2000)
    parser.add_argument("--total-games", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--m0", type=int, default=16)
    parser.add_argument("--c-scale", type=float, default=0.02)
    parser.add_argument("--c-visit", type=float, default=50.0)
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--openings", default=os.path.join(IMPORT_ROOT, "SSM/data/openings_200.txt"))
    parser.add_argument("--book-plies", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--slots-per-client", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--opp-ckpt", default=None, help="对手模型检查点（若给出则开启跨代对弈）")
    parser.add_argument("--temp-plies", type=int, default=15, help="前 N 步启用温度退火采样")
    parser.add_argument("--temperature", type=float, default=1.0, help="初始采样温度")
    parser.add_argument("--min-book-plies", type=int, default=6, help="开局随机截断最小深度")
    parser.add_argument("--pcr-rate", type=float, default=0.5, help="快速步 PCR 比例 (0.0~1.0)")
    parser.add_argument("--pcr-fast-sims", type=int, default=16, help="快速步模拟数")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    games_per_worker = args.total_games // args.workers
    if args.total_games % args.workers != 0:
        raise ValueError(f"total_games ({args.total_games}) 必须能被 workers ({args.workers}) 整除")

    ckpts = [os.path.abspath(args.ckpt)]
    if args.opp_ckpt:
        ckpts.append(os.path.abspath(args.opp_ckpt))

    print(f"[GpuServer] 正在启动 GPU 共享服务 (models={len(ckpts)}, clients={args.workers}, slots={args.slots_per_client}, chunk={args.chunk})...")
    server = GpuServer(
        checkpoints=ckpts,
        n_clients=args.workers,
        slots_per_client=args.slots_per_client,
        chunk=args.chunk,
        precision=args.precision,
    ).start()
    print(f"[GpuServer] 服务已就绪：dir={server.dir}")

    t0 = time.time()
    procs = []
    worker_dirs = []

    import signal

    def _kill_children(*_a):
        print("\n[Master] 收到终止信号，正在关闭所有 Worker 与 GpuServer...")
        for _i, _p, _log_fh, _wdir, _w_first, _n_games in procs:
            if _p.poll() is None:
                _p.terminate()
        server.stop()
        sys.exit(143)

    signal.signal(signal.SIGTERM, _kill_children)
    signal.signal(signal.SIGINT, _kill_children)

    try:
        for i in range(args.workers):
            w_first = args.first_game + i * games_per_worker
            wdir = os.path.join(args.out, f"_w{i}")
            if os.path.exists(wdir):
                shutil.rmtree(wdir)
            os.makedirs(wdir, exist_ok=True)
            worker_dirs.append((i, wdir, w_first, games_per_worker))

            engine_kw = {
                "checkpoint": os.path.abspath(args.ckpt),
                "simulations": args.simulations,
                "m0": args.m0,
                "c_scale": args.c_scale,
                "c_visit": args.c_visit,
                "g": args.g,
                "engine": "server",
                "server_dir": server.dir,
                "server_chunk": args.chunk,
                "server_precision": args.precision,
                "temp_plies": args.temp_plies,
                "temperature": args.temperature,
                "min_book_plies": args.min_book_plies,
                "pcr_rate": args.pcr_rate,
                "pcr_fast_sims": args.pcr_fast_sims,
            }
            if args.opp_ckpt:
                engine_kw["opp_checkpoint"] = os.path.abspath(args.opp_ckpt)

            wconf = {
                "engine": {
                    "factory": "SSM.kit:make_selfplay_factory",
                    "root": IMPORT_ROOT,
                    "kwargs": engine_kw,
                },
                "selfplay": {
                    "games": games_per_worker,
                    "seed": args.seed,
                    "max_plies": 300,
                    "concurrency": args.concurrency,
                    "openings": os.path.abspath(args.openings) if args.openings else None,
                    "book_plies": args.book_plies,
                    "min_book_plies": args.min_book_plies,
                    "first_game": w_first,
                    "workers": 1,
                },
                "sink": {
                    "factory": "SSM.kit:make_v3_sink",
                    "kwargs": {
                        "out_dir": wdir,
                        "gen_id": 1,
                        "ckpt_step": 34,
                        "elo": 2567.5,
                        "shard_size": games_per_worker,
                    },
                },
            }
            cpath = os.path.join(wdir, "config.json")
            with open(cpath, "w", encoding="utf-8") as f:
                json.dump(wconf, f, indent=2, ensure_ascii=False)

            log_path = os.path.join(wdir, "worker.log")
            log_fh = open(log_path, "w", encoding="utf-8")
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "Kit",
                "selfplay",
                cpath,
            ]
            env = dict(os.environ, UNICHESS_IMPORT_ROOT=IMPORT_ROOT)
            py_path = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{IMPORT_ROOT}:{py_path}" if py_path else IMPORT_ROOT
            p = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env, cwd=IMPORT_ROOT)
            procs.append((i, p, log_fh, wdir, w_first, games_per_worker))
            print(f"[Worker {i}] 已启动 (PID={p.pid})：局号区间 [{w_first}, {w_first + games_per_worker})")

        # 等待所有 Worker 完成
        failed = []
        for i, p, log_fh, wdir, w_first, n_games in procs:
            rc = p.wait()
            log_fh.close()
            print(f"[Worker {i}] 运行结束，退出码={rc}")
            if rc != 0:
                failed.append((i, wdir, rc))

        if failed:
            raise RuntimeError(f"以下 Worker 发生错误：{failed}，请查看各自 worker.log 排查")

        elapsed = time.time() - t0
        print(f"[Worker] 全部 {args.workers} 个 Worker 正常结束，耗时 {elapsed:.1f} 秒！正在聚合切片...")

        # 读取主 manifest.json
        main_manifest_path = os.path.join(args.out, "manifest.json")
        if os.path.exists(main_manifest_path):
            with open(main_manifest_path, "r", encoding="utf-8") as f:
                main_mf = json.load(f)
        else:
            main_mf = {"shards": [], "games": 0, "steps": 0}

        existing_shards = list(main_mf.get("shards", []))
        total_games = main_mf.get("games", 0)
        total_steps = main_mf.get("steps", 0)

        next_shard_idx = len(existing_shards)

        for i, wdir, w_first, n_games in worker_dirs:
            w_manifest_path = os.path.join(wdir, "manifest.json")
            with open(w_manifest_path, "r", encoding="utf-8") as f:
                w_mf = json.load(f)

            w_shards = w_mf.get("shards", [])
            for w_shard in w_shards:
                target_shard_name = f"shard-selfplay-{next_shard_idx:05d}"
                for ext in (".actions.bin", ".meta.npz", ".pipol.bin", ".pipol.offsets.bin"):
                    src = os.path.join(wdir, w_shard + ext)
                    dst = os.path.join(args.out, target_shard_name + ext)
                    shutil.move(src, dst)
                existing_shards.append(target_shard_name)
                next_shard_idx += 1

            total_games += w_mf.get("games", 0)
            total_steps += w_mf.get("steps", 0)

            # 聚合 .games.jsonl
            w_jsonl = os.path.join(wdir, ".games.jsonl")
            if os.path.exists(w_jsonl):
                with open(w_jsonl, "r", encoding="utf-8") as rf, open(
                    os.path.join(args.out, ".games.jsonl"), "a", encoding="utf-8"
                ) as wf:
                    shutil.copyfileobj(rf, wf)

            # 清理 worker 目录
            shutil.rmtree(wdir)

        # 写回更新后的主 manifest.json
        main_mf["shards"] = existing_shards
        main_mf["games"] = total_games
        main_mf["steps"] = total_steps
        with open(main_manifest_path, "w", encoding="utf-8") as f:
            json.dump(main_mf, f, indent=1, ensure_ascii=False)

        print(f"[Done] 聚合完成！目前主目录共有 {len(existing_shards)} 个分片，累计 {total_games} 局，{total_steps} 步。")

    finally:
        print("[GpuServer] 正在停止 GPU 共享服务...")
        server.stop()
        print("[GpuServer] 已停止并清理共享内存。")


if __name__ == "__main__":
    main()
