"""SSM 多 Worker + GpuServer 高性能 Arena 评测脚本。

架构设计：
1. 启动 GpuServer（/dev/shm 共享内存 + FIFO，CUDA Graph 批不变模式），托管 A 与 B 双方模型，独占 GPU；
2. 启动 N 个纯 CPU Worker 进程（利用 Intel Core Ultra 20 核 CPU 充分并行，绕开 Python GIL）；
3. 每个 Worker 负责一部分成对开局对局，通过 IPC 向 GpuServer 发送批量请求；
4. GpuServer 跨 Worker 跨对局攒大批执行前向推理；
5. 调用 Kit.pipelines.match.run_match 原生完成成对结算、SPRT 早停判定与汇总入库。
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import sys
import time
from pathlib import Path

# 确保导入根在 sys.path
IMPORT_ROOT = os.environ.get("UNICHESS_IMPORT_ROOT", str(Path(__file__).resolve().parents[2]))
if IMPORT_ROOT not in sys.path:
    sys.path.insert(0, IMPORT_ROOT)

from Kit.pipelines.match import MatchConfig, run_match
from Kit.registry import EngineSpec
from SSM.infer.gpu_server import GpuServer


def main():
    parser = argparse.ArgumentParser(description="SSM GpuServer Multi-Worker Match")
    parser.add_argument("config", help="评测配置文件路径 (JSON)")
    parser.add_argument("--out", default=None, help="结果输出路径 (.jsonl)")
    parser.add_argument("--workers", type=int, default=6, help="纯 CPU Worker 进程数 (默认 6)")
    parser.add_argument("--concurrency", type=int, default=8, help="每 Worker 并发协程数 (默认 8，总并发 48)")
    parser.add_argument("--chunk", type=int, default=64, help="CUDA Graph 批块大小 (默认 64)")
    parser.add_argument("--slots-per-client", type=int, default=1024, help="每客户端状态槽上限 (默认 1024)")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "tf32"], help="推理精度 (fp32/tf32)")
    parser.add_argument("--pairs", type=int, default=None, help="覆盖对局对数 (可选)")
    parser.add_argument("--simulations", type=int, default=None, help="覆盖搜索模拟数 (可选)")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        print(f"[MatchServer] 配置文件不存在：{config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        conf = json.load(f)

    spec_a_dict = conf["a"]
    spec_b_dict = conf["b"]
    match_dict = dict(conf["match"])

    if args.pairs is not None:
        match_dict["pairs"] = args.pairs
    if args.simulations is not None:
        match_dict["simulations"] = args.simulations
    match_dict["workers"] = args.workers
    match_dict["concurrency"] = args.concurrency

    cfg = MatchConfig.from_dict(match_dict)
    spec_a = EngineSpec.from_dict(spec_a_dict)
    spec_b = EngineSpec.from_dict(spec_b_dict)

    # 提取双方检查点
    ckpt_a = spec_a.kwargs.get("checkpoint")
    ckpt_b = spec_b.kwargs.get("checkpoint")

    # 判定是否双方都接入 GpuServer
    use_gpu_server = False
    ckpts = []
    if spec_a.factory.startswith("SSM.") and spec_b.factory.startswith("SSM.") and ckpt_a and ckpt_b:
        use_gpu_server = True
        p_a = os.path.abspath(ckpt_a)
        p_b = os.path.abspath(ckpt_b)
        ckpts = [p_a] if p_a == p_b else [p_a, p_b]

    server = None
    if use_gpu_server:
        print(f"[MatchServer] 启动 GPU 共享服务 (models={len(ckpts)}, clients={args.workers}, "
              f"slots={args.slots_per_client}, chunk={args.chunk}, precision={args.precision})...")
        server = GpuServer(
            checkpoints=ckpts,
            n_clients=args.workers,
            slots_per_client=args.slots_per_client,
            chunk=args.chunk,
            precision=args.precision,
        ).start()
        print(f"[MatchServer] GPU 服务已就绪：dir={server.dir}")

        def _cleanup(*_):
            print("\n[MatchServer] 收到终止信号，正在关闭 GPU 服务...", file=sys.stderr)
            if server is not None:
                server.stop()
            sys.exit(143)

        signal.signal(signal.SIGTERM, _cleanup)
        signal.signal(signal.SIGINT, _cleanup)

        # 注入 runtime 调度参数，保持 identity() 与配置哈希干净稳定
        kwargs_a = dict(spec_a.kwargs)
        kwargs_a.pop("engine", None)
        runtime_a = {
            "engine": "server",
            "server_dir": server.dir,
            "server_chunk": args.chunk,
            "server_precision": args.precision,
        }
        spec_a = dataclasses.replace(spec_a, kwargs=kwargs_a, runtime=runtime_a)

        kwargs_b = dict(spec_b.kwargs)
        kwargs_b.pop("engine", None)
        runtime_b = {
            "engine": "server",
            "server_dir": server.dir,
            "server_chunk": args.chunk,
            "server_precision": args.precision,
        }
        spec_b = dataclasses.replace(spec_b, kwargs=kwargs_b, runtime=runtime_b)

    out_path = Path(args.out) if args.out else None
    t0 = time.perf_counter()
    try:
        print(f"[MatchServer] 启动成对对局评测：pairs={cfg.pairs} (共 {2 * cfg.pairs} 局), "
              f"workers={cfg.workers}, concurrency={cfg.concurrency}...")
        summary = run_match(cfg, spec_a=spec_a, spec_b=spec_b, out_path=out_path)
        elapsed = time.perf_counter() - t0
        print(f"\n[MatchServer] 评测圆满完成！耗时: {elapsed:.1f}s ({elapsed / summary['games']:.3f}s/局)")
        print(f"战绩: A ({summary['a']}) {summary['a_wins']} 胜 / {summary['b_wins']} 负 / {summary['draws']} 和")
        print(f"得分率: {summary['score_a']:.4f}  Elo: {summary['elo']:+.2f} (95% CI: [{summary['elo_ci95'][0]:+.2f}, {summary['elo_ci95'][1]:+.2f}])")
        print(f"优势概率 (LOS): {summary['los']:.4f}")
        return summary
    finally:
        if server is not None:
            server.stop()
            print("[MatchServer] GPU 服务已释放。")


if __name__ == "__main__":
    main()
