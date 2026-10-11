"""搜索强度扫描（sims-sweep）：同一对检查点在多个模拟数下与同一对手对弈。

用途（计划 §5.3 的主判据）：价值头改造是否生效，看的是**搜索曲线的形状**而不是单点胜负——
若 0-sim 持平而 32/64-sim 上移，说明 MCTS 回溯用的价值评估变好了；全线不动则说明无效。
0-sim（纯网络直觉）那一点用 ``tools/evaluate_raw_policy.py`` 单独测，本工具覆盖 sims ≥ 1。

做法：GpuServer 只起一次（模型加载才是贵的那步），在其上按 sims 列表逐点跑
``Kit.pipelines.match.run_match``；每点关闭 SPRT 早停（固定局数才好横向比较），
结果写到 ``<out_stem>_sims<N>.jsonl`` 并汇总成一个 JSON。
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


def _parse_sims(text: str) -> list:
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            v = int(tok)
            if v < 1:
                raise ValueError(f"simulations 必须 >= 1（0-sim 用 evaluate_raw_policy.py）：{v}")
            out.append(v)
    if not out:
        raise ValueError("--sims 不能为空")
    return out


def main():
    parser = argparse.ArgumentParser(description="SSM 搜索强度扫描（sims-sweep）")
    parser.add_argument("config", help="评测配置 JSON（{\"a\":..., \"b\":..., \"match\":...}）")
    parser.add_argument("--sims", default="8,32,64", help="逗号分隔的模拟数列表（默认 8,32,64）")
    parser.add_argument("--pairs", type=int, default=50, help="每个 sims 点的成对开局数（默认 50 = 100 局）")
    parser.add_argument("--out", default=None, help="汇总 JSON 输出路径（默认 <config>_sims_sweep.json）")
    parser.add_argument("--workers", type=int, default=6, help="纯 CPU Worker 进程数（默认 6）")
    parser.add_argument("--concurrency", type=int, default=8, help="每 Worker 并发协程数（默认 8）")
    parser.add_argument("--chunk", type=int, default=64, help="CUDA Graph 批块大小（默认 64）")
    parser.add_argument("--slots-per-client", type=int, default=1024, help="每客户端状态槽上限")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "tf32"], help="推理精度")
    args = parser.parse_args()

    sims_list = _parse_sims(args.sims)
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        print(f"[SimsSweep] 配置文件不存在：{config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path, encoding="utf-8") as fh:
        conf = json.load(fh)

    spec_a = EngineSpec.from_dict(conf["a"])
    spec_b = EngineSpec.from_dict(conf["b"])
    match_dict = dict(conf["match"])
    match_dict["pairs"] = args.pairs
    match_dict["workers"] = args.workers
    match_dict["concurrency"] = args.concurrency
    match_dict.pop("sprt", None)          # 扫描点固定局数，不做序贯早停
    base_cfg = MatchConfig.from_dict(match_dict)

    ckpt_a = spec_a.kwargs.get("checkpoint")
    ckpt_b = spec_b.kwargs.get("checkpoint")
    use_gpu_server = (spec_a.factory.startswith("SSM.") and spec_b.factory.startswith("SSM.")
                      and ckpt_a and ckpt_b)

    out_base = (Path(args.out) if args.out else
                config_path.with_name(config_path.stem + "_sims_sweep.json"))
    out_base.parent.mkdir(parents=True, exist_ok=True)

    server = None
    if use_gpu_server:
        p_a, p_b = os.path.abspath(ckpt_a), os.path.abspath(ckpt_b)
        ckpts = [p_a] if p_a == p_b else [p_a, p_b]
        print(f"[SimsSweep] 启动 GPU 共享服务 (models={len(ckpts)}, clients={args.workers}, "
              f"slots={args.slots_per_client}, chunk={args.chunk}, precision={args.precision})...")
        server = GpuServer(checkpoints=ckpts, n_clients=args.workers,
                           slots_per_client=args.slots_per_client, chunk=args.chunk,
                           precision=args.precision).start()

        def _cleanup(*_):
            print("\n[SimsSweep] 收到终止信号，正在关闭 GPU 服务...", file=sys.stderr)
            if server is not None:
                server.stop()
            sys.exit(143)

        signal.signal(signal.SIGTERM, _cleanup)
        signal.signal(signal.SIGINT, _cleanup)

        runtime = {"engine": "server", "server_dir": server.dir,
                   "server_chunk": args.chunk, "server_precision": args.precision}
        kw_a = {k: v for k, v in spec_a.kwargs.items() if k != "engine"}
        kw_b = {k: v for k, v in spec_b.kwargs.items() if k != "engine"}
        spec_a = dataclasses.replace(spec_a, kwargs=kw_a, runtime=runtime)
        spec_b = dataclasses.replace(spec_b, kwargs=kw_b, runtime=runtime)

    rows = []
    try:
        for sims in sims_list:
            cfg = dataclasses.replace(base_cfg, simulations=sims)
            point_out = out_base.with_name(f"{out_base.stem}_sims{sims}.jsonl")
            t0 = time.perf_counter()
            print(f"\n[SimsSweep] === sims={sims}：pairs={cfg.pairs}（{2 * cfg.pairs} 局）===")
            summary = run_match(cfg, spec_a=spec_a, spec_b=spec_b, out_path=point_out)
            elapsed = time.perf_counter() - t0
            row = {
                "simulations": sims,
                "games": summary["games"],
                "a_wins": summary["a_wins"], "b_wins": summary["b_wins"],
                "draws": summary["draws"],
                "score_a": summary["score_a"], "elo": summary["elo"],
                "elo_ci95": summary["elo_ci95"], "los": summary["los"],
                "elapsed_s": round(elapsed, 1),
                "results": str(point_out),
            }
            rows.append(row)
            print(f"[SimsSweep] sims={sims}: 得分率 {row['score_a']:.4f}  "
                  f"Elo {row['elo']:+.2f} (CI [{row['elo_ci95'][0]:+.2f}, {row['elo_ci95'][1]:+.2f}])  "
                  f"LOS {row['los']:.4f}  ({elapsed:.1f}s)")
    finally:
        if server is not None:
            server.stop()
            print("[SimsSweep] GPU 服务已释放。")

    print("\n[SimsSweep] 汇总：")
    print(f"{'sims':>6} {'局数':>6} {'得分率':>9} {'Elo':>9} {'LOS':>8} {'耗时s':>8}")
    for r in rows:
        print(f"{r['simulations']:>6} {r['games']:>6} {r['score_a']:>9.4f} "
              f"{r['elo']:>+9.2f} {r['los']:>8.4f} {r['elapsed_s']:>8.1f}")
    payload = {
        "a": conf["a"], "b": conf["b"],
        "pairs": args.pairs, "workers": args.workers, "concurrency": args.concurrency,
        "precision": args.precision, "points": rows,
    }
    with open(out_base, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"\n[SimsSweep] 汇总已写入 {out_base}")
    return rows


if __name__ == "__main__":
    main()
