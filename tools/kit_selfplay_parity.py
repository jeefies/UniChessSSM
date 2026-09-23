"""P3 验收：kit 驱动的 S 自对弈与原生成器 ``ssm_gumbel_selfplay.Driver`` 逐字节对照。

同一权重、seed、开局库、book_plies、预算，两边各生成 games 局写 v3 分片，要求
actions / pipol / pipol.offsets 逐字节相同、meta 数组逐项相同，生成统计逐项相同。
只在并发 1 下成立（``SeqModel.step`` 随批大小有 ~1e-5 浮点差）。

用法（远端，先 nvidia-smi 确认 GPU 空闲）::

    python tools/kit_selfplay_parity.py --ckpt runs/stage_b_training_1000_gen3/best.pt \\
        --games 8 --n_sims 64 --max_plies 80 --out /tmp/kit_sp_parity
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
KIT_ROOT = os.environ.get("UNICHESS_KIT_ROOT", os.path.join(os.path.dirname(HERE), "Kit"))
if os.path.isdir(KIT_ROOT) and KIT_ROOT not in sys.path:
    sys.path.insert(0, KIT_ROOT)

import ssm_gumbel_selfplay as S  # noqa: E402
from stateseq import kit_adapter as ka  # noqa: E402
from stateseq.data.gshards import V3ShardWriter  # noqa: E402
from stateseq.depth_hist import hist_summary  # noqa: E402
from unichess_kit.pipelines.selfplay import SelfPlayConfig, run_selfplay  # noqa: E402


def compare_shard_dirs(dir_a: str, dir_b: str) -> list:
    """逐文件对照 → 差异列表（空 = 一致）。meta.npz 比数组而非字节（zip 时间戳）。"""
    names_a = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dir_a, "shard-*")))
    names_b = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dir_b, "shard-*")))
    if names_a != names_b:
        return [f"文件集不同：{names_a} vs {names_b}"]
    if not names_a:
        return ["没有分片文件"]
    diffs = []
    for name in names_a:
        pa, pb = os.path.join(dir_a, name), os.path.join(dir_b, name)
        if name.endswith(".npz"):
            with np.load(pa) as za, np.load(pb) as zb:
                for k in sorted(set(za.files) | set(zb.files)):
                    if (k not in za.files or k not in zb.files or za[k].dtype != zb[k].dtype
                            or za[k].tobytes() != zb[k].tobytes()):
                        diffs.append(f"{name}[{k}] 不同")
        else:
            with open(pa, "rb") as fa, open(pb, "rb") as fb:
                if fa.read() != fb.read():
                    diffs.append(f"{name} 字节不同")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--n_sims", type=int, default=64)
    ap.add_argument("--m0", type=int, default=16)
    ap.add_argument("--max_plies", type=int, default=80, help="整盘 ply 上限（含 book）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--book_plies", type=int, default=6)
    ap.add_argument("--gen_id", type=int, default=9)
    ap.add_argument("--openings-file", default=os.path.join(HERE, "data", "openings_200.txt"))
    ap.add_argument("--out", required=True, help="输出目录（会清空其下 s/ 与 kit/）")
    args = ap.parse_args()
    s_dir, k_dir = os.path.join(args.out, "s"), os.path.join(args.out, "kit")
    for d in (s_dir, k_dir):
        shutil.rmtree(d, ignore_errors=True)

    model = S.ModelWrapper(args.ckpt, "cuda")

    # ---- 原版：单进程 Driver，并发 1 ----
    cfg = S.SelfPlayConfig(ckpt=args.ckpt, out_dir=s_dir, tag="p", num_games=args.games,
                           concurrency=1, n_sims=args.n_sims, m0=args.m0, seed=args.seed,
                           max_plies=args.max_plies, gen_id=args.gen_id,
                           openings_path=args.openings_file, book_plies=args.book_plies)
    t0 = time.time()
    writer = V3ShardWriter(s_dir, "p")
    drv = S.Driver(model, cfg, writer, openings=S.load_openings(args.openings_file, args.book_plies))
    drv.run(progress_every=10 ** 9)
    writer.flush()
    t_s = time.time() - t0

    # ---- kit：同一模型对象 ----
    ev = ka.SsmEvaluator(model.seq, "cuda", f"S:{os.path.abspath(args.ckpt)}")
    factory = ka.make_selfplay_factory(evaluator=ev, simulations=args.n_sims, m0=args.m0, g=1.0)
    t0 = time.time()
    writer = V3ShardWriter(k_dir, "p")
    sink = ka.V3Sink(writer, gen_id=args.gen_id)
    summary = run_selfplay(SelfPlayConfig(games=args.games, seed=args.seed,
                                          max_plies=args.max_plies, concurrency=1,
                                          openings=args.openings_file,
                                          book_plies=args.book_plies), factory, sink)
    writer.flush()
    t_k = time.time() - t0

    sh = factory.shared
    checks = {
        "shards": compare_shard_dirs(s_dir, k_dir),
        "plies": [drv.total_plies, sink.plies],
        "termination": [drv.term_reason_counts, sink.term_reason_counts],
        "book_memo": [[drv.book_memo_hits, drv.book_memo_misses],
                      [sh.book_memo_hits, sh.book_memo_misses]],
        "budget_violations": [drv.budget_violations, sh.budget_violations],
        "expand_hist": [drv.expand_hist, sh.expand_hist],
        "nodes_sims_depth": [[drv.total_nodes, drv.total_sims, drv.total_max_depth],
                             [sh.n_nodes, sh.sims, sh.max_depth]],
    }
    ok = not checks["shards"] and all(v[0] == v[1] for k, v in checks.items() if k != "shards")
    result = {"ok": ok, "games": summary["games"], "plies": sink.plies,
              "s_elapsed_s": round(t_s, 1), "kit_elapsed_s": round(t_k, 1),
              "kit_forwards": ev.n_forwards, "expand_depth": hist_summary(sh.expand_hist),
              "checks": checks, "config": vars(args)}
    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print(f"{'逐字节一致' if ok else '不一致'}：{summary['games']} 局 {sink.plies} ply，"
          f"S {t_s:.0f}s / kit {t_k:.0f}s；分片差异 {checks['shards'] or '无'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
