"""Stage B Gumbel 自对弈生成器（规格 §2.4 / §2.3 / §2.5）——驱动在 UniChessKit。

对局循环、跨局 GPU 拼批、开局注入由 ``unichess_kit.pipelines.selfplay.run_selfplay`` 负责；
S 侧只提供 Player（``stateseq.kit_adapter.SsmSelfPlayer``：R cache 懒追赶 + 路径重算展开 +
Gumbel 顺序减半 + book ply 的 π′ 跨局共享）与 RecordSink（``V3Sink`` → v3 分片）。
切换前与原生成器（``Driver`` / ``GameState``，见 git 历史）在并发 1 下逐字节对照一致
（git 历史中 ``93f1ca8`` 的 ``tests/test_kit_selfplay.py`` 与 ``tools/kit_selfplay_parity.py``：
随机权重与真实权重 gen3 8 局 598 ply 均逐字节一致）。

多进程（``--workers N``）：各进程取不相交的全局局序号区间（``--first-game``）、同一 seed，
局键 / 开局 / 随机数流只取决于全局序号——拆给几个进程都是同一批对局（并发 1 时逐字节相同）。
每代局数：首轮闭环 2k–5k；主循环 25k/代。
输出：v3 分片（actions + pipol + 扩展 meta）+ manifest（含 gen 节生成统计）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

sys.stdout.reconfigure(line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
KIT_ROOT = os.environ.get("UNICHESS_KIT_ROOT", os.path.join(os.path.dirname(HERE), "Kit"))
if os.path.isdir(KIT_ROOT) and KIT_ROOT not in sys.path:
    sys.path.append(KIT_ROOT)  # 追加而非前插：Kit 的 tests 包不得遮蔽本仓库的 tests

from stateseq.conditions import TimeControlBucket  # noqa: E402
from stateseq.data.gshards import V3ShardWriter  # noqa: E402
from stateseq.depth_hist import hist_merge, hist_summary  # noqa: E402
from stateseq.gumbel import C_SCALE, C_VISIT, TERM_CODES  # noqa: E402
from stateseq.kit_adapter import V3Sink, book_pipol_rng, make_selfplay_factory  # noqa: E402,F401
from unichess_kit.pipelines.selfplay import SelfPlayConfig as KitSelfPlayConfig  # noqa: E402
from unichess_kit.pipelines.selfplay import run_selfplay  # noqa: E402


# ------------------------- 配置 -------------------------

@dataclass
class SelfPlayConfig:
    ckpt: str
    out_dir: str
    tag: str
    num_games: int = 2000
    concurrency: int = 128
    n_sims: int = 256
    m0: int = 16
    seed: int = 42
    c_visit: float = C_VISIT
    c_scale: float = C_SCALE
    max_plies: int = 300  # 整盘上限（含 book）
    gen_id: int = 1
    ckpt_step: int = 0
    device: str = "cuda"  # mamba 步进核只有 CUDA 实现
    elo: float = 2567.5
    tc_bucket: TimeControlBucket = TimeControlBucket.RAPID
    gumbel_g: float = 1.0  # Gumbel 噪声尺度；评测/换代 arena 用 g=0
    openings_path: str = ""
    book_plies: int = 6  # 开局注入 ply 数（着法仍走 book，π′ 由搜索产生并跨局共享）
    first_game: int = 0  # 全局局序号起点（多进程分片）


# ------------------------- 生成主循环 -------------------------

def generate(cfg: SelfPlayConfig, progress_every: int = 20) -> dict:
    writer = V3ShardWriter(cfg.out_dir, cfg.tag)
    factory = make_selfplay_factory(cfg.ckpt, device=cfg.device, simulations=cfg.n_sims,
                                    m0=cfg.m0, g=cfg.gumbel_g, c_visit=cfg.c_visit,
                                    c_scale=cfg.c_scale)
    sink = V3Sink(writer, gen_id=cfg.gen_id, ckpt_step=cfg.ckpt_step, elo=cfg.elo,
                  tc_bucket=cfg.tc_bucket)
    kcfg = KitSelfPlayConfig(games=cfg.num_games, seed=cfg.seed, max_plies=cfg.max_plies,
                             concurrency=cfg.concurrency, openings=cfg.openings_path or None,
                             book_plies=cfg.book_plies, first_game=cfg.first_game)
    t0 = time.time()

    def progress(_record: dict, done: int) -> None:
        if done % progress_every == 0 or done == cfg.num_games:
            elapsed = time.time() - t0
            print(f"[{elapsed:7.1f}s] 已完成 {done}/{cfg.num_games} 局，"
                  f"{done / max(elapsed, 1e-6):.3f} games/s")

    summary = run_selfplay(kcfg, factory, sink, progress=progress)
    writer.flush()
    elapsed = time.time() - t0
    if cfg.openings_path:
        print(f"开局库 {summary['opening_lines']} 条（来自 {cfg.openings_path}，每条裁至 "
              f"{cfg.book_plies} ply，丢弃非法线 {summary['opening_lines_dropped']} 条）")

    sh = factory.shared
    plies = sink.plies
    stats = {
        "games": sink.games,
        "plies": plies,
        "elapsed_s": elapsed,
        "games_per_s": sink.games / max(elapsed, 1e-6),
        "plies_per_s": plies / max(elapsed, 1e-6),
        "avg_search_nodes_per_ply": sh.n_nodes / max(plies, 1),
        "avg_sims_per_ply": sh.sims / max(plies, 1),
        "avg_max_tree_depth": sh.max_depth / max(plies, 1),
        "expand_depth": hist_summary(sh.expand_hist),
        "budget_violations": sh.budget_violations,
        "termination_reason_counts": dict(zip(TERM_CODES, sink.term_reason_counts)),
        "truncated_rate": sink.truncated_games / max(sink.games, 1),
        "book_memo_hits": sh.book_memo_hits,
        "book_memo_misses": sh.book_memo_misses,
        "book_memo_hit_rate": sh.book_memo_hits / max(sh.book_memo_hits + sh.book_memo_misses, 1),
        "concurrency": cfg.concurrency,
        "n_sims": cfg.n_sims,
        "m0": cfg.m0,
        "gen_id": cfg.gen_id,
        "ckpt_step": cfg.ckpt_step,
        "c_visit": cfg.c_visit,
        "c_scale": cfg.c_scale,
        "book_plies": cfg.book_plies,
        "max_plies": cfg.max_plies,
        # 复现一批数据需要 seed + first_game + games（与进程数无关）
        "seed": cfg.seed,
        "first_game": cfg.first_game,
        "model_forwards": factory.evaluator.n_forwards,
        "batch": summary["batch"],
    }
    manifest_path = os.path.join(cfg.out_dir, "manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except FileNotFoundError:
        manifest = {}
    manifest["gen"] = stats
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)

    print(f"生成完毕：{sink.games} 局，{elapsed:.1f}s，"
          f"{stats['games_per_s']:.3f} games/s，{stats['plies_per_s']:.2f} plies/s，"
          f"封顶率 {stats['truncated_rate']:.1%}")
    return stats


# ------------------------- 多进程编排：跨核并行（CPU 侧才是当前瓶颈） -------------------------
#
# 单进程内对局协程只把"同一进程内并发局"的模型前向拼批，树逻辑仍是单核 Python，GPU 因而
# 长期低利用率。这里起 N 个独立 OS 进程（各自独立 CUDA context，避免 CUDA fork 后不安全的
# 问题），各取一段不相交的全局局序号；全部结束后把分片搬回顶层目录并合并 manifest。

def _merge_worker_outputs(out_dir: str, worker_dirs: list[str], wall_elapsed: float,
                          args: argparse.Namespace) -> dict:
    combined_shards: list[str] = []
    total_games = total_steps = total_skipped = 0
    total_plies = total_nodes = total_sims = 0
    total_max_depth = 0
    expand_hist: list[int] = []
    budget_violations = 0
    term_counts = [0] * len(TERM_CODES)
    truncated_games = 0
    book_hits = book_misses = 0
    last_cfg: dict = {}

    for wd in worker_dirs:
        mpath = os.path.join(wd, "manifest.json")
        with open(mpath, encoding="utf-8") as fh:
            wm = json.load(fh)
        for shard in wm.get("shards", []):
            for ext in (".actions.bin", ".meta.npz", ".pipol.bin", ".pipol.offsets.bin"):
                src = os.path.join(wd, shard + ext)
                if os.path.exists(src):
                    shutil.move(src, os.path.join(out_dir, shard + ext))
            combined_shards.append(shard)
        total_games += wm.get("games", 0)
        total_steps += wm.get("steps", 0)
        total_skipped += wm.get("skipped", 0)
        gen = wm.get("gen", {})
        total_plies += gen.get("plies", 0)
        total_nodes += gen.get("avg_search_nodes_per_ply", 0.0) * gen.get("plies", 0)
        total_sims += gen.get("avg_sims_per_ply", 0.0) * gen.get("plies", 0)
        total_max_depth += gen.get("avg_max_tree_depth", 0.0) * gen.get("plies", 0)
        expand_hist = hist_merge(expand_hist, gen.get("expand_depth", {}).get("hist"))
        budget_violations += gen.get("budget_violations", 0)
        for k, v in gen.get("termination_reason_counts", {}).items():
            term_counts[TERM_CODES.index(k)] += v
        truncated_games += round(gen.get("truncated_rate", 0.0) * gen.get("games", 0))
        book_hits += gen.get("book_memo_hits", 0)
        book_misses += gen.get("book_memo_misses", 0)
        last_cfg = {k: gen.get(k) for k in ("concurrency", "n_sims", "m0", "gen_id", "ckpt_step",
                                            "c_visit", "c_scale", "book_plies", "max_plies")
                    if k in gen}
        shutil.rmtree(wd, ignore_errors=True)

    stats = {
        "games": total_games,
        "plies": total_plies,
        "elapsed_s": wall_elapsed,
        "games_per_s": total_games / max(wall_elapsed, 1e-6),
        "plies_per_s": total_plies / max(wall_elapsed, 1e-6),
        "avg_search_nodes_per_ply": total_nodes / max(total_plies, 1),
        "avg_sims_per_ply": total_sims / max(total_plies, 1),
        "avg_max_tree_depth": total_max_depth / max(total_plies, 1),
        "expand_depth": hist_summary(expand_hist),
        "budget_violations": budget_violations,
        "termination_reason_counts": dict(zip(TERM_CODES, term_counts)),
        "truncated_rate": truncated_games / max(total_games, 1),
        "book_memo_hits": book_hits,
        "book_memo_misses": book_misses,
        "book_memo_hit_rate": book_hits / max(book_hits + book_misses, 1),
        "workers": len(worker_dirs),
        "seed": getattr(args, "seed", None),
        "first_game": getattr(args, "first_game", 0),
        **last_cfg,
    }
    manifest = {"shards": combined_shards, "months": [], "games": total_games,
                "steps": total_steps, "skipped": total_skipped, "gen": stats}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    return stats


def worker_ranges(games: int, workers: int, first_game: int = 0) -> list[tuple[int, int]]:
    """把 [first_game, first_game+games) 切成 workers 段 → [(起点, 局数)]（局数可为 0）。"""
    base, rem = divmod(games, workers)
    out, start = [], first_game
    for i in range(workers):
        n = base + (1 if i < rem else 0)
        out.append((start, n))
        start += n
    return out


def run_workers(args: argparse.Namespace) -> None:
    n = args.workers
    os.makedirs(args.out, exist_ok=True)
    procs = []
    t0 = time.time()
    for i, (first, n_games) in enumerate(worker_ranges(args.games, n, args.first_game)):
        if n_games == 0:
            continue
        wdir = os.path.join(args.out, f"_w{i}")
        os.makedirs(wdir, exist_ok=True)
        cmd = [sys.executable, os.path.abspath(__file__),
               "--ckpt", args.ckpt, "--out", wdir, "--tag", f"{args.tag}-w{i}",
               "--games", str(n_games), "--first-game", str(first),
               "--concurrency", str(args.concurrency),
               "--n_sims", str(args.n_sims), "--m0", str(args.m0),
               "--seed", str(args.seed), "--max_plies", str(args.max_plies),
               "--gen_id", str(args.gen_id), "--ckpt_step", str(args.ckpt_step),
               "--g", str(args.g),
               "--c_visit", str(args.c_visit), "--c_scale", str(args.c_scale),
               "--book-plies", str(args.book_plies)]
        if args.openings:
            cmd.extend(["--openings", args.openings])
        log_path = os.path.join(wdir, "worker.log")
        log_fh = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        procs.append((proc, log_fh, wdir))
        print(f"[worker {i}] 启动 pid={proc.pid} 局 [{first}, {first + n_games})", flush=True)

    failed = []
    for i, (proc, log_fh, wdir) in enumerate(procs):
        rc = proc.wait()
        log_fh.close()
        print(f"[worker {i}] 退出码 {rc}", flush=True)
        if rc != 0:
            failed.append((i, wdir))
    if failed:
        raise RuntimeError(f"worker 进程失败：{failed}；查看各自 worker.log 排查后重试，"
                           f"不合并已产出的部分分片（避免正式数据混入未验证的失败批次）")

    elapsed = time.time() - t0
    stats = _merge_worker_outputs(args.out, [wd for _, _, wd in procs], elapsed, args)
    print(f"[全部 worker 完成] {n} 进程，{stats['games']} 局，{elapsed:.1f}s，"
          f"{stats['games_per_s']:.3f} games/s，{stats['plies_per_s']:.2f} plies/s，"
          f"封顶率 {stats['truncated_rate']:.1%}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="stage_b")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--first-game", type=int, default=0,
                    help="全局局序号起点（局键/开局/随机数流按全局序号；--workers 自动分段）")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--n_sims", type=int, default=256)
    ap.add_argument("--m0", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_plies", type=int, default=300, help="整盘 ply 上限（含 book）")
    ap.add_argument("--gen_id", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1,
                    help="并行 OS 进程数（跨核；每进程独立 CUDA context）。>1 时委派给 run_workers。")
    ap.add_argument("--ckpt_step", type=int, default=0)
    ap.add_argument("--g", type=float, default=1.0,
                    help="Gumbel 噪声尺度；1.0 训练/生成，0.0 评测/换代 arena")
    ap.add_argument("--c_visit", type=float, default=C_VISIT,
                    help="σ 展幅常数 c_visit；搜索与 π′ 导出共用同一值")
    ap.add_argument("--c_scale", type=float, default=C_SCALE,
                    help="σ 展幅常数 c_scale；搜索与 π′ 导出共用同一值")
    ap.add_argument("--openings", default="",
                    help="开局着法文件路径（每行 SAN/UCI 着法序列，如 data/openings_200.txt）")
    ap.add_argument("--book-plies", type=int, default=6,
                    help="开局注入 ply 数：着法走 book，π′ 由搜索产生并按 (开局, ply) 跨局共享")
    args = ap.parse_args()

    if args.workers > 1:
        run_workers(args)
        return

    cfg = SelfPlayConfig(
        ckpt=args.ckpt,
        out_dir=args.out,
        tag=args.tag,
        num_games=args.games,
        concurrency=args.concurrency,
        n_sims=args.n_sims,
        m0=args.m0,
        seed=args.seed,
        max_plies=args.max_plies,
        gen_id=args.gen_id,
        ckpt_step=args.ckpt_step,
        gumbel_g=args.g,
        c_visit=args.c_visit,
        c_scale=args.c_scale,
        openings_path=args.openings,
        book_plies=args.book_plies,
        first_game=args.first_game,
    )
    generate(cfg)


if __name__ == "__main__":
    main()
