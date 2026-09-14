"""分片构建：.pgn.zst（可为截断前缀流）→ 动作列表分片。

单进程流水线：zstd 流式解压 → python-chess 解析过滤 → 动作 id 转换 → 分片原子写入。
解析吞吐实测 ~1.2k 局/s（单核上限），月片 12M 局约 3 小时，与下载并行足够。
Elo 统计（P1/P99 截断 + 均值/方差）随构建写入 manifest，供训练期加权/标准化。

用法：
    python tools/stateseq_build_shards.py --pgn data/raw/lichess_standard_2026-08.pgn.zst \
        --month 2026-08 --out data/shards
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import zstandard as zstd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import chess.pgn  # noqa: E402

from stateseq.actions import move_to_action  # noqa: E402
from stateseq.data.gshards import META_DTYPE, ShardBuilder, encode_game_record, make_game_key, write_manifest  # noqa: E402
from stateseq.data.pgns import game_metadata, keep_game  # noqa: E402


def iter_games_zst(path: str):
    """流式解压 + 逐局产出（截断流容忍：末尾半局丢弃）。自动识别是否 zst。"""
    if path.endswith(".zst"):
        dctx = zstd.ZstdDecompressor()
        fh = open(path, "rb")
        reader = dctx.stream_reader(fh)
        text = __import__("io").TextIOWrapper(reader, encoding="utf-8", errors="replace")
    else:
        text = open(path, encoding="utf-8", errors="replace")
    with text:
        while True:
            try:
                game = chess.pgn.read_game(text)
            except Exception:  # noqa: BLE001 - 截断尾部的半局
                return
            if game is None:
                return
            yield game


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--month", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "shards"))
    ap.add_argument("--max-games", type=int, default=0, help="0=不限（调试用）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    builder = ShardBuilder(args.out, tag=args.month)
    elos: list[float] = []
    skipped = 0
    t0 = time.time()

    for idx, game in enumerate(iter_games_zst(args.pgn)):
        if args.max_games and builder.games >= args.max_games:
            break
        meta = game_metadata(game)
        if not keep_game(game, meta):
            skipped += 1
            continue
        try:
            actions = [move_to_action(mv) for mv in game.mainline_moves()]
        except ValueError:
            skipped += 1
            continue
        if not actions:
            skipped += 1
            continue
        if meta["elo_mean"] is not None:
            elos.append(meta["elo_mean"])
        key = make_game_key(args.month, idx)
        rec_meta, rec_actions = encode_game_record(actions, meta["time_control"],
                                                   meta["result"], meta["elo_mean"], key)
        builder.add(rec_meta, rec_actions)
        if builder.games % 200000 == 0 and builder.games:
            dt = time.time() - t0
            print(f"{args.month}: {builder.games} 局 {builder.steps} 步 "
                  f"{builder.games/dt:.0f} 局/s", flush=True)

    builder.flush()
    dt = time.time() - t0

    # Elo 统计：P1/P99 截断 + 标准化参数
    elo_arr = np.asarray(elos, dtype=np.float64) if elos else np.array([1500.0])
    e_min, e_max = np.percentile(elo_arr, [1.0, 99.0])
    clipped = np.clip(elo_arr, e_min, e_max)
    elo_stats = {
        "e_min": float(e_min), "e_max": float(e_max),
        "mean": float(clipped.mean()), "std": float(max(clipped.std(), 1.0)),
        "n": int(len(elo_arr)),
    }

    # 汇总 manifest：扫描目录内全部片（跨月累积）
    all_meta = sorted(glob.glob(os.path.join(args.out, "shard-*-*.meta.npz")))
    shards = [m[: -len(".meta.npz")] for m in all_meta]
    total_games = 0
    total_steps = 0
    for base in shards:
        npz = np.load(base + ".meta.npz")
        total_games += len(npz["metas"])
        total_steps += int(npz["metas"]["n_plies"].sum())
    months = sorted({os.path.basename(s).split("-")[1] for s in shards})
    write_manifest(args.out, [os.path.basename(s) for s in shards], months,
                   {"games": total_games, "steps": total_steps, "skipped": skipped})
    # elo_stats 并入 manifest（dataset 从 manifest 读；多月构建时后写者覆盖，分布近似无妨）
    with open(os.path.join(args.out, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["elo_stats"] = elo_stats
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print(f"DONE {args.month}: 本批 {builder.games} 局 / 累计 {total_games} 局 "
          f"{total_steps} 步，跳过 {skipped}，用时 {dt/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
