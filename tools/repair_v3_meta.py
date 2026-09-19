"""修复 v3 分片的终止元信息（termination_reason / is_truncated / result）。

背景：生成器 `GameState.result()` 曾用 `is_repetition(3)` / `is_fifty_moves()`（**严格**判定）
分类终局，而对局循环用 `is_game_over(claim_draw=True)`（含"下一着可申和"）退出——两者差一 ply，
于是所有规则申和局都落到兜底分支被记成"300 ply 封顶截断"。实测 gen2k 前 400 局：218 条
truncated 中 173 条（79%）实为三次重复/五十步申和，真实封顶率 11% 而非 54%。

影响面仅限 meta 三字段：动作序列、π′ 目标、z 标签（申和局本就记和）均正确，因此**不需要重跑生成**，
按棋盘重放重算 meta 即可。本工具就地重写 `.meta.npz`（原子替换）并刷新 manifest 的 gen 统计。

用法：
    python tools/repair_v3_meta.py runs/stage_b_gen2k [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stateseq.actions import move_to_action
from stateseq.adapter import classify_final_board
from stateseq.gumbel import TERM_CODES


def _resolve_move(action: int, board: chess.Board) -> chess.Move | None:
    for m in board.legal_moves:
        if move_to_action(m) == action:
            return m
    return None


def replay_final_board(actions: np.ndarray) -> chess.Board:
    board = chess.Board()
    for a in actions:
        mv = _resolve_move(int(a), board)
        if mv is None:
            raise ValueError(f"动作 {int(a)} 在 {board.fen()} 上不合法（分片数据损坏）")
        board.push(mv)
    return board


def repair_shard(base: str, dry_run: bool) -> dict:
    npz = np.load(base + ".meta.npz")
    metas = npz["metas"].copy()
    offsets = npz["offsets"]
    pool = np.memmap(base + ".actions.bin", dtype=np.uint16, mode="r")

    stats = {"games": len(metas), "reason_changed": 0, "trunc_changed": 0,
             "result_changed": 0, "before": {}, "after": {}}
    for i in range(len(metas)):
        start, n = int(offsets[i]), int(metas[i]["n_plies"])
        board = replay_final_board(np.asarray(pool[start:start + n]))
        result, reason, is_truncated = classify_final_board(board)
        code = TERM_CODES.index(reason)
        old_reason = TERM_CODES[int(metas[i]["termination_reason"])]
        stats["before"][old_reason] = stats["before"].get(old_reason, 0) + 1
        stats["after"][reason] = stats["after"].get(reason, 0) + 1
        if int(metas[i]["termination_reason"]) != code:
            stats["reason_changed"] += 1
            metas[i]["termination_reason"] = code
        if int(metas[i]["is_truncated"]) != int(is_truncated):
            stats["trunc_changed"] += 1
            metas[i]["is_truncated"] = 1 if is_truncated else 0
        if int(metas[i]["result"]) != result:
            stats["result_changed"] += 1
            metas[i]["result"] = result

    if not dry_run:
        d = os.path.dirname(base)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        os.close(fd)
        with open(tmp, "wb") as fh:
            np.savez(fh, metas=metas, offsets=offsets)
        os.replace(tmp, base + ".meta.npz")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mpath = os.path.join(args.shard_dir, "manifest.json")
    with open(mpath, encoding="utf-8") as fh:
        manifest = json.load(fh)

    total = {"games": 0, "reason_changed": 0, "trunc_changed": 0, "result_changed": 0}
    before: dict[str, int] = {}
    after: dict[str, int] = {}
    for name in manifest["shards"]:
        st = repair_shard(os.path.join(args.shard_dir, name), args.dry_run)
        for k in total:
            total[k] += st[k]
        for k, v in st["before"].items():
            before[k] = before.get(k, 0) + v
        for k, v in st["after"].items():
            after[k] = after.get(k, 0) + v
        print(f"  {name}: {st['games']} 局，reason 改 {st['reason_changed']}，"
              f"is_truncated 改 {st['trunc_changed']}，result 改 {st['result_changed']}")

    n = max(total["games"], 1)
    print(f"\n合计 {total['games']} 局：termination_reason 修正 {total['reason_changed']}，"
          f"is_truncated 修正 {total['trunc_changed']}，result 修正 {total['result_changed']}")
    print("修复前分布:", {k: f"{v} ({v/n:.1%})" for k, v in sorted(before.items())})
    print("修复后分布:", {k: f"{v} ({v/n:.1%})" for k, v in sorted(after.items())})

    if not args.dry_run:
        gen = manifest.get("gen", {})
        gen["termination_reason_counts"] = {k: after.get(k, 0) for k in TERM_CODES}
        gen["truncated_rate"] = after.get("truncated", 0) / n
        gen["meta_repaired"] = "termination_reason/is_truncated 按 claim_draw 口径重算"
        manifest["gen"] = gen
        with open(mpath, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=1)
        print(f"manifest 已更新：真实封顶率 {gen['truncated_rate']:.1%}")


if __name__ == "__main__":
    main()
