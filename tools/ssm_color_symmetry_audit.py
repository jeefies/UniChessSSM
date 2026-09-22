#!/usr/bin/env python3
"""颜色对称性与执黑棋力审计：排查"SSM 执黑 100% 败率"的根因。

三部分：
1. 镜像对称测试（bug 探测器）：对局 B1 每 ply 取 ``B2 = B1.mirror()``（python-chess
   mirror：纵向翻转 + 交换棋子颜色 + 交换走子方/易位权/过路兵，且保留半回合与全回合
   计数）。若模型与链路颜色对称，则 ``q(B1) + q(B2) ≈ 0``、策略分布互为镜像、
   白走/黑走 ply 的 q 均值互为相反数。系统性偏离 = 颜色偏差。
2. 人类棋谱策略一致率（棋力度量）：在人类验证局上按行棋方颜色统计模型 top-1 一致率
   与人类着法概率，直接比较模型执白与执黑的策略质量。
3. 数据颜色平衡：自对弈/人类分片的白黑胜率分布。

用法（远端 GPU）：::

    python tools/ssm_color_symmetry_audit.py \\
        --ckpt runs/stage_b_training_2500_gen2/best.pt \\
        --openings data/openings_200.txt --games 8 --book-plies 6 \\
        --selfplay-shards runs/stage_b_gen_2500_gen2 \\
        --human-shards data/shards --human-games 40 \\
        --out runs/color_symmetry_audit.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess  # noqa: E402

from stateseq.actions import action_to_move, move_to_action  # noqa: E402
from stateseq.adapter import encode_board, wdl_logits_to_q  # noqa: E402
from stateseq.data.sequences import _board_key  # noqa: E402


def _flip_sq(sq: int) -> int:
    """格纵向镜像（rank 翻转，file 不变）。"""
    return sq ^ 56


def _flip_move(mv: chess.Move) -> chess.Move:
    return chess.Move(_flip_sq(mv.from_square), _flip_sq(mv.to_square), promotion=mv.promotion)


def _flip_action(action: int) -> int:
    return move_to_action(_flip_move(action_to_move(action)))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def load_model(ckpt_path: str, device: str):
    import torch

    from stateseq.model import SeqModel

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    seq = SeqModel(dropout=0.0)
    seq.load_state_dict(state)
    seq.to(device).eval()
    return seq


def _step(seq, board: chess.Board, occ: int, cache, device: str):
    import torch

    feats, tc, elo_std, color = encode_board(board, occ)
    f = torch.from_numpy(np.asarray(feats, dtype=np.float32)).reshape(1, -1).to(device)
    logits, wdl, _mlh, _x, cache_new = seq.step(
        f,
        torch.tensor([int(tc)], dtype=torch.long, device=device),
        torch.tensor([float(elo_std)], dtype=torch.float32, device=device),
        torch.tensor([int(color)], dtype=torch.long, device=device),
        cache,
    )
    return logits[0].cpu().numpy(), wdl[0].cpu().numpy(), cache_new


def _probe(seq, board: chess.Board, occ: dict, cache, device: str):
    """对 board 做一次前向：返回 (q, {action: prob}, 新 cache)，并更新 occurrence。"""
    key = _board_key(board)
    prior = occ.get(key, 0)
    logits, wdl, cache_new = _step(seq, board, prior, cache, device)
    occ[key] = prior + 1
    q = wdl_logits_to_q(wdl)
    legal = list(board.legal_moves)
    acts = [move_to_action(m) for m in legal]
    probs = _softmax(logits[np.asarray(acts, dtype=np.int64)])
    return q, dict(zip(acts, probs.tolist())), cache_new


def run_mirror_game(seq, device: str, book_line: list[str], book_plies: int,
                    max_plies: int, seed: int) -> list[dict]:
    """一盘对局 + 每 ply 的镜像配对记录（B2 = B1.mirror()）。"""
    import torch

    b1 = chess.Board()
    c1 = seq.initial_cache(1, device=device, dtype=torch.float32)
    c2 = seq.initial_cache(1, device=device, dtype=torch.float32)
    occ1: dict = {}
    occ2: dict = {}
    rng = random.Random(seed)
    records: list[dict] = []
    book_iter = iter(book_line[:book_plies])

    def do_ply(choose) -> bool:
        nonlocal c1, c2
        if b1.is_game_over(claim_draw=True):
            return False
        b2 = b1.mirror()
        q1, p1, c1 = _probe(seq, b1, occ1, c1, device)
        q2, p2, c2 = _probe(seq, b2, occ2, c2, device)
        acts1 = np.asarray(list(p1.keys()), dtype=np.int64)
        probs1 = np.asarray(list(p1.values()), dtype=np.float64)
        flip_ids = np.asarray([_flip_action(int(a)) for a in acts1], dtype=np.int64)
        missing = [int(f) for f in flip_ids if int(f) not in p2]
        if missing:
            raise RuntimeError(f"镜像动作不在 B2 合法集: {missing[:5]} @ {b1.fen()}")
        probs2 = np.asarray([p2[int(f)] for f in flip_ids], dtype=np.float64)
        records.append({
            "ply": len(records),
            "color1": 1 if b1.turn == chess.WHITE else 0,
            "q1": float(q1),
            "q2": float(q2),
            "q_asym": float(q1 + q2),
            "policy_l1": float(0.5 * np.abs(probs1 - probs2).sum()),
            "top1_match": bool(_flip_action(int(acts1[int(np.argmax(probs1))]))
                               == int(max(p2, key=p2.get))),
        })
        mv = choose(p1)
        if mv not in b1.legal_moves:
            raise RuntimeError(f"选着非法: {mv.uci()} @ {b1.fen()}")
        b1.push(mv)
        return True

    def book_choose(_p):
        san = next(book_iter, None)
        if san is None:
            return rng.choice(list(b1.legal_moves))
        return b1.parse_san(san)

    def sample_choose(p):
        acts = list(p.keys())
        w = np.asarray(list(p.values()), dtype=np.float64)
        if w.sum() <= 0:
            return rng.choice(list(b1.legal_moves))
        mv = action_to_move(int(rng.choices(acts, weights=w, k=1)[0]))
        if mv not in b1.legal_moves:
            mv = rng.choice(list(b1.legal_moves))
        return mv

    while len(records) < max_plies:
        in_book = len(records) < book_plies
        if not do_ply(book_choose if in_book else sample_choose):
            break
    return records


def human_agreement(seq, device: str, shard_dir: str, n_games: int,
                    seed: int = 20260922) -> dict:
    """人类验证局上按颜色统计模型策略与人类着法的一致率。"""
    import torch

    from stateseq.data.gshards import ShardReader

    reader = ShardReader(shard_dir)
    total = len(reader.meta_all)
    rng = random.Random(seed)
    idxs = rng.sample(range(total), min(n_games, total))
    stats = {"white": [], "black": []}
    with torch.no_grad():
        for gi in idxs:
            meta, actions = reader.game(gi)
            board = chess.Board()
            cache = seq.initial_cache(1, device=device, dtype=torch.float32)
            occ: dict = {}
            for act in actions:
                key = _board_key(board)
                prior = occ.get(key, 0)
                occ[key] = prior + 1
                _q, p, cache = _probe(seq, board, occ, cache, device)
                acts = list(p.keys())
                probs = np.asarray(list(p.values()), dtype=np.float64)
                a = int(act)
                if a in p:
                    color = "white" if board.turn == chess.WHITE else "black"
                    stats[color].append({
                        "top1": bool(acts[int(np.argmax(probs))] == a),
                        "human_prob": float(p[a]),
                        "entropy": float(-(probs * np.log(np.maximum(probs, 1e-12))).sum()),
                    })
                mv = action_to_move(a)
                if mv not in board.legal_moves:
                    break
                board.push(mv)
                if board.is_game_over(claim_draw=False):
                    break
    out = {}
    for color, items in stats.items():
        if not items:
            out[color] = {"plies": 0}
            continue
        out[color] = {
            "plies": len(items),
            "top1_rate": float(np.mean([it["top1"] for it in items])),
            "human_prob_mean": float(np.mean([it["human_prob"] for it in items])),
            "entropy_mean": float(np.mean([it["entropy"] for it in items])),
        }
    if out.get("white", {}).get("plies") and out.get("black", {}).get("plies"):
        out["top1_gap_white_minus_black"] = out["white"]["top1_rate"] - out["black"]["top1_rate"]
        out["human_prob_gap_white_minus_black"] = (
            out["white"]["human_prob_mean"] - out["black"]["human_prob_mean"])
    return out


def _shard_result_stats(shard_dir: str) -> dict:
    """v3/v2 分片的白视角 result 分布（0 白胜/1 和/2 黑胜）。"""
    from stateseq.data.gshards import ShardReader, V3ShardReader

    try:
        reader = V3ShardReader(shard_dir)
        results = np.asarray(reader.meta_all["result"])
        kind = "v3"
    except Exception:
        reader = ShardReader(shard_dir)
        results = (np.concatenate([m["result"] for m in reader.metas])
                   if reader.metas else np.array([]))
        kind = "v2"
    n = len(results)
    if n == 0:
        return {"kind": kind, "games": 0}
    return {
        "kind": kind,
        "games": int(n),
        "white_win": int((results == 0).sum()),
        "draw": int((results == 1).sum()),
        "black_win": int((results == 2).sum()),
        "white_win_rate": float((results == 0).mean()),
        "black_win_rate": float((results == 2).mean()),
        "white_score_rate": float((results == 0).mean() + 0.5 * (results == 1).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="颜色对称性与执黑棋力审计")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--book-plies", type=int, default=6)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--openings", default="data/openings_200.txt")
    ap.add_argument("--selfplay-shards", default="")
    ap.add_argument("--human-shards", default="")
    ap.add_argument("--human-games", type=int, default=40)
    ap.add_argument("--out", default="runs/color_symmetry_audit.json")
    args = ap.parse_args()

    import torch

    seq = load_model(args.ckpt, args.device)

    book_lines: list[list[str]] = []
    if args.openings and os.path.exists(args.openings):
        with open(args.openings, "r", encoding="utf-8") as fh:
            book_lines = [ln.strip().split() for ln in fh if ln.strip()]
        print(f"已加载 {len(book_lines)} 条开局（{args.openings}）")
    else:
        print(f"警告：开局文件不存在（{args.openings}），使用随机开局")

    # ---- 1. 镜像对称测试 ----
    all_records: list[dict] = []
    with torch.no_grad():
        for g in range(args.games):
            line = book_lines[g % len(book_lines)] if book_lines else []
            recs = run_mirror_game(seq, args.device, line, args.book_plies,
                                   args.max_plies, seed=20260922 + g)
            all_records.extend(recs)
            print(f"  [mirror] game {g + 1}/{args.games}: {len(recs)} plies", flush=True)

    q_asym = np.array([r["q_asym"] for r in all_records])
    pol_l1 = np.array([r["policy_l1"] for r in all_records])
    top1 = np.array([r["top1_match"] for r in all_records])
    wq = np.array([r["q1"] for r in all_records if r["color1"] == 1])
    bq = np.array([r["q1"] for r in all_records if r["color1"] == 0])
    wq_m = np.array([r["q2"] for r in all_records if r["color1"] == 1])

    summary = {
        "ckpt": args.ckpt,
        "games": args.games,
        "plies": int(len(all_records)),
        "mirror_symmetry": {
            "q_asym_abs_mean": float(np.abs(q_asym).mean()),
            "q_asym_abs_p95": float(np.percentile(np.abs(q_asym), 95)),
            "q_asym_abs_max": float(np.abs(q_asym).max()),
            "q_asym_signed_mean": float(q_asym.mean()),
            "policy_l1_mean": float(pol_l1.mean()),
            "policy_l1_p95": float(np.percentile(pol_l1, 95)),
            "top1_match_rate": float(top1.mean()),
            "white_to_move_q_mean": float(wq.mean()) if len(wq) else None,
            "black_to_move_q_mean": float(bq.mean()) if len(bq) else None,
            "color_prior_gap": float(wq.mean() + wq_m.mean()) if len(wq) else None,
        },
    }

    # ---- 2. 人类棋谱按颜色策略一致率 ----
    if args.human_shards and os.path.exists(args.human_shards):
        print("[human] 统计人类棋谱策略一致率 ...", flush=True)
        with torch.no_grad():
            summary["human_agreement"] = human_agreement(
                seq, args.device, args.human_shards, args.human_games)

    # ---- 3. 数据颜色平衡 ----
    if args.selfplay_shards:
        summary["selfplay_data"] = _shard_result_stats(args.selfplay_shards)
    if args.human_shards and os.path.exists(args.human_shards):
        summary["human_data"] = _shard_result_stats(args.human_shards)

    ms = summary["mirror_symmetry"]
    verdict = {
        "value_color_bias": bool(ms["q_asym_abs_mean"] > 0.05
                                 or abs(ms["color_prior_gap"] or 0.0) > 0.05),
        "policy_color_bias": bool(ms["policy_l1_mean"] > 0.05 or ms["top1_match_rate"] < 0.90),
    }
    summary["verdict"] = verdict

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)

    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\n结论：价值颜色偏差={'有' if verdict['value_color_bias'] else '无'}，"
          f"策略颜色偏差={'有' if verdict['policy_color_bias'] else '无'}")
    print(f"已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
