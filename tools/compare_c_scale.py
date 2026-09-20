"""c_scale 对照实验：在固定局面上比较 c_scale=1.0 与 0.1（同一 checkpoint）。

设计要点（对应审查意见）：

1. **完整前缀，不只是 FEN。** R 是递归主干，历史不同 ⇒ 同一 FEN 也得到不同 q/logits。
   这里从真实对局分片抽取 (局, ply) 前缀，逐着推进 R，保证两侧看到的历史逐位一致。
2. **重新执行完整搜索，而不是对旧 Q 重新 softmax。** 改变尺度不只改最终的 π′，
   还改非根选择（select_action）与访问分配 ⇒ 必须整棵搜索重跑。
3. **配对随机种子。** 两个尺度用同一个 Gumbel 噪声序列，差异只归因于尺度。
   另跑第二个种子看"跨噪声稳定性"。
4. **记录四组量**：目标熵/最大概率/KL；最优与次优的打分差、原始 Q 差、局部 span；
   选着变化与根访问分配；预算/合法性/终局真值。

用法：
  python tools/compare_c_scale.py --shards runs/stage_b_gen_fix500 --positions 128 \
      --out runs/c_scale_compare
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stateseq.adapter import encode_board, wdl_logits_to_q
from stateseq.actions import move_to_action
from stateseq.data.gshards import V3ShardReader
from stateseq.data.sequences import _board_key
from stateseq.gumbel import (
    C_VISIT, M0, N_SIMS, Node, completed_q, export_pi_prime, qtransform_completed, softmax,
)
from tools.ssm_gumbel_selfplay import (
    GameState, ModelWrapper, SelfPlayConfig, concat_caches, split_cache,
)

SCALES = (1.0, 0.1)
SEEDS = (0, 1)


def _entropy(p) -> float:
    p = np.asarray(p, np.float64)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _kl(p, q) -> float:
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    m = (p > 0) & (q > 0)
    return float((p[m] * np.log(p[m] / q[m])).sum())


def _sample_positions(reader: V3ShardReader, n: int, rng: np.random.Generator):
    """按开局/中局/残局三段抽取前缀，尽量铺满。"""
    metas = reader.meta_all
    n_games = len(metas)
    probes = []
    bands = [("opening", 3, 24), ("midgame", 40, 130), ("late", 150, 10 ** 9)]
    per_band = max(1, n // len(bands))
    for name, lo, hi in bands:
        made, tries = 0, 0
        while made < per_band and tries < per_band * 60:
            tries += 1
            gi = int(rng.integers(0, n_games))
            n_plies = int(metas[gi]["n_plies"])
            if n_plies <= lo + 1:
                continue
            target = int(rng.integers(lo, max(lo + 2, min(hi, n_plies))))
            if not (0 < target < n_plies):
                continue
            g = reader.game(gi)
            prefix = [int(a) for a in g["actions"][:target]]
            probes.append({"game_idx": gi, "prefix": prefix, "target_ply": target,
                           "phase": name, "board": chess.Board(), "cache": None,
                           "occurrence": {}, "played": 0})
            made += 1
    return probes


def _advance_all(model, probes) -> None:
    """Phase A：逐着推进 R（每 ply 一次前向，跨位置拼批）。"""
    for p in probes:
        p["cache"] = model.initial_cache(1)
    pending = list(range(len(probes)))
    step = 0
    while pending:
        feats, tcs, elos, colors, caches = [], [], [], [], []
        for i in pending:
            pr = probes[i]
            key = _board_key(pr["board"])
            f, tc, elo, color = encode_board(pr["board"], pr["occurrence"].get(key, 0))
            feats.append(f); tcs.append(int(tc)); elos.append(float(elo)); colors.append(int(color))
            caches.append(pr["cache"])
        logits, wdl, mlh, x, cache_new = model.step_batch(
            np.stack(feats).astype(np.float32), tcs, elos, colors, concat_caches(caches))
        per_item = split_cache(cache_new, len(pending))
        still = []
        for i, pi in enumerate(pending):
            pr = probes[pi]
            pr["cache"] = per_item[i]
            key = _board_key(pr["board"])
            pr["occurrence"][key] = pr["occurrence"].get(key, 0) + 1
            want = pr["prefix"][pr["played"]]
            mv = next((m for m in pr["board"].legal_moves if move_to_action(m) == want), None)
            if mv is None:
                raise RuntimeError(f"位置 {pi} 前缀动作 {want} 非法 @ {pr['board'].fen()}")
            pr["board"].push(mv)
            pr["played"] += 1
            if pr["played"] < pr["target_ply"]:
                still.append(pi)
        pending = still
        step += 1
        if step % 25 == 0:
            print(f"  [Phase A] 推进 {step} ply，剩余 {len(pending)} 个位置", flush=True)
    print(f"  [Phase A] 完成：{len(probes)} 个位置全部就位", flush=True)


def _build_root(model, probe):
    """构造根节点（与生成器 _play_ply 同口径：encode-before-increment）。"""
    key = _board_key(probe["board"])
    feats, tc_val, elo_std, color = encode_board(probe["board"], probe["occurrence"].get(key, 0))
    logits, wdl, mlh, x, cache_new = model.step_batch(
        np.asarray(feats, np.float32).reshape(1, -1), [int(tc_val)],
        [float(elo_std)], [int(color)], probe["cache"])
    q = wdl_logits_to_q(wdl[0])
    legal = [a for m in probe["board"].legal_moves if (a := move_to_action(m)) is not None]
    legal_arr = np.array(legal, dtype=np.int64)
    logits_full = np.full(1936, -3e4, dtype=np.float32)
    logits_full[legal_arr] = logits[0][legal_arr]
    node = Node(legal=legal_arr, logits=logits_full[legal_arr], q=float(q), depth=0, path=())
    return node, cache_new


def _run_search(model, probes, scale: float, seed: int, cfg_n_sims: int, cfg_m0: int,
                base_seed: int) -> list[dict]:
    """对全部位置跑一遍完整 Gumbel 搜索，返回每个位置的度量。"""
    cfg = SelfPlayConfig(ckpt="probe", out_dir="/tmp/probe", tag="probe",
                         c_visit=C_VISIT, c_scale=scale)
    cfg.n_sims = cfg_n_sims
    cfg.m0 = cfg_m0
    cfg.gumbel_g = 1.0

    states = []
    for pr in probes:
        node, cache_new = _build_root(model, pr)
        rng_pos = np.random.default_rng([base_seed, pr["game_idx"], seed])
        host = GameState(0, model, cfg, np.random.SeedSequence(0))
        host.board = pr["board"].copy()
        host.root_cache = cache_new
        host.occurrence = dict(pr["occurrence"])
        host.rng = rng_pos
        st = {"probe": pr, "node": node, "host": host, "done": False, "result": None}
        st["gen"] = host._order_halving_gen(node)
        st["req"] = st["gen"].send(None)
        states.append(st)

    while any(not st["done"] for st in states):
        active = [st for st in states if not st["done"]]
        logits, wdl, mlh, x, cache_new = model.step_batch(
            np.stack([st["req"][0] for st in active]).astype(np.float32),
            [st["req"][1] for st in active], [st["req"][2] for st in active],
            [st["req"][3] for st in active],
            concat_caches([st["req"][4] for st in active]))
        per_item = split_cache(cache_new, len(active))
        for st, i in zip(active, range(len(active))):
            resp = (logits[i], wdl[i], mlh[i], x[i], per_item[i])
            try:
                st["req"] = st["gen"].send(resp)
            except StopIteration as e:
                st["result"] = e.value
                st["done"] = True

    out = []
    for st in states:
        node, res = st["node"], st["result"]
        pr = st["probe"]
        pi_pol = softmax(node.logits)
        ids, pi_target = export_pi_prime(node, C_VISIT, scale)
        cq = completed_q(node)
        span = float(cq.max() - cq.min()) if cq.size else 0.0
        sig = qtransform_completed(node, C_VISIT, scale)
        score = node.logits + sig
        order = np.argsort(-score)
        # 终局前可能只剩 1 个合法着；此时"最优/次优差"无定义，记 None 而不是造假值
        if len(order) >= 2:
            best_i, second_i = int(order[0]), int(order[1])
            score_gap = float(score[best_i] - score[second_i])
            raw_q_gap = float(cq[best_i] - cq[second_i])
            logit_gap = float(node.logits[best_i] - node.logits[second_i])
        else:
            best_i = int(order[0])
            score_gap = raw_q_gap = logit_gap = None
        visits = node.n.astype(np.float64) if node.n.size else np.zeros(0)
        rec = {
            "game_idx": pr["game_idx"], "phase": pr["phase"], "ply": pr["target_ply"],
            "scale": scale, "seed": seed,
            "n_legal": int(len(node.legal)),
            "target_entropy": _entropy(pi_target),
            "target_max_prob": float(pi_target.max()),
            "target_entropy_frac": _entropy(pi_target) / max(np.log(len(node.legal)), 1e-8),
            "kl_target_vs_policy": _kl(pi_target, pi_pol),
            "policy_entropy": _entropy(pi_pol),
            "score_gap": score_gap,
            "raw_q_gap": raw_q_gap,
            "logit_gap": logit_gap,
            "completed_q_span": span,
            "chosen_action": int(node.legal[best_i]),
            "chosen_is_best_score": True,
            "root_visits_top1_share": float(visits.max() / visits.sum()) if visits.sum() else 0.0,
            "root_n_max": int(visits.max()) if visits.size else 0,
            "root_visits_candidates": int((visits > 0).sum()),
            "sims_used": int(res["sims_used"]),
            "budget_ok": bool(res["sims_used"] == cfg_n_sims),
            "n_nodes": int(res["n_nodes"]),
            "n_terminal": int(res["n_terminal"]),
            "max_depth": int(res["max_depth"]),
            "q_min": float(cq.min()) if cq.size else None,
            "q_max": float(cq.max()) if cq.size else None,
        }
        out.append(rec)
    return out


def _report(records, args) -> None:
    import collections
    by_key = collections.defaultdict(dict)
    for r in records:
        by_key[(r["game_idx"], r["ply"])][(r["scale"], r["seed"])] = r

    def agg(rs, key):
        v = [r[key] for r in rs if r[key] is not None]
        if not v:
            return None
        return {"mean": float(np.mean(v)), "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)), "min": float(np.min(v)),
                "max": float(np.max(v))}

    summary = {}
    for scale in SCALES:
        rs = [r for r in records if r["scale"] == scale]
        summary[f"c_scale={scale}"] = {
            "n_positions": len(rs),
            "target_entropy": agg(rs, "target_entropy"),
            "target_entropy_frac": agg(rs, "target_entropy"),
            "target_max_prob": agg(rs, "target_max_prob"),
            "kl_target_vs_policy": agg(rs, "kl_target_vs_policy"),
            "policy_entropy": agg(rs, "policy_entropy"),
            "score_gap": agg(rs, "score_gap"),
            "raw_q_gap": agg(rs, "raw_q_gap"),
            "logit_gap": agg(rs, "logit_gap"),
            "completed_q_span": agg(rs, "completed_q_span"),
            "root_visits_top1_share": agg(rs, "root_visits_top1_share"),
            "sims_used": agg(rs, "sims_used"),
            "max_depth": agg(rs, "max_depth"),
            "budget_ok_rate": float(np.mean([r["budget_ok"] for r in rs])),
            "target_entropy_frac_mean": float(np.mean([r["target_entropy_frac"] for r in rs])),
            "frac_entropy_lt_0p01": float(np.mean([r["target_entropy"] < 0.01 for r in rs])),
            "frac_maxprob_gt_0p999": float(np.mean([r["target_max_prob"] > 0.999 for r in rs])),
        }

    # 选着一致性与跨噪声稳定性
    agree = diff_scale = 0
    seed_stable = tot_seed = 0
    for key, bys in by_key.items():
        s1 = bys.get((1.0, 0))
        s01 = bys.get((0.1, 0))
        if s1 and s01:
            diff_scale += 1
            agree += int(s1["chosen_action"] == s01["chosen_action"])
        a, b = bys.get((1.0, 0)), bys.get((1.0, 1))
        if a and b:
            tot_seed += 1
            seed_stable += int(a["chosen_action"] == b["chosen_action"])
    cross = {
        "positions_compared": diff_scale,
        "chosen_action_agree_1p0_vs_0p1": agree,
        "chosen_action_agree_rate": agree / max(diff_scale, 1),
        "seed_stability_positions": tot_seed,
        "seed_stability_agree": seed_stable,
        "seed_stability_rate": seed_stable / max(tot_seed, 1),
    }
    summary["cross_scale"] = cross

    with open(os.path.join(args.out, "c_scale_compare.json"), "w", encoding="utf-8") as fh:
        json.dump({"config": {"positions": len(by_key), "n_sims": args.n_sims,
                              "m0": args.m0, "scales": list(SCALES), "seeds": list(SEEDS),
                              "ckpt": args.ckpt, "shards": args.shards},
                   "summary": summary, "records": records}, fh, ensure_ascii=False, indent=1)

    print("\n===== c_scale 对照汇总 =====")
    for k in [f"c_scale={s}" for s in SCALES]:
        s = summary[k]
        print(f"\n[{k}] {s['n_positions']} 个位置")
        e, mp = s["target_entropy"], s["target_max_prob"]
        print(f"  目标熵    mean={e['mean']:.4f} median={e['median']:.4f} p90={e['p90']:.4f} "
              f"（熵<0.01 占 {s['frac_entropy_lt_0p01']:.1%}，max_prob>0.999 占 {s['frac_maxprob_gt_0p999']:.1%}）")
        print(f"  最大概率  mean={mp['mean']:.4f} median={mp['median']:.4f}")
        kl, pe = s["kl_target_vs_policy"], s["policy_entropy"]
        print(f"  KL(π′‖π) mean={kl['mean']:.4f} median={kl['median']:.4f}；原 policy 熵 mean={pe['mean']:.4f}")
        sg, rg = s["score_gap"], s["raw_q_gap"]
        print(f"  打分差    mean={sg['mean']:.3f} median={sg['median']:.3f}；原始 Q 差 mean={rg['mean']:.4f}")
        sp = s["completed_q_span"]
        print(f"  局部 span mean={sp['mean']:.4f} median={sp['median']:.4f}")
        rv, d = s["root_visits_top1_share"], s["max_depth"]
        print(f"  根访问 top1 占比 mean={rv['mean']:.3f}；平均树深 {d['mean']:.2f}；预算合规 {s['budget_ok_rate']:.1%}")
    print(f"\n[跨尺度] 选着一致率 {cross['chosen_action_agree_1p0_vs_0p1']}/"
          f"{cross['positions_compared']} = {cross['chosen_action_agree_rate']:.1%}")
    print(f"[跨噪声] c_scale=1.0 下两个种子的选着一致率 "
          f"{cross['seed_stability_agree']}/{cross['seed_stability_positions']} = "
          f"{cross['seed_stability_rate']:.1%}")
    print(f"\n已写 {os.path.join(args.out, 'c_scale_compare.json')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="runs/stage_b_gen_fix500")
    ap.add_argument("--ckpt", default="runs/stage_a_20260915/best.pt")
    ap.add_argument("--positions", type=int, default=128)
    ap.add_argument("--n_sims", type=int, default=N_SIMS)
    ap.add_argument("--m0", type=int, default=M0)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--out", default="runs/c_scale_compare")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)

    model = ModelWrapper(args.ckpt, device)
    reader = V3ShardReader(args.shards)
    rng = np.random.default_rng(args.seed)
    probes = _sample_positions(reader, args.positions, rng)
    counts = {b: sum(1 for p in probes if p["phase"] == b) for b in ("opening", "midgame", "late")}
    print(f"取样 {len(probes)} 个位置：{counts}")

    _advance_all(model, probes)

    records = []
    for scale in SCALES:
        for seed in SEEDS:
            print(f"  [Phase B] c_scale={scale} seed={seed}", flush=True)
            records.extend(_run_search(model, probes, scale, seed,
                                       args.n_sims, args.m0, args.seed))
    _report(records, args)


if __name__ == "__main__":
    main()
