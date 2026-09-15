"""Stage A 离线诊断：value / dyn / policy / 动作对应性（tools/stateseq_diag.py）。

只读 checkpoint 与数据，短前向评估；可与 Stage A 训练并发（eval + no_grad + bf16 autocast +
小 microbatch + 少量 workers）。诊断口径尽量复刻训练时验证（stateseq.losses / train/stage_a.py
的 run_val）：val 子集 = SequenceDataset.val_batch 同口径（seed 777、8×32 局）。

产物（--out-dir，默认 checkpoint 所在 run 目录）：
  diag_<step>.json   全量指标
  diag_<step>.md     人类可读报告（同时打印到 stdout）

用法（远端）：
  python tools/stateseq_diag.py --train-games 1024 --val-batches 8 --also-best
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from stateseq import losses  # noqa: E402
from stateseq.actions import move_to_action  # noqa: E402
from stateseq.data.dataset import SequenceDataset, _worker_build  # noqa: E402
from stateseq.heads import apply_legal_mask  # noqa: E402
from stateseq.model import SeqModel, TrainBatch  # noqa: E402

VAL_SEED = 777        # 与 SequenceDataset.val_batch 默认一致（训练时验证口径）
TRAIN_FREQ_SEED = 20260916  # 固定训练子集抽样种子（仅用于全局 WDL 频率 → 先验 q）
D_SEED = 4242         # 块 D 抽样种子
D_MODEL_DIM = 512
RESULT_NAMES = ("胜(W)", "和(D)", "负(L)")
T_BUCKETS = ((0, 20, "0-20"), (21, 60, "21-60"), (61, None, ">60"))
D_BUCKETS = ((1, 10, "1-10"), (11, 30, "11-30"), (31, None, ">30"))


# ---------------------------------------------------------------- 数据装载

def to_device(batch: TrainBatch, device: str) -> TrainBatch:
    return TrainBatch(*[getattr(batch, f).to(device) for f in batch.__dataclass_fields__])


def make_batches(ds: SequenceDataset, idxs: np.ndarray, microbatch: int):
    """复用库内多进程重放与 _collate（只读使用，不改库代码）。

    返回 [(items, meta 数组, TrainBatch(cpu), valid(cpu))]；items 含 (global_idx, data, w, elo_std, tc)。
    """
    out = []
    for j in range(0, len(idxs), microbatch):
        items = ds.pool.map(_worker_build, [int(i) for i in idxs[j:j + microbatch]])
        # 必须与 SequenceDataset._collate 同序（按局长度降序），否则 meta/行错位
        items = sorted(items, key=lambda it: -len(it[1]["actions"]))
        coll = ds._collate(items)
        metas = ds.reader.meta_all[[int(it[0]) for it in items]]
        out.append((items, metas, coll["batch"], coll["valid"]))
    return out


def val_subset_indices(ds: SequenceDataset, n_batches: int,
                       sel_microbatch: int = 32) -> np.ndarray:
    """与 SequenceDataset.val_batch(seed=777, microbatch=sel_microbatch) 同口径的局索引。"""
    rng = np.random.default_rng(VAL_SEED)
    return rng.choice(ds.val_indices,
                      size=min(n_batches * sel_microbatch, len(ds.val_indices)),
                      replace=False)


def train_subset_indices(ds: SequenceDataset, n_games: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(ds.train_indices, size=min(n_games, len(ds.train_indices)), replace=False)


def train_wdl_freq_from_meta(ds: SequenceDataset, idxs: np.ndarray) -> np.ndarray:
    """训练子集全局 WDL 频率（按局面计、行棋方归一），由分片 meta 精确推算，无需重放。

    meta.result 为白视角；t 偶数白走（标签=result）、t 奇数黑走（标签=2-result）。
    """
    meta = ds.reader.meta_all[idxs]
    counts = np.zeros(3, dtype=np.float64)
    for res, n in zip(meta["result"].astype(np.int64), meta["n_plies"].astype(np.int64)):
        n_white = (n + 1) // 2
        n_black = n // 2
        counts[int(res)] += n_white
        counts[2 - int(res)] += n_black
    return counts / counts.sum()


# ---------------------------------------------------------------- 前向评估

class ValAccumulator:
    """在 val 子集上前向，累计块 A/B/C 所需的全部统计量。"""

    def __init__(self) -> None:
        self.batch_core: dict[str, list[float]] = {}
        # 逐局面数组（仅 valid 位）
        self.p = []        # (N,3) softmax 后 WDL 概率
        self.t = []        # (N,) ply 索引
        self.d = []        # (N,) 距终局 ply = n_plies − t（分片 meta 未截断 n_plies）
        self.res = []      # (N,) 真实结果标签
        self.elo = []      # (N,) elo_weight
        self.ce = []       # (N,) 模型 value CE
        self.top1 = []     # (N,) bool（合法着掩码后 argmax == 实走）
        self.top3 = []
        self.single = []   # (N,) bool 只有一个合法着
        self.pol_ce_w = []  # (N,) 带 elo 权重的 policy CE
        self.pol_ce_uw = []
        self.x_rows = []   # (N,512) E 输出（latent 健康）
        # dyn 全局量（跨 batch 求和）
        self.dyn_eng_sum = 0.0
        self.dyn_cnt = 0
        self.n_pos = 0

    def add_batch(self, model: SeqModel, batch: TrainBatch, valid: torch.Tensor,
                  device: str, metas: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        b = to_device(batch, device)
        v = valid.to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = model.encode(b.features)                       # (B,T,512)
            cond = model._cond_expand(b, None)
            h = model.trunk(x + cond)
            pol, wdl, _mlh = model.f(h)
            d_out = model.d(x)
            delta_hat = model.g(h[:, :-1], b.actions[:, 1:])   # (B,T-1,512)

        core: dict[str, float] = {}
        # ---- value（与 losses.value_loss 同口径：masked mean CE）----
        ce = F.cross_entropy(wdl.reshape(-1, 3).float(), b.results.reshape(-1).long(),
                             reduction="none").reshape(wdl.shape[:2])
        vmask = v.bool()
        core["val_value_ce"] = float((ce * vmask).sum() / vmask.sum().clamp(min=1))

        # ---- policy CE（加权 = 训练口径；另记未加权）----
        ce_p = F.cross_entropy(pol.reshape(-1, pol.shape[-1]).float(),
                               b.actions.reshape(-1).long(),
                               reduction="none").reshape(pol.shape[:2])
        w_elo = b.elo_weight.unsqueeze(1).expand_as(ce_p)
        core["val_policy_ce_w"] = float((ce_p * w_elo * vmask).sum()
                                        / (w_elo * vmask).sum().clamp(min=1e-8))
        core["val_policy_ce_uw"] = float((ce_p * vmask).sum() / vmask.sum().clamp(min=1))

        # ---- dyn（pos_mask = valid[:,1:]，与 forward_train 一致）----
        _l_dyn, diag_dyn = losses.dyn_loss(delta_hat, x[:, 1:], x[:, :-1], vmask[:, 1:])
        core["val_dyn_mse"] = float(_l_dyn)
        core["val_dyn_rel_err"] = diag_dyn["dyn_rel_err"]
        delta = (x[:, 1:] - x[:, :-1]).float()
        vm1 = vmask[:, 1:]
        per_t = delta.pow(2).sum(-1) / delta.shape[-1]
        self.dyn_eng_sum += float((per_t * vm1).sum())
        self.dyn_cnt += int(vm1.sum())

        # ---- recon whole-board acc（训练口径）----
        _l_rec, diag_rec = losses.recon_loss(d_out, b.features, vmask)
        core["val_recon_whole_board_acc"] = diag_rec["recon_whole_board_acc"]
        for k, val in core.items():
            self.batch_core.setdefault(k, []).append(val)

        # ---- 逐局面落盘（CPU numpy）----
        probs = torch.softmax(wdl.float(), dim=-1)              # (B,T,3)
        masked = apply_legal_mask(pol.float(), b.legal_mask)    # 非法着 -inf
        top1_hit = (masked.argmax(-1) == b.actions)
        top3_hit = (masked.topk(3, dim=-1).indices == b.actions.unsqueeze(-1)).any(-1)
        n_legal = b.legal_mask.sum(-1)

        # 每桶局面数等：t / d 矩阵（n_plies 取分片 meta 未截断总局数）
        bsz, seqlen = vmask.shape
        t_mat = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, seqlen)
        n_plies = [int(m["n_plies"]) for m in metas]
        d_mat = torch.tensor(n_plies, device=device).unsqueeze(1) - t_mat

        idx = vmask.cpu().numpy()
        self.p.append(probs[vmask].cpu().numpy().astype(np.float32))
        self.t.append(t_mat[vmask].cpu().numpy().astype(np.int32))
        self.d.append(d_mat[vmask].cpu().numpy().astype(np.int32))
        self.res.append(b.results[vmask].cpu().numpy().astype(np.int64))
        self.elo.append(w_elo[vmask].cpu().numpy().astype(np.float32))
        self.ce.append(ce[vmask].cpu().numpy().astype(np.float32))
        self.top1.append(top1_hit[vmask].cpu().numpy())
        self.top3.append(top3_hit[vmask].cpu().numpy())
        self.single.append((n_legal[vmask] == 1).cpu().numpy())
        self.pol_ce_w.append((ce_p * w_elo)[vmask].cpu().numpy().astype(np.float32))
        self.pol_ce_uw.append(ce_p[vmask].cpu().numpy().astype(np.float32))
        self.x_rows.append(x[vmask].float().cpu().numpy().astype(np.float32))
        self.n_pos += int(idx.sum())
        return x.float().cpu(), h.float().cpu()  # 供块 D 取 h_{t-1}/x_{t-1}


def forward_val_subset(model, batches, device):
    """返回 (acc, stored_x, stored_h)：stored_* 供块 D 取 h_{t-1}/x_{t-1}。"""
    acc = ValAccumulator()
    stored_x, stored_h = [], []
    for _items, metas, batch, valid in batches:
        x, h = acc.add_batch(model, batch, valid, device, metas)
        stored_x.append(x)
        stored_h.append(h)
    return acc, stored_x, stored_h


# ---------------------------------------------------------------- 块 A：value 分析

def _bucket_mask(arr: np.ndarray, lo: int, hi: int | None) -> np.ndarray:
    m = arr >= lo
    if hi is not None:
        m &= arr <= hi
    return m


def _prior_ce_from_freq(freq: np.ndarray) -> float:
    """用标签分布 q 作常数预测时的 CE = −Σ_k q_k·log q_k（q_k=0 项跳过）。"""
    q = np.clip(freq, 1e-12, 1.0)
    return float(-(freq * np.log(q)).sum())


def analyze_value(acc: ValAccumulator, q_train: np.ndarray) -> dict:
    p = np.concatenate(acc.p)                    # (N,3)
    t = np.concatenate(acc.t)
    d = np.concatenate(acc.d)
    res = np.concatenate(acc.res)
    ce = np.concatenate(acc.ce)
    n = len(res)

    freq_val = np.bincount(res, minlength=3) / n
    prior_ce_val = float(-np.log(np.clip(q_train, 1e-12, 1.0))[res].mean())
    model_ce_val = float(ce.mean())

    # 预测分布 mean/std/P5/P50/P95
    pred_dist = {}
    for k, name in enumerate(("pW", "pD", "pL")):
        col = p[:, k]
        pred_dist[name] = {
            "mean": float(col.mean()), "std": float(col.std()),
            "p5": float(np.percentile(col, 5)), "p50": float(np.percentile(col, 50)),
            "p95": float(np.percentile(col, 95)),
        }

    # 3×3 表：按真实结果分组的平均预测
    table_3x3 = {}
    for r, name in enumerate(RESULT_NAMES):
        m = res == r
        table_3x3[name] = {"n": int(m.sum()),
                           "pW": float(p[m, 0].mean()), "pD": float(p[m, 1].mean()),
                           "pL": float(p[m, 2].mean())}

    # 分桶 CE（按 t / 按 d），桶内先验 CE 用桶内频率
    def bucket_rows(key: np.ndarray, buckets) -> list[dict]:
        rows = []
        for lo, hi, label in buckets:
            m = _bucket_mask(key, lo, hi)
            if m.sum() == 0:
                rows.append({"bucket": label, "n": 0})
                continue
            fq = np.bincount(res[m], minlength=3) / m.sum()
            rows.append({
                "bucket": label, "n": int(m.sum()),
                "freqW": float(fq[0]), "freqD": float(fq[1]), "freqL": float(fq[2]),
                "model_ce": float(ce[m].mean()), "prior_ce": _prior_ce_from_freq(fq),
            })
        return rows

    # 校准：按 argmax 置信度 10 bin
    conf = p.max(-1)
    pred_cls = p.argmax(-1)
    hit = pred_cls == res
    cal_rows = []
    edges = np.linspace(0.0, 1.0, 11)
    brier = float(((p - np.eye(3)[res]) ** 2).sum(-1).mean())
    for i in range(10):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < 9 else conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        cal_rows.append({
            "bin": f"[{edges[i]:.1f},{edges[i+1]:.1f}]",
            "n": int(m.sum()), "mean_conf": float(conf[m].mean()),
            "acc": float(hit[m].mean()),
        })

    return {
        "n_positions": n,
        "val_wdl_freq": {"W": float(freq_val[0]), "D": float(freq_val[1]), "L": float(freq_val[2])},
        "train_wdl_freq": {"W": float(q_train[0]), "D": float(q_train[1]), "L": float(q_train[2])},
        "prior_ce_on_val": prior_ce_val,
        "model_ce_on_val": model_ce_val,
        "value_gain": prior_ce_val - model_ce_val,
        "pred_distribution": pred_dist,
        "table_3x3": table_3x3,
        "buckets_by_ply": bucket_rows(t, T_BUCKETS),
        "buckets_by_dist_to_end": bucket_rows(d, D_BUCKETS),
        "truncation_note": ("d 由分片 meta 的未截断 n_plies 计算；>200 ply 的局重放截断到 T_MAX=200，"
                            "其局面只覆盖 t<200（t≤200 时 d=n_plies−t 仍精确）。"),
        "calibration": {"rows": cal_rows, "brier": brier},
    }


# ---------------------------------------------------------------- 块 B：dyn / latent

def analyze_dyn_latent(acc: ValAccumulator) -> dict:
    core = {k: float(np.mean(v)) for k, v in acc.batch_core.items()}
    dyn_mse = core["val_dyn_mse"]
    delta_energy = acc.dyn_eng_sum / max(acc.dyn_cnt, 1)
    x = np.concatenate(acc.x_rows)               # (N,512)
    # latent 健康：跨局面方差（先对每维求跨局面方差，再报 mean/std）
    var_per_dim = x.var(axis=0)                  # (512,)
    norm = np.linalg.norm(x, axis=-1)            # (N,)
    return {
        "dyn_mse_masked_mean": dyn_mse,
        "delta_energy_E_norm2_over_d": float(delta_energy),
        "dyn_rel_err_global": float(dyn_mse / max(delta_energy, 1e-12)),
        "dyn_rel_err_batch_mean": core["val_dyn_rel_err"],
        "latent_cross_position_var_per_dim": {
            "mean": float(var_per_dim.mean()), "std": float(var_per_dim.std()),
        },
        "latent_norm2": {"mean": float(norm.mean()), "std": float(norm.std())},
    }


# ---------------------------------------------------------------- 块 C：policy

def analyze_policy(acc: ValAccumulator) -> dict:
    t = np.concatenate(acc.t)
    elo = np.concatenate(acc.elo)
    top1 = np.concatenate(acc.top1)
    top3 = np.concatenate(acc.top3)
    single = np.concatenate(acc.single)

    def rate(mask: np.ndarray) -> dict:
        return {"n": int(mask.sum()), "top1": float(top1[mask].mean()),
                "top3": float(top3[mask].mean())}

    by_ply = {label: rate(_bucket_mask(t, lo, hi)) for lo, hi, label in T_BUCKETS}

    # Elo 权重低/中/高三档（按局面级 elo_weight 三分位）
    q1, q2 = np.quantile(elo, [1 / 3, 2 / 3])
    by_elo = {
        f"低(w<{q1:.2f})": rate(elo < q1),
        f"中({q1:.2f}≤w<{q2:.2f})": rate((elo >= q1) & (elo < q2)),
        f"高(w≥{q2:.2f})": rate(elo >= q2),
    }
    overall = rate(np.ones_like(top1, dtype=bool))
    overall["single_legal_frac"] = float(single.mean())
    nonsingle = ~single
    overall["top1_excl_single_legal"] = float(top1[nonsingle].mean())
    overall["n_excl_single_legal"] = int(nonsingle.sum())
    return {"overall": overall, "by_ply": by_ply, "by_elo_weight": by_elo,
            "elo_terciles": [float(q1), float(q2)]}


# ---------------------------------------------------------------- 块 D：动作对应性

def find_legal_move(board, action: int):
    for mv in board.legal_moves:
        if move_to_action(mv) == int(action):
            return mv
    return None


def action_correspondence(model, ds, batches, stored_x, stored_h, device: str,
                          n_samples: int = 100, seed: int = D_SEED) -> dict:
    """g 头动作对应性：同一 h_{t-1} 下，正确配对 (Δ̂_a,Δ_a)/(Δ̂_b,Δ_b) vs 交换配对的误差。

    局面由 python-chess 重放得到（Python 侧特征提取 = stateseq.features.encode，非 C++ 构建器）。
    """
    import chess
    from stateseq.data.sequences import _board_key
    from stateseq.features import encode

    rng = np.random.default_rng(seed)
    # 候选 (batch_i, row, 有效长度)
    cands = []
    for bi, (items, _metas, batch, valid) in enumerate(batches):
        lens = valid.sum(1).cpu().numpy()
        for row in range(len(items)):
            if lens[row] >= 3:
                cands.append((bi, row, int(lens[row])))
    if not cands:
        return {"skipped": "val 子集中无可采样局面"}

    feats_a: list = []
    feats_b: list = []
    x_prev_rows: list = []
    h_prev_rows: list = []
    act_a: list = []
    act_b: list = []
    made = 0
    attempts = 0
    while made < n_samples and attempts < n_samples * 20:
        attempts += 1
        bi, row, ln = cands[int(rng.integers(len(cands)))]
        t = int(rng.integers(1, ln))
        gidx = int(batches[bi][0][row][0])
        _meta, actions = ds.reader.game(gidx)
        board = chess.Board()
        occ: dict = {}
        ok = True
        for i in range(t):
            key = _board_key(board)
            occ[key] = occ.get(key, 0) + 1
            mv = find_legal_move(board, int(actions[i]))
            if mv is None:
                ok = False
                break
            board.push(mv)
        if not ok:
            continue
        occ[_board_key(board)] = occ.get(_board_key(board), 0) + 1
        a = int(actions[t])
        legal = list(board.legal_moves)
        others = [m for m in legal if move_to_action(m) != a]
        if not others:
            continue
        mv_a = find_legal_move(board, a)
        if mv_a is None:
            continue
        mv_b = others[int(rng.integers(len(others)))]
        for mv, sink in ((mv_a, feats_a), (mv_b, feats_b)):
            bb = board.copy()
            bb.push(mv)
            sink.append(encode(bb, occurrence=occ.get(_board_key(bb), 0)))
        h_prev_rows.append(stored_h[bi][row, t - 1].numpy())
        x_prev_rows.append(stored_x[bi][row, t - 1].numpy())
        act_a.append(a)
        act_b.append(move_to_action(mv_b))
        made += 1
    if made < 10:
        return {"skipped": f"可用样本过少（{made}）"}

    H = torch.tensor(np.stack(h_prev_rows), device=device)            # (N,512)
    XP = torch.tensor(np.stack(x_prev_rows), device=device)           # (N,512)
    Xa = torch.tensor(np.stack(feats_a), device=device)
    Xb = torch.tensor(np.stack(feats_b), device=device)
    Aa = torch.tensor(np.array(act_a), dtype=torch.long, device=device)
    Ab = torch.tensor(np.array(act_b), dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        xa = model.encode(Xa).float()
        xb = model.encode(Xb).float()
        dha = model.g(H, Aa).float()
        dhb = model.g(H, Ab).float()
    delta_a = xa - XP
    delta_b = xb - XP

    def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x - y).pow(2).sum(-1) / x.shape[-1]                   # (N,)

    err_correct = 0.5 * (mse(dha, delta_a) + mse(dhb, delta_b))
    err_swapped = 0.5 * (mse(dha, delta_b) + mse(dhb, delta_a))
    return {
        "n_pairs": made,
        "err_correct_mean": float(err_correct.mean()),
        "err_swapped_mean": float(err_swapped.mean()),
        "correct_less_than_swapped_frac": float((err_correct < err_swapped).float().mean()),
    }


# ---------------------------------------------------------------- 报告

def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return str(obj)
    return obj


def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def build_markdown(report: dict) -> str:
    a = report["value"]
    b = report["dyn_latent"]
    c = report["policy"]
    d = report.get("action_correspondence", {})
    best = report.get("best")
    L = []
    w = L.append
    w(f"# Stage A 离线诊断 @ step {report['step']}（{report['ckpt']}）")
    w("")
    w(f"- 生成时间：{report['generated_at']}；val 子集 {a['n_positions']} 局面"
      f"（seed 777、{report['val_games']} 局）；数据 `{report['data']}`")
    w("- 口径：与训练时验证一致（masked mean CE / bf16 autocast / eval+no_grad）")
    w("")
    w("## 一致性检查（对照训练时 val 指标）")
    w("")
    w("| 指标 | 诊断值 | 参考（训练 val） |")
    w("|---|---|---|")
    s = report["sanity"]
    w(f"| value CE | {fmt(s['val_value_ce'])} | ~0.85 |")
    w(f"| dyn_rel_err | {fmt(s['val_dyn_rel_err'])} | ~0.49 |")
    w(f"| recon whole-board acc | {fmt(s['val_recon_whole_board_acc'], 3)} | ~0.92 |")
    w(f"| policy CE（Elo 加权） | {fmt(s['val_policy_ce_w'])} | ~1.96 |")
    w("")
    w("## A. Value")
    w("")
    fq, ft = a["val_wdl_freq"], a["train_wdl_freq"]
    w(f"- WDL 频率：val = W {fq['W']:.3f} / D {fq['D']:.3f} / L {fq['L']:.3f}；"
      f"train 子集 = W {ft['W']:.3f} / D {ft['D']:.3f} / L {ft['L']:.3f}")
    w(f"- 先验 CE（train 频率→val）= {fmt(a['prior_ce_on_val'])}；"
      f"模型 CE = {fmt(a['model_ce_on_val'])}；**value_gain = {fmt(a['value_gain'])}**")
    w("")
    w("### 预测分布（val 全部有效位）")
    w("")
    w("| 分量 | mean | std | P5 | P50 | P95 |")
    w("|---|---|---|---|---|---|")
    for k in ("pW", "pD", "pL"):
        r = a["pred_distribution"][k]
        w(f"| {k} | {fmt(r['mean'])} | {fmt(r['std'])} | {fmt(r['p5'])} | {fmt(r['p50'])} | {fmt(r['p95'])} |")
    w("")
    w("### 3×3 表（按真实结果的平均预测）")
    w("")
    w("| 真实 | n | pW | pD | pL |")
    w("|---|---|---|---|---|")
    for name in RESULT_NAMES:
        r = a["table_3x3"][name]
        w(f"| {name} | {r['n']} | {fmt(r['pW'])} | {fmt(r['pD'])} | {fmt(r['pL'])} |")
    w("")
    w("### 分桶 CE（val）")
    w("")
    w("按已走 ply t：")
    w("")
    w("| t 桶 | 局面数 | freqW | freqD | freqL | 模型 CE | 桶内先验 CE |")
    w("|---|---|---|---|---|---|---|")
    for r in a["buckets_by_ply"]:
        if r.get("n"):
            w(f"| {r['bucket']} | {r['n']} | {fmt(r['freqW'],3)} | {fmt(r['freqD'],3)} | "
              f"{fmt(r['freqL'],3)} | {fmt(r['model_ce'])} | {fmt(r['prior_ce'])} |")
    w("")
    w("按距终局 d = n_plies − t：")
    w("")
    w("| d 桶 | 局面数 | freqW | freqD | freqL | 模型 CE | 桶内先验 CE |")
    w("|---|---|---|---|---|---|---|")
    for r in a["buckets_by_dist_to_end"]:
        if r.get("n"):
            w(f"| {r['bucket']} | {r['n']} | {fmt(r['freqW'],3)} | {fmt(r['freqD'],3)} | "
              f"{fmt(r['freqL'],3)} | {fmt(r['model_ce'])} | {fmt(r['prior_ce'])} |")
    w("")
    w(f"> {a['truncation_note']}")
    w("")
    w(f"### 校准（10 bin，Brier = {fmt(a['calibration']['brier'])}）")
    w("")
    w("| 置信度 bin | n | 平均置信度 | 命中率 |")
    w("|---|---|---|---|")
    for r in a["calibration"]["rows"]:
        w(f"| {r['bin']} | {r['n']} | {fmt(r['mean_conf'])} | {fmt(r['acc'], 3)} |")
    w("")
    w("## B. Dyn 与 latent 健康")
    w("")
    w("| 指标 | 值 |")
    w("|---|---|")
    w(f"| dyn mse（masked mean） | {fmt(b['dyn_mse_masked_mean'])} |")
    w(f"| delta_energy E‖Δ‖²/d | {fmt(b['delta_energy_E_norm2_over_d'])} |")
    w(f"| dyn rel err（全局 = mse/energy） | {fmt(b['dyn_rel_err_global'])} |")
    w(f"| dyn rel err（batch 均值，训练口径） | {fmt(b['dyn_rel_err_batch_mean'])} |")
    lv = b["latent_cross_position_var_per_dim"]
    w(f"| latent 跨局面方差/维 mean±std | {fmt(lv['mean'])} ± {fmt(lv['std'])} |")
    ln = b["latent_norm2"]
    w(f"| ‖x‖₂ mean±std | {fmt(ln['mean'], 2)} ± {fmt(ln['std'], 2)} |")
    w("")
    w("## C. Policy（合法着掩码后 argmax/topk）")
    w("")
    o = c["overall"]
    w(f"- 总体：Top-1 {fmt(o['top1'], 3)} / Top-3 {fmt(o['top3'], 3)}（{o['n']} 局面）")
    w(f"- 单合法着局面比例 {fmt(o['single_legal_frac'], 3)}；"
      f"剔除后 Top-1 = {fmt(o['top1_excl_single_legal'], 3)}（{o['n_excl_single_legal']} 局面）")
    w("")
    w("按 ply 桶：")
    w("")
    w("| t 桶 | n | Top-1 | Top-3 |")
    w("|---|---|---|---|")
    for label, r in c["by_ply"].items():
        w(f"| {label} | {r['n']} | {fmt(r['top1'], 3)} | {fmt(r['top3'], 3)} |")
    w("")
    w("按 Elo 权重档：")
    w("")
    w("| 档 | n | Top-1 | Top-3 |")
    w("|---|---|---|---|")
    for label, r in c["by_elo_weight"].items():
        w(f"| {label} | {r['n']} | {fmt(r['top1'], 3)} | {fmt(r['top3'], 3)} |")
    w("")
    w("## D. 动作对应性（g 头）")
    w("")
    if "skipped" in d:
        w(f"- 跳过：{d['skipped']}")
    else:
        w(f"- {d['n_pairs']} 对（a=实走着，b=随机其他合法着；正确配对 vs 交换配对）：")
        w(f"  - err_correct = {fmt(d['err_correct_mean'])}，err_swapped = {fmt(d['err_swapped_mean'])}")
        w(f"  - correct<swapped 比例 = {fmt(d['correct_less_than_swapped_frac'], 3)}")
    w("")
    if best:
        w(f"## best.pt 对照（step {best['step']}）")
        w("")
        w("| 指标 | latest | best |")
        w("|---|---|---|")
        for k in ("val_value_ce", "val_policy_ce_w", "val_dyn_mse", "val_dyn_rel_err",
                  "val_recon_whole_board_acc"):
            w(f"| {k} | {fmt(s[k])} | {fmt(best['sanity'][k])} |")
        w("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main

def evaluate_ckpt(model, batches, device) -> tuple[dict, ValAccumulator, list, list]:
    model.eval()
    acc, stored_x, stored_h = forward_val_subset(model, batches, device)
    sanity = {k: float(np.mean(v)) for k, v in acc.batch_core.items()}
    return sanity, acc, stored_x, stored_h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/stage_a_20260915/latest.pt")
    ap.add_argument("--also-best", action="store_true")
    ap.add_argument("--train-games", type=int, default=1024)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--microbatch", type=int, default=32,
                    help="val 子集选局口径（训练 --microbatch，只影响选局数量）")
    ap.add_argument("--fwd-batch", type=int, default=4,
                    help="前向评估批大小（局数）；训练占用 GPU 时用小值防 OOM")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "shards"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-d-samples", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.ckpt))
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    # 训练占满 GPU 时，直接 map_location=cuda 会在反序列化阶段 OOM；先 CPU 加载再搬模型
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    step = int(ckpt.get("step", -1)) + 1
    model = SeqModel(dropout=0.1)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    del ckpt
    print(f"加载 {args.ckpt}（step {step}），耗时 {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    ds = SequenceDataset(args.data, workers=args.workers)
    val_idxs = val_subset_indices(ds, args.val_batches, args.microbatch)
    train_idxs = train_subset_indices(ds, args.train_games, TRAIN_FREQ_SEED)
    q_train = train_wdl_freq_from_meta(ds, train_idxs)
    print(f"数据集 {ds.n_games} 局；val 子集 {len(val_idxs)} 局（seed {VAL_SEED}），"
          f"训练频率子集 {len(train_idxs)} 局（seed {TRAIN_FREQ_SEED}）；"
          f"装载耗时 {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    val_batches = make_batches(ds, val_idxs, args.fwd_batch)
    print(f"重放 val 子集 {len(val_batches)} 批，耗时 {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    sanity, acc, stored_x, stored_h = evaluate_ckpt(model, val_batches, device)
    print(f"前向评估完成（{acc.n_pos} 局面），耗时 {time.time()-t0:.1f}s", flush=True)
    print("sanity: " + ", ".join(f"{k} {v:.4f}" for k, v in sanity.items()), flush=True)

    value = analyze_value(acc, q_train)
    dyn_latent = analyze_dyn_latent(acc)
    policy = analyze_policy(acc)
    # 全局面口径（与训练 8×32 批平均的微小差异：批大小不同导致的加权不同）
    sanity_global = {
        "val_value_ce_global": value["model_ce_on_val"],
        "val_policy_ce_w_global": float(np.concatenate(acc.pol_ce_w).mean()
                                        / np.concatenate(acc.elo).mean()),
        "val_policy_ce_uw_global": float(np.concatenate(acc.pol_ce_uw).mean()),
    }
    sanity = {**sanity, **sanity_global}

    t0 = time.time()
    try:
        diag_d = action_correspondence(model, ds, val_batches, stored_x, stored_h,
                                       device, n_samples=args.n_d_samples)
    except Exception as exc:  # noqa: BLE001 - 诊断脚本不因块 D 失败而整体失败
        import traceback
        diag_d = {"skipped": f"块 D 异常：{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()}
    print(f"块 D 完成：{diag_d}，耗时 {time.time()-t0:.1f}s", flush=True)

    report: dict = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "ckpt": os.path.abspath(args.ckpt),
        "step": step,
        "data": os.path.abspath(args.data),
        "val_games": int(len(val_idxs)),
        "train_freq_games": int(len(train_idxs)),
        "sanity": sanity,
        "value": value,
        "dyn_latent": dyn_latent,
        "policy": policy,
        "action_correspondence": diag_d,
        "best": None,
    }

    if args.also_best:
        best_path = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "best.pt")
        if os.path.exists(best_path):
            bck = torch.load(best_path, map_location="cpu", weights_only=False)
            bmodel = SeqModel(dropout=0.1)
            bmodel.load_state_dict(bck["model"])
            bmodel.to(device)
            bsanity, _, _, _ = evaluate_ckpt(bmodel, val_batches, device)
            report["best"] = {"step": int(bck.get("step", -1)) + 1, "sanity": bsanity,
                              "train_val_at_save": bck.get("val")}
            del bmodel
            print("best sanity: " + ", ".join(f"{k} {v:.4f}" for k, v in bsanity.items()),
                  flush=True)
        else:
            report["best"] = {"skipped": f"{best_path} 不存在"}

    md = build_markdown(report)
    stem = os.path.join(out_dir, f"diag_{step}")
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write(md)
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(report), fh, ensure_ascii=False, indent=1)
    print(f"已写出 {stem}.md / {stem}.json", flush=True)
    print("\n" + md)

    ds.close()


if __name__ == "__main__":
    main()
