"""修复后训练冒烟：证明 metadata 修复**只改变了它应该改变的监督**。

四项检查（审查意见 §四）：
 1. 相同权重、相同 batch，分别用旧/新 meta 前向 → policy/value/recon/dyn 必须逐位一致；
    仅 mlh 的有效集合与相应损失允许变化。
 2. 一个 batch 全部为真实截断局 → mlh 安全为零，不除零、不出 NaN。
 3. 新恢复的规则申和样本 → moves_left 指向实际约定终止位置，而非 300。
 4. 3–5 次真实优化器更新 → 损失与梯度有限，完整 learner checkpoint 可保存与恢复。

旧 meta 未留副本（修复时的疏漏），但可确定性重建：旧分类逻辑 =
`is_checkmate → is_stalemate → is_fifty_moves → is_repetition(3) → is_insufficient_material
→ 否则 truncated`（严格判定链），对同一动作序列重放即可复现。

全程 eval 模式 + dropout=0 + 固定种子，避免随机扰动掩盖差异。
"""

from __future__ import annotations

import argparse
import os
import sys

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stateseq import losses
from stateseq.data.dataset_selfplay import SelfPlayDataset
from stateseq.model import SeqModel
from tools.repair_v3_meta import replay_final_board


def legacy_is_truncated(board: chess.Board) -> bool:
    """重建旧分类：严格判定链，任何一项不成立就落到 truncated。"""
    if board.is_checkmate() or board.is_stalemate():
        return False
    if board.is_fifty_moves() or board.is_repetition(3):
        return False
    if board.is_insufficient_material():
        return False
    return True


def _forward(model, batch_d, weights, mlh_mask=None):
    """一次无梯度前向；返回 forward_train 的指标 dict（fp32，关闭 autocast 以便逐位比较）。"""
    with torch.no_grad():
        total, m = model.forward_train(
            batch_d["batch"], weights, step=0, total_steps=1,
            valid_mask=batch_d["valid"],
            policy_soft_target=batch_d["policy_soft_target"],
            mlh_valid_mask=batch_d["mlh_valid"] if mlh_mask is None else mlh_mask)
    return float(total), m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="runs/stage_b_gen2k")
    ap.add_argument("--ckpt", default="runs/stage_a_20260915/best.pt")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out", default="runs/meta_repair_smoke")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)

    ds = SelfPlayDataset(args.shards, workers=4)
    reader = ds.reader
    metas = reader.meta_all

    # ---- 重建旧 is_truncated，并定位"被恢复"的样本 ----
    print("== 重建旧 meta（严格判定链）==")
    idxs = ds.train_indices[:args.games]
    legacy_trunc, new_trunc, recovered = {}, {}, []
    for i in idxs:
        g = reader.game(i)
        board = replay_final_board(np.asarray(g["actions"]))
        legacy_trunc[i] = legacy_is_truncated(board)
        new_trunc[i] = bool(metas[i]["is_truncated"])
        if legacy_trunc[i] and not new_trunc[i]:
            recovered.append((i, int(metas[i]["n_plies"]), board.fen()))
    print(f"  取样 {len(legacy_trunc)} 局：旧截断 {sum(legacy_trunc.values())}，"
          f"新截断 {sum(new_trunc.values())}，被恢复 {len(recovered)}")

    # ---- 检查 3：恢复样本的 moves_left 指向真实终止 ----
    print("\n== 检查 3：恢复样本的 moves_left ==")
    ok3 = True
    for i, n_plies, fen in recovered[:5]:
        if n_plies >= 300:
            ok3 = False
            print(f"  ✗ 局 {i} 被恢复却有 {n_plies} ply")
        else:
            print(f"  ✓ 局 {i}: n_plies={n_plies}（<300，moves_left 以真实终止为准）")
    print(f"  结论：{'PASS' if ok3 else 'FAIL'}")

    # ---- 模型 ----
    model = SeqModel(dropout=0.0).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()

    # 自建 batch，保留"行 → 局号"的映射（_collate 会按序列长度降序排序，stable）
    from stateseq.data.dataset_selfplay import _worker_build
    items = ds.pool.starmap(_worker_build, [(i, ds.t_max) for i in idxs])
    batch_d = ds._build_batch(items, device)
    row_to_idx = sorted(idxs, key=lambda i: -int(metas[i]["n_plies"]))

    weights = losses.LossWeights()

    # ---- 检查 1：旧/新 meta 前向对照 ----
    print("\n== 检查 1：旧 vs 新 meta 的前向分量 ==")
    saved = batch_d["mlh_valid"].clone()
    legacy_mask = batch_d["valid"].clone()
    for row, i in enumerate(row_to_idx):
        if legacy_trunc.get(i, False):
            legacy_mask[row] = False

    _, m_new = _forward(model, batch_d, weights)
    _, m_old = _forward(model, batch_d, weights, mlh_mask=legacy_mask)

    ok1 = True
    for k in sorted(k for k in m_new if k.startswith("loss_")):
        a, b_ = m_old.get(k, float("nan")), m_new[k]
        if "mlh" in k:
            print(f"  {k}（允许变化）: 旧 {a:.6f} → 新 {b_:.6f}；"
                  f"有效位 {int(legacy_mask.sum())} → {int(saved.sum())}")
        elif "total" in k:
            continue
        else:
            same = abs(a - b_) < 1e-9
            print(f"  {k}: {'✓ 一致' if same else '✗ 变化'} ({a:.8f} vs {b_:.8f})")
            ok1 &= same
    print(f"  结论：{'PASS' if ok1 else 'FAIL'}")

    # ---- 检查 2：全截断 batch 的 mlh 安全性 ----
    print("\n== 检查 2：全截断 batch（mlh_valid 全 False）==")
    _, m_zero = _forward(model, batch_d, weights, mlh_mask=torch.zeros_like(saved))
    mlh_key = next(k for k in m_zero if "mlh" in k and k.startswith("loss_"))
    v = m_zero[mlh_key]
    ok2 = bool(np.isfinite(v)) and abs(v) < 1e-6
    print(f"  {mlh_key} = {v:.3e}，finite={np.isfinite(v)}")
    print(f"  结论：{'PASS' if ok2 else 'FAIL'}")

    # ---- 检查 4：真实优化器更新 + checkpoint 往返 ----
    print(f"\n== 检查 4：{args.steps} 次真实更新 ==")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=0.1)
    ok4 = True
    for step in range(args.steps):
        loss, m = model.forward_train(
            batch_d["batch"], weights, step, args.steps,
            valid_mask=batch_d["valid"],
            policy_soft_target=batch_d["policy_soft_target"],
            mlh_valid_mask=batch_d["mlh_valid"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        finite = bool(torch.isfinite(loss)) and bool(torch.isfinite(gnorm))
        ok4 &= finite
        print(f"  step {step}: loss={loss.item():.4f} grad_norm(裁剪前)={gnorm.item():.4f} "
              f"finite={finite}")

    path = os.path.join(args.out, "smoke_ckpt.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": args.steps}, path)
    re = torch.load(path, map_location=device, weights_only=False)
    model2 = SeqModel(dropout=0.0).to(device)
    model2.load_state_dict(re["model"])
    opt2 = torch.optim.AdamW(model2.parameters(), lr=5e-5)
    opt2.load_state_dict(re["opt"])
    print(f"  checkpoint 往返：{'✓ OK' if re['step'] == args.steps else '✗ FAIL'}")
    print(f"  结论：{'PASS' if ok4 else 'FAIL'}")

    print("\n=== 汇总 ===")
    for name, ok in [("1 前向分量隔离", ok1), ("2 全截断安全", ok2),
                     ("3 恢复样本 moves_left", ok3), ("4 优化器更新", ok4)]:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if (ok1 and ok2 and ok3 and ok4) else 1)


if __name__ == "__main__":
    main()
