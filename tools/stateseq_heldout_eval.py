# Held-out 评估：从 val 划分采与 best 选择完全不相交的 4,096 局，统一口径出最终指标。
# 用法：python tools/stateseq_heldout_eval.py [--ckpt runs/stage_a_20260915/best.pt] [--games 4096]
"""评审 review.txt 待办②：独立 held-out 评估。

- 取样：val 划分（97,123 局，从未参与训练），seed 20260916，排除训练时验证用过的
  seed-777 子集（256 局）——与 best.pt 的选择完全不相交。
- 指标口径与 tools/stateseq_diag.py 一致（masked mean CE / bf16 autocast / eval+no_grad）。
- 先验频率仍用训练子集 meta 估计（TRAIN_FREQ_SEED 子集，与诊断相同，保证可比）。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import stateseq_diag as d  # noqa: E402  # tools/ 下的诊断模块（复用其全部内部函数）
from stateseq.model import SeqModel  # noqa: E402

HELDOUT_SEED = 20260916


def heldout_indices(ds, n_games: int, exclude: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(HELDOUT_SEED)
    pool = np.setdiff1d(ds.val_indices, np.asarray(exclude))
    return rng.choice(pool, size=min(n_games, pool.size), replace=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/stage_a_20260915/best.pt")
    ap.add_argument("--games", type=int, default=4096)
    ap.add_argument("--fwd-batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "shards"))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.ckpt))
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    step = int(ckpt.get("step", -1)) + 1
    model = SeqModel(dropout=0.1)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    del ckpt
    print(f"加载 {args.ckpt}（step {step}），耗时 {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    ds = d.SequenceDataset(args.data, workers=args.workers)
    sel777 = d.val_subset_indices(ds, 8, 32)          # 训练时验证/best 选择用过的 256 局
    idxs = heldout_indices(ds, args.games, sel777)
    train_idxs = d.train_subset_indices(ds, 2048, d.TRAIN_FREQ_SEED)
    q_train = d.train_wdl_freq_from_meta(ds, train_idxs)
    overlap = np.intersect1d(idxs, sel777).size
    print(f"数据集 {ds.n_games} 局；held-out {len(idxs)} 局（val 划分，seed {HELDOUT_SEED}，"
          f"与 seed-777 子集交集 {overlap}）；装载耗时 {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    batches = d.make_batches(ds, idxs, args.fwd_batch)
    print(f"重放 {len(batches)} 批，耗时 {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    sanity, acc, _, _ = d.evaluate_ckpt(model, batches, device)
    print(f"前向评估完成（{acc.n_pos} 局面），耗时 {time.time()-t0:.1f}s", flush=True)
    print("sanity: " + ", ".join(f"{k} {v:.4f}" for k, v in sanity.items()), flush=True)

    value = d.analyze_value(acc, q_train)
    dyn_latent = d.analyze_dyn_latent(acc)
    policy = d.analyze_policy(acc)

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "ckpt": os.path.abspath(args.ckpt),
        "step": step,
        "heldout_games": int(len(idxs)),
        "heldout_seed": HELDOUT_SEED,
        "heldout_positions": int(acc.n_pos),
        "disjoint_from_best_selection": int(overlap) == 0,
        "sanity": sanity,
        "value": value,
        "dyn_latent": dyn_latent,
        "policy": policy,
    }
    base = os.path.join(out_dir, f"heldout{len(idxs)}_step{step}")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(d.to_jsonable(report), fh, ensure_ascii=False, indent=2)

    L = []
    w = L.append
    w(f"# Stage A held-out 评估 @ step {step}（{len(idxs)} 局，seed {HELDOUT_SEED}）")
    w("")
    w(f"- 局面数：{acc.n_pos}；与 best 选择子集不相交：{overlap == 0}")
    v = value
    w(f"- value：先验 CE {v['prior_ce_on_val']:.4f} → 模型 CE {v['model_ce_on_val']:.4f}，"
      f"value_gain = {v['value_gain']:.4f} nats")
    w(f"- policy：Top-1 {policy['overall']['top1']:.4f} / Top-3 {policy['overall']['top3']:.4f}"
      f"（{policy['overall']['n']} 局面）")
    b = dyn_latent
    w(f"- dyn：mse {b['dyn_mse']:.4f} / energy {b['delta_energy']:.4f} / rel {b['dyn_rel_global']:.4f}")
    w(f"- recon 全盘 acc：{sanity.get('val_recon_whole_board_acc', float('nan')):.4f}")
    w("")
    w("分桶 value CE（按距终局 d）：")
    for row in v["by_dist"]["rows"]:
        w(f"- d={row['label']}：n={row['n']}，模型 CE {row['model_ce']:.4f}，先验 {row['prior_ce']:.4f}")
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print(base + ".json / .md", flush=True)


if __name__ == "__main__":
    main()
