"""训练指标绘图：metrics.jsonl → 多面板 PNG（损失 / 诊断 / grad norm / 吞吐 / val）。"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def smooth(xs, ys, w=11):
    if len(xs) < w:
        return xs, ys
    k = [1.0 / w] * w
    ys_s = [sum(ys[max(0, i - w // 2):i + w // 2 + 1]) / len(ys[max(0, i - w // 2):i + w // 2 + 1]) for i in range(len(ys))]
    return xs, ys_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(HERE, "runs", "stage_a_20260915"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="Stage A 训练全景")
    args = ap.parse_args()
    out = args.out or os.path.join(args.run, "metrics.png")

    train, val = [], []
    with open(os.path.join(args.run, "metrics.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            (val if "val_loss_total" in rec else train).append(rec)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    def series(recs, key):
        xs = [r["step"] for r in recs if key in r]
        ys = [r[key] for r in recs if key in r]
        return xs, ys

    ax = axes[0][0]
    for key, label in [("loss_policy", "policy CE"), ("loss_value", "value CE"),
                       ("loss_recon", "recon"), ("loss_dyn", "dyn")]:
        xs, ys = series(train, key)
        ax.plot(xs, ys, alpha=0.25)
        xs_s, ys_s = smooth(xs, ys)
        ax.plot(xs_s, ys_s, label=label)
    ax.set_yscale("log")
    ax.set_title("train losses (log, raw + smoothed)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    xs, ys = series(train, "loss_mlh")
    ax.plot(xs, ys, alpha=0.25)
    xs, ys = smooth(xs, ys)
    ax.plot(xs, ys, label="mlh Huber")
    ax.set_title("moves-left Huber loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0][2]
    for key, label in [("recon_whole_board_acc", "whole-board acc"), ("dyn_rel_err", "dyn rel err")]:
        xs, ys = series(train, key)
        ax.plot(xs, ys, label=label)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.set_title("recon / dyn 健康 (§10.2)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    for key in ["gn_E", "gn_R", "gn_f", "gn_D", "gn_g", "gn_cond"]:
        xs, ys = series(train, key)
        ax.plot(xs, ys, label=key[3:])
    ax.set_yscale("log")
    ax.set_title("grad norm by module (§11)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    xs, ys = series(train, "pos_per_s")
    xs, ys = smooth(xs, ys, 21)
    ax.plot(xs, ys)
    ax.set_title("throughput (pos/s)")
    ax.grid(alpha=0.3)

    ax = axes[1][2]
    for key, label in [("val_loss_policy", "val policy"), ("val_loss_value", "val value"),
                       ("val_loss_recon", "val recon"), ("val_loss_dyn", "val dyn")]:
        xs, ys = series(val, key)
        if xs:
            ax.plot(xs, ys, marker="o", label=label)
    ax.set_yscale("log")
    ax.set_title("validation (every 1000 steps)")
    ax.legend()
    ax.grid(alpha=0.3)

    vlast = val[-1] if val else {}
    fig.suptitle(
        f"{args.title} @ step {train[-1]['step']}/37758（1 epoch 完成）  |  "
        f"final val: policy CE {vlast.get('val_loss_policy', float('nan')):.3f}  "
        f"value CE {vlast.get('val_loss_value', float('nan')):.3f}  "
        f"recon {vlast.get('val_loss_recon', float('nan')):.4f}  "
        f"dyn {vlast.get('val_loss_dyn', float('nan')):.3f}  "
        f"pos/s {train[-1]['pos_per_s']:.0f}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
