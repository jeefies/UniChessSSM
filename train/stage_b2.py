"""Stage B · Gumbel 自对弈训练器（规格 §2.6，改自 stage_a.py）。

超参（§3 锁定）：AdamW β(0.9,0.999) wd 0.1；lr 3e-5→3e-6 cosine，warmup min(200, 步数×10%)；
grad clip 1.0；bf16 autocast；有效 batch 512 局；每代遍历预算 min(3×buffer÷512, 2000) 步。
checkpoint/原子保存/SIGTERM：沿用 Stage A。

用法（远端）：
    python train/stage_b2.py --data data/shards_v3 --selfplay data/shards_selfplay \
        --out runs/stage_b2_20260916 --ckpt runs/stage_a_20260915/best.pt \
        --microbatch 32 --accum 16 --workers 12
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from stateseq import losses  # noqa: E402
from stateseq.data.dataset import SequenceDataset  # noqa: E402
from stateseq.data.gshards import V3ShardReader  # noqa: E402
from stateseq.model import SeqModel, count_parameters  # noqa: E402

_STOP = False


def _handle_sigterm(signum, frame):  # noqa: ANN001
    global _STOP
    _STOP = True


MODULE_PREFIXES = [("cond", "cond"), ("e.", "E"), ("r.", "R"), ("f.", "f"), ("d.", "D"), ("g.", "g")]


def grad_norms(model: SeqModel) -> dict[str, float]:
    sq: dict[str, float] = {label: 0.0 for _, label in MODULE_PREFIXES}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        for prefix, label in MODULE_PREFIXES:
            if name.startswith(prefix):
                sq[label] += float(g.pow(2).sum())
                break
    return {f"gn_{k}": math.sqrt(v) for k, v in sq.items()}


def save_atomic(state: dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="v2 人类棋谱分片目录")
    ap.add_argument("--selfplay", required=True, help="v3 自对弈分片目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True, help="Stage A best.pt 路径")
    ap.add_argument("--microbatch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=0, help="0 = 自动 min(200, 步数×10%)")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 数据源
    human_ds = SequenceDataset(args.data, workers=args.workers)
    try:
        sp_ds = SequenceDataset(args.selfplay, workers=args.workers)
    except Exception:
        sp_ds = None

    eff_batch = args.microbatch * args.accum
    buffer_games = len(human_ds.train_indices) + (len(sp_ds.train_indices) if sp_ds else 0)
    steps_total = min(3 * buffer_games // eff_batch, 2000)
    warmup = min(200, int(steps_total * 0.1)) if args.warmup == 0 else args.warmup
    print(f"训练局数 human={len(human_ds.train_indices)} selfplay={len(sp_ds.train_indices) if sp_ds else 0}；"
          f"有效 batch {eff_batch}；总步数 {steps_total}；warmup {warmup}", flush=True)

    model = SeqModel(dropout=0.1).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    print("参数: " + ", ".join(f"{k} {v/1e6:.2f}M" for k, v in count_parameters(model).items()), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                            weight_decay=0.1, fused=(device == "cuda"))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(steps_total - warmup, 1)
        return 0.1 + 0.45 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    weights = losses.LossWeights()

    start_step = 0
    best_val = float("inf")
    latest = os.path.join(args.out, "latest.pt")
    if args.resume and os.path.exists(latest):
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_step = ckpt["step"] + 1
        best_val = ckpt.get("best_val", best_val)
        torch.set_rng_state(ckpt["rng_cpu"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(ckpt["rng_gpu"].cpu())
        print(f"恢复自 step {start_step}（best_val {best_val:.4f}）", flush=True)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    metrics_path = os.path.join(args.out, "metrics.jsonl")
    mfh = open(metrics_path, "a", encoding="utf-8")

    def run_val() -> dict[str, float]:
        model.eval()
        agg: dict[str, list[float]] = {}
        for vb in human_ds.val_batch(args.val_batches, args.microbatch, device):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, m = model.forward_train(vb["batch"], weights, step=0, total_steps=steps_total,
                                           valid_mask=vb["valid"])
            for k, v in m.items():
                if k.startswith("loss_") or k in ("recon_whole_board_acc", "dyn_rel_err"):
                    agg.setdefault(k, []).append(v)
        model.train()
        return {f"val_{k}": float(np.mean(v)) for k, v in agg.items()}

    def soft_ce_from_pipol(logits: torch.Tensor, pipol_actions: torch.Tensor, pipol_probs: torch.Tensor,
                           valid: torch.Tensor, elo_w: torch.Tensor) -> torch.Tensor:
        """从 v3 pipol 重建 soft target 并计算软 CE。"""
        b, t = logits.shape[:2]
        target = torch.zeros_like(logits)
        for i in range(b):
            for j in range(t):
                if not valid[i, j]:
                    continue
                acts = pipol_actions[i, j]
                probs = pipol_probs[i, j]
                if len(acts) == 0:
                    continue
                target[i, j, acts] = probs.to(target.device)
        log_p = torch.log_softmax(logits, dim=-1)
        ce = -(target * log_p).sum(dim=-1)
        w = elo_w.unsqueeze(1).expand_as(ce) * valid.float()
        return (ce * w).sum() / w.sum().clamp(min=1e-8)

    model.train()
    t0 = time.time()
    pos_seen = 0
    pending_metrics: dict[str, float] = {}
    batch_iter = None

    for step in range(start_step, steps_total):
        opt.zero_grad(set_to_none=True)
        data_wait = 0.0
        for _ in range(args.accum):
            t_data = time.time()
            if batch_iter is None:
                batch_iter = iter(human_ds.epoch_batches(args.microbatch, device, shuffle=True))
            try:
                item = next(batch_iter)
            except StopIteration:
                batch_iter = iter(human_ds.epoch_batches(args.microbatch, device, shuffle=True))
                item = next(batch_iter)
            data_wait += time.time() - t_data
            with torch.autocast("cuda", dtype=torch.bfloat16):
                total, m = model.forward_train(item["batch"], weights, step, steps_total,
                                               valid_mask=item["valid"])
            (total / args.accum).backward()
            pos_seen += int(item["valid"].sum())
            pending_metrics = m
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if (step + 1) % args.log_every == 0 or step == start_step:
            rec = {
                "step": step + 1, "lr": sched.get_last_lr()[0],
                "pos_per_s": pos_seen / max(time.time() - t0, 1e-6),
                "data_wait_s": data_wait,
                **pending_metrics, **grad_norms(model),
            }
            mfh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mfh.flush()
            print(f"step {step+1}/{steps_total} loss {pending_metrics['loss_total']:.4f} "
                  f"pol {pending_metrics['loss_policy']:.3f} val {pending_metrics['loss_value']:.3f} "
                  f"pos/s {rec['pos_per_s']:.0f} data_wait {data_wait*1000:.0f}ms", flush=True)
            t0, pos_seen = time.time(), 0

        if (step + 1) % args.val_every == 0 or (step + 1) == steps_total:
            vm = run_val()
            mfh.write(json.dumps({"step": step + 1, **vm}, ensure_ascii=False) + "\n")
            mfh.flush()
            print(f"  VAL {vm}", flush=True)
            if vm.get("val_loss_policy", float("inf")) < best_val:
                best_val = vm["val_loss_policy"]
                save_atomic({"model": model.state_dict(), "step": step, "val": vm},
                            os.path.join(args.out, "best.pt"))

        if (step + 1) % args.save_every == 0:
            save_atomic({
                "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "step": step, "args": vars(args), "best_val": best_val,
                "rng_cpu": torch.get_rng_state(), "rng_gpu": torch.cuda.get_rng_state() if torch.cuda.is_available() else torch.tensor([]),
            }, latest)
            print(f"  saved latest.pt @ step {step+1}", flush=True)

        if _STOP:
            save_atomic({
                "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "step": step, "args": vars(args), "best_val": best_val,
                "rng_cpu": torch.get_rng_state(), "rng_gpu": torch.cuda.get_rng_state() if torch.cuda.is_available() else torch.tensor([]),
            }, latest)
            print(f"SIGTERM：已保存 latest.pt @ step {step+1}，退出", flush=True)
            break

    human_ds.close()
    if sp_ds:
        sp_ds.close()
    mfh.close()
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
