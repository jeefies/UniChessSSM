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
from stateseq.data.dataset_selfplay import B2_T_MAX, SelfPlayDataset  # noqa: E402
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
    ap.add_argument("--mem-fraction", type=float, default=0.0, help="显存软上限比例 (0.0=不限制, 0.5=限制使用至多50%%显存)")
    ap.add_argument("--threads", type=int, default=0, help="PyTorch CPU 线程数限制 (0=保持默认)")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=0, help="0 = 自动 min(200, 步数×10%%)")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--w-selfplay", type=float, default=0.85, help="来源权重（§2.6 锁定 0.85）")
    # 有效来源权重和 = 0.95（谜题 0.05 未接线）。这是整体损失缩放，**不等价于学习率乘 0.95**：
    # AdamW 的自适应分母会抵消梯度的统一缩放（m/√v 对 g→cg 不变），实际影响还受
    # 裁剪、ε 与解耦 weight decay 干扰。本轮保留 0.85/0.10，不补偿 LR，不为凑 1 强行接谜题。
    ap.add_argument("--w-human", type=float, default=0.10, help="来源权重（§2.6 锁定 0.10；谜题 0.05 plumbing 未接入，暂不参与）")
    ap.add_argument("--mlh-log", action="store_true", default=False, help="启用 mlh Log-Huber 变换 (torch.log1p(F.relu(...)), delta=0.5)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and args.mem_fraction > 0.0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
        print(f"限制 PyTorch 显存上限至 {args.mem_fraction * 100:.1f}%", flush=True)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
        print(f"限制 PyTorch CPU 计算线程数至 {args.threads}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 数据源：完整 300 ply（§2.6 不得静默继承 Stage A 的 T_MAX=200）
    human_ds = SequenceDataset(args.data, workers=args.workers, t_max=B2_T_MAX)
    sp_ds = SelfPlayDataset(args.selfplay, workers=args.workers, t_max=B2_T_MAX)
    if not sp_ds.train_indices:
        raise RuntimeError(f"自对弈分片 {args.selfplay} 无可训练局——Stage B2 的核心监督来源缺失，"
                           f"不应静默退化为纯人类数据训练，请检查生成产物")

    eff_batch = args.microbatch * args.accum
    # §2.6 遍历预算以自对弈 replay buffer 局数为准（不含人类语料——人类库固定且远大于
    # buffer，混入会使预算恒被 2000 步上限吃满，失去"控制对陈旧搜索目标过拟合遍数"的本意）。
    buffer_games = len(sp_ds.train_indices)
    steps_total = min(3 * buffer_games // eff_batch, 2000)
    if steps_total < 1:
        raise RuntimeError(f"遍历预算不足 1 步（buffer {buffer_games} 局 / 有效 batch {eff_batch}）——"
                           f"请增大 replay buffer 或减小 microbatch×accum")
    warmup = min(200, int(steps_total * 0.1)) if args.warmup == 0 else args.warmup
    print(f"训练局数 human={len(human_ds.train_indices)} selfplay={len(sp_ds.train_indices)}；"
          f"有效 batch {eff_batch}；总步数 {steps_total}；warmup {warmup}；"
          f"来源权重 w_selfplay={args.w_selfplay} w_human={args.w_human} w_puzzle=0 "
          f"active_source_weight_sum={args.w_selfplay + args.w_human:.2f}", flush=True)

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
    # §3 锁定：损失权重（来源内）pol 1.0 / val 1.0 / recon 0.1 / dyn 0.5 / mlh 0.1。
    # losses.LossWeights 默认 w_v=0.8 是 Stage A 遗留值，Stage B 显式改为 1.0。
    # recon 固定 0.1，禁用 Stage A 的 1.0→0.1 退火（§2.6）。
    weights = losses.LossWeights(w_v=1.0, w_r_start=0.1, w_r_end=0.1)

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
                                           valid_mask=vb["valid"],
                                           log_target=args.mlh_log)
            for k, v in m.items():
                if k.startswith("loss_") or k in ("recon_whole_board_acc", "dyn_rel_err"):
                    agg.setdefault(f"human_{k}", []).append(v)
        for vb in sp_ds.val_batch(args.val_batches, args.microbatch, device):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, m = model.forward_train(vb["batch"], weights, step=0, total_steps=steps_total,
                                           valid_mask=vb["valid"],
                                           policy_soft_target=vb["policy_soft_target"],
                                           mlh_valid_mask=vb["mlh_valid"],
                                           log_target=args.mlh_log)
            for k, v in m.items():
                if k.startswith("loss_") or k in ("recon_whole_board_acc", "dyn_rel_err"):
                    agg.setdefault(f"selfplay_{k}", []).append(v)
        model.train()
        # §2.6：每代以自对弈 held-out policy CE 选当代表——val_loss_policy 取自对弈口径。
        out = {f"val_{k}": float(np.mean(v)) for k, v in agg.items()}
        if "val_selfplay_loss_policy" in out:
            out["val_loss_policy"] = out["val_selfplay_loss_policy"]
        return out

    model.train()
    t0 = time.time()
    pos_seen = 0
    pending_metrics: dict[str, float] = {}
    human_iter = None
    sp_iter = None

    step = start_step - 1  # resume 后无可跑步数时，收尾保存仍有确定的 step
    for step in range(start_step, steps_total):
        opt.zero_grad(set_to_none=True)
        data_wait = 0.0
        for _ in range(args.accum):
            t_data = time.time()
            if human_iter is None:
                human_iter = iter(human_ds.epoch_batches(args.microbatch, device, shuffle=True))
            try:
                h_item = next(human_iter)
            except StopIteration:
                human_iter = iter(human_ds.epoch_batches(args.microbatch, device, shuffle=True))
                h_item = next(human_iter)
            if sp_iter is None:
                sp_iter = iter(sp_ds.epoch_batches(args.microbatch, device, shuffle=True))
            try:
                sp_item = next(sp_iter)
            except StopIteration:
                sp_iter = iter(sp_ds.epoch_batches(args.microbatch, device, shuffle=True))
                sp_item = next(sp_iter)
            data_wait += time.time() - t_data

            # 按来源分别归约损失，再显式加权求和（§2.6；不是按样本条数占比）。
            with torch.autocast("cuda", dtype=torch.bfloat16):
                total_h, m_h = model.forward_train(h_item["batch"], weights, step, steps_total,
                                                   valid_mask=h_item["valid"],
                                                   log_target=args.mlh_log)
                total_sp, m_sp = model.forward_train(sp_item["batch"], weights, step, steps_total,
                                                     valid_mask=sp_item["valid"],
                                                     policy_soft_target=sp_item["policy_soft_target"],
                                                     mlh_valid_mask=sp_item["mlh_valid"],
                                                     log_target=args.mlh_log)
                total = args.w_selfplay * total_sp + args.w_human * total_h
            (total / args.accum).backward()
            pos_seen += int(h_item["valid"].sum()) + int(sp_item["valid"].sum())
            pending_metrics = {
                "loss_total": float(total.detach()),
                **{f"human_{k}": v for k, v in m_h.items()},
                **{f"selfplay_{k}": v for k, v in m_sp.items()},
            }
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
                  f"sp_pol {pending_metrics['selfplay_loss_policy']:.3f} "
                  f"sp_val {pending_metrics['selfplay_loss_value']:.3f} "
                  f"human_pol {pending_metrics['human_loss_policy']:.3f} "
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
    sp_ds.close()
    mfh.close()
    # Always save full checkpoint at end (short runs may never hit save_every)
    save_atomic({
        "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
        "step": step, "args": vars(args), "best_val": best_val,
        "rng_cpu": torch.get_rng_state(), "rng_gpu": torch.cuda.get_rng_state() if torch.cuda.is_available() else torch.tensor([]),
    }, latest)
    print(f"训练结束：保存 latest.pt @ step {step+1}", flush=True)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
