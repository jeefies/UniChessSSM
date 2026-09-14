"""Stage A 吞吐探测：GPU 纯 fwd+bwd 峰值 + 数据加载器速度 → 推荐 microbatch/accum。

用法：python tools/stateseq_throughput.py --data data/shards_pilot [--microbatch-list 16,32,64] [--secs 30]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from stateseq import losses  # noqa: E402
from stateseq.data.dataset import SequenceDataset  # noqa: E402
from stateseq.model import SeqModel  # noqa: E402


def probe_gpu(model, microbatch: int, device: str, secs: int) -> tuple[float, float]:
    """固定 T=128 的合成批测 fwd+bwd 吞吐（位置/s）与峰值显存。"""
    torch.manual_seed(0)
    b, t = microbatch, 128
    features = torch.randn(b, t, 785, device=device)
    legal = torch.zeros(b, t, 1936, dtype=torch.bool, device=device)
    legal[:, :, :50] = True
    from stateseq.model import TrainBatch

    batch = TrainBatch(
        features=features,
        actions=torch.randint(0, 1936, (b, t), device=device),
        legal_mask=legal,
        results=torch.randint(0, 3, (b, t), device=device),
        moves_left=torch.rand(b, t, device=device) * 100,
        elo_weight=torch.ones(b, device=device),
        tc_bucket=torch.zeros(b, dtype=torch.long, device=device),
        elo_std=torch.zeros(b, device=device),
        color=torch.randint(0, 2, (b, t), device=device),
    )
    valid = torch.ones(b, t, dtype=torch.bool, device=device)
    weights = losses.LossWeights()
    model.train()
    n = 0
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    while time.time() - t0 < secs:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total, _ = model.forward_train(batch, weights, 0, 100000, valid_mask=valid)
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        n += 1
    dt = time.time() - t0
    pos = b * t * n
    return pos / dt, torch.cuda.max_memory_allocated() / 1024**3


def probe_loader(dataset, microbatch: int, device: str, secs: int) -> float:
    t0 = time.time()
    pos = 0
    for item in dataset.epoch_batches(microbatch, device, shuffle=True):
        pos += int(item["valid"].sum())
        if time.time() - t0 > secs:
            break
    return pos / (time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "shards_pilot"))
    ap.add_argument("--microbatch-list", default="16,32,64")
    ap.add_argument("--secs", type=int, default=25)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    device = "cuda"
    model = SeqModel(dropout=0.1).to(device)
    for mb in [int(x) for x in args.microbatch_list.split(",")]:
        pos_s, giB = probe_gpu(model, mb, device, args.secs)
        print(f"GPU microbatch={mb:3d} T=128: {pos_s:8.0f} pos/s, 峰值显存 {giB:.1f} GiB", flush=True)

    dataset = SequenceDataset(args.data, workers=args.workers)
    for mb in [16, 64]:
        pos_s = probe_loader(dataset, mb, device, args.secs)
        print(f"加载器 microbatch={mb:3d}: {pos_s:8.0f} pos/s（{args.workers} workers）", flush=True)
    dataset.close()


if __name__ == "__main__":
    main()
