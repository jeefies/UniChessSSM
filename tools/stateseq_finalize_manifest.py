"""manifest 汇总：扫描 v2 分片 → 局数/步数/月表 + Elo 抽样统计 → manifest.json。

用法：python tools/stateseq_finalize_manifest.py --out data/shards
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data", "shards"))
    args = ap.parse_args()

    metas = sorted(glob.glob(os.path.join(args.out, "shard-*-w*.meta.bin")))
    shards = [os.path.basename(m)[: -len(".meta.bin")] for m in metas]
    games = steps = 0
    elo_samples: list[np.ndarray] = []
    for base in shards:
        raw = np.fromfile(os.path.join(args.out, base + ".meta.bin"), dtype=np.uint8).reshape(-1, 16)
        n_plies = raw[:, 0:2].copy().view(np.uint16).reshape(-1)
        games += len(n_plies)
        steps += int(n_plies.sum())
        es = os.path.join(args.out, base + ".elosample.bin")
        if os.path.exists(es):
            elo_samples.append(np.fromfile(es, dtype=np.float32))
    months = sorted({s.split("-")[1] for s in shards})

    elo_arr = np.concatenate(elo_samples) if elo_samples else np.array([1500.0], dtype=np.float64)
    elo_arr = elo_arr.astype(np.float64)
    e_min, e_max = np.percentile(elo_arr, [1.0, 99.0])
    clipped = np.clip(elo_arr, e_min, e_max)
    elo_stats = {
        "e_min": float(e_min), "e_max": float(e_max),
        "mean": float(clipped.mean()), "std": float(max(clipped.std(), 1.0)),
        "n": int(len(elo_arr)),
    }

    manifest = {
        "format": "v2",
        "shards": [s + "" for s in shards],
        "months": months,
        "games": games,
        "steps": steps,
        "elo_stats": elo_stats,
    }
    fd, tmp = tempfile.mkstemp(dir=args.out, suffix=".tmp") if False else (None, args.out + "/manifest.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(args.out, "manifest.json"))
    print(f"manifest: {games} 局 {steps} 步, months={months}, elo_stats={elo_stats}")


if __name__ == "__main__":
    main()
