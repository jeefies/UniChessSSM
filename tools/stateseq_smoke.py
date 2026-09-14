"""阶段 0 冒烟：小批 PGN 跑通 E/R/f/D/g 与数据管线（设计文档 §8 阶段 0）。

用法（远端）：
    python tools/stateseq_smoke.py [--pgn tests/fixtures/sample.pgn] [--games 4] [--ply 60]

输出：参数预算实测（对照 §5.6 ~27M）、整序列 forward_train 五损失与诊断指标。
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from stateseq import losses  # noqa: E402
from stateseq.conditions import TimeControlBucket  # noqa: E402
from stateseq.data.pgns import iter_games  # noqa: E402
from stateseq.data.sequences import game_to_sequence  # noqa: E402
from stateseq.model import SeqModel, TrainBatch, count_parameters  # noqa: E402


def build_batch(pgn_path: str, max_games: int, max_ply: int, device: str) -> TrainBatch:
    games = []
    for game, meta in iter_games(pgn_path):
        games.append((game, meta))
        if len(games) >= max_games:
            break
    if not games:
        raise SystemExit(f"未从 {pgn_path} 取到合格棋局（需 ≥{10} ply、Standard、有结果）")
    seqs = [game_to_sequence(g, m)[:max_ply] for g, m in games]
    t = max(len(s) for s in seqs)
    bsz, n_actions = len(seqs), len(seqs[0][0].legal_mask)
    features = torch.zeros(bsz, t, 785, device=device)
    legal = torch.zeros(bsz, t, n_actions, dtype=torch.bool, device=device)
    actions = torch.zeros(bsz, t, dtype=torch.long, device=device)
    results = torch.zeros(bsz, t, dtype=torch.long, device=device)
    moves_left = torch.zeros(bsz, t, device=device)
    color = torch.zeros(bsz, t, dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        for j, r in enumerate(seq):
            features[i, j] = torch.from_numpy(r.features).to(device)
            legal[i, j] = torch.from_numpy(r.legal_mask).to(device)
            actions[i, j] = r.action
            results[i, j] = r.result
            moves_left[i, j] = float(r.moves_left)
            color[i, j] = r.color
    elo_w = torch.ones(bsz, device=device)
    tc = torch.full((bsz,), int(TimeControlBucket.UNKNOWN), dtype=torch.long, device=device)
    elo = torch.zeros(bsz, device=device)
    return TrainBatch(features, actions, legal, results, moves_left, elo_w, tc, elo, color)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", default=os.path.join(HERE, "tests", "fixtures", "sample.pgn"))
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--ply", type=int, default=60)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = SeqModel(dropout=0.1).to(args.device)
    counts = count_parameters(model)
    print("== 参数预算实测（对照 §5.6 预算 ~27M） ==")
    for k, v in counts.items():
        print(f"  {k:12s} {v/1e6:8.3f} M")
    if args.device == "cpu":
        print("!! CPU 上 Mamba2 前向不可用，仅打印参数预算；前向冒烟请用 --device cuda")

    batch = build_batch(args.pgn, args.games, args.ply, args.device)
    weights = losses.LossWeights()
    total, metrics = model.forward_train(batch, weights, step=0, total_steps=100000)
    total.backward()
    print("== forward_train 指标（随机初始化首轮） ==")
    for k, v in metrics.items():
        print(f"  {k:24s} {v:.6f}")
    assert total.isfinite(), "总损失非有限"
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
