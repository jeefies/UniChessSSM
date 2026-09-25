"""验收 #6：单 batch 过拟合冒烟（设计文档 §10.1）。

固定一小 batch（真实小样本棋谱序列）训练百步：总损失显著下降、无 NaN、各分量有限。
需要 CUDA（R 主干前向/反向仅 GPU 可行）；无 GPU 时跳过。
"""

from __future__ import annotations
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_IMPORT_ROOT = _os.path.dirname(_HERE)   # import 根：~/UniChess：SSM 与 Kit 都是它的顶层包
HERE = _HERE
KIT_ROOT = _os.environ.get("UNICHESS_KIT_ROOT", _os.path.join(_IMPORT_ROOT, "Kit"))
if _IMPORT_ROOT not in _sys.path:
    _sys.path.insert(0, _IMPORT_ROOT)
if _os.path.isdir(KIT_ROOT) and KIT_ROOT not in _sys.path:
    _sys.path.append(KIT_ROOT)   # 追加而非前插：Kit 的 tests 包不得遮蔽本仓库的 tests


import math
import unittest

import torch

from SSM.conditions import TimeControlBucket
from SSM.model import TrainBatch


def _load_sample_batch(max_games: int = 4, max_ply: int = 60):
    """从 data/samples/sample.pgn（或联网取样）构造一个整序列批。"""
    import glob
    import os

    import numpy as np

    from SSM.dataset.pgns import fetch_sample_pgns, iter_games
    from SSM.dataset.sequences import game_to_sequence

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = sorted(
        glob.glob(os.path.join(here, "tests", "fixtures", "*.pgn"))
        + glob.glob(os.path.join(here, "data", "samples", "*.pgn"))
    )
    games = []
    for path in candidates:
        for game, meta in iter_games(path):
            games.append((game, meta))
            if len(games) >= max_games:
                break
        if len(games) >= max_games:
            break
    if not games:
        out = os.path.join(here, "data", "samples", "lichess_sample.pgn")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fetch_sample_pgns(out)
        for game, meta in iter_games(out):
            games.append((game, meta))
            if len(games) >= max_games:
                break
    if not games:
        raise unittest.SkipTest("无可用样本棋谱（离线且未 bundled sample.pgn）")

    seqs = [game_to_sequence(g, m)[:max_ply] for g, m in games]
    t = max(len(s) for s in seqs)
    bsz, n_actions = len(seqs), len(seqs[0][0].legal_mask)
    device = "cuda"
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


@unittest.skipUnless(torch.cuda.is_available(), "R 主干前向/反向需 CUDA")
class OverfitSmokeTest(unittest.TestCase):
    def test_overfit_tiny_batch(self):
        from SSM.model import losses
        from SSM.model import SeqModel

        torch.manual_seed(0)
        model = SeqModel(dropout=0.1).float().cuda()
        batch = _load_sample_batch()
        weights = losses.LossWeights()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.1)

        first = last = None
        for step in range(120):
            total, metrics = model.forward_train(batch, weights, step=step, total_steps=100000)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for v in metrics.values():
                self.assertFalse(math.isnan(v), f"step {step} 出现 NaN: {metrics}")
            if first is None:
                first = metrics["loss_total"]
            last = metrics["loss_total"]
        self.assertLess(last, first * 0.9, f"损失未显著下降: {first} -> {last}")
        print(f"\n过拟合冒烟: {first:.4f} -> {last:.4f}")


if __name__ == "__main__":
    unittest.main()
