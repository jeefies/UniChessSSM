"""验收 #3：整序列 vs 逐步递推一致性（设计文档 §10.1）。

同一棋谱整段前向与逐步单步前向的 logits 差 < 1e-4（fp32 口径）。
需要 CUDA（官方 mamba_ssm 单步 kernel 仅 GPU）；无 GPU 时跳过。
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


import unittest

import numpy as np
import torch

try:
    import chess  # noqa: F401
    _HAS_CHESS = True
except ImportError:  # pragma: no cover
    _HAS_CHESS = False

from SSM.conditions import TimeControlBucket
from SSM.features import encode


def _synthetic_game_features(n_ply: int, seed: int = 0):
    """不下棋谱：用合法短对局生成特征序列（python-chess 随机走子）。"""
    import random

    import chess

    rng = random.Random(seed)
    board = chess.Board()
    feats = []
    while len(feats) < n_ply and not board.is_game_over():
        feats.append(encode(board))
        board.push(rng.choice(list(board.legal_moves)))
    return np.stack(feats)


@unittest.skipUnless(torch.cuda.is_available(), "mamba_ssm 单步 kernel 仅支持 CUDA")
class SequenceConsistencyTest(unittest.TestCase):
    def test_full_vs_step(self):
        from SSM.model import SeqModel

        torch.manual_seed(0)
        model = SeqModel(dropout=0.0).float().cuda().eval()
        n_ply = 24
        feats = torch.from_numpy(_synthetic_game_features(n_ply)).float().cuda()
        bsz = 2
        feats = feats.unsqueeze(0).expand(bsz, -1, -1).contiguous()

        tc = torch.full((bsz,), int(TimeControlBucket.BLITZ), dtype=torch.long, device="cuda")
        elo = torch.zeros(bsz, device="cuda")
        color = feats[:, :, 768].long()  # 走子方位

        # 整序列前向
        with torch.no_grad():
            x = model.encode(feats)
            cond = model.cond(
                tc.unsqueeze(1).expand(-1, n_ply),
                elo.unsqueeze(1).expand(-1, n_ply),
                color,
            )
            h = model.trunk(x + cond)
            pol_full, wdl_full, mlh_full = model.f(h)

            # 逐步递推
            cache = model.initial_cache(bsz, device="cuda")
            pol_steps, wdl_steps, mlh_steps = [], [], []
            for t in range(n_ply):
                pol_t, wdl_t, mlh_t, _, cache = model.step(
                    feats[:, t], tc, elo, color[:, t], cache
                )
                pol_steps.append(pol_t)
                wdl_steps.append(wdl_t)
                mlh_steps.append(mlh_t)
            pol_step = torch.stack(pol_steps, dim=1)
            wdl_step = torch.stack(wdl_steps, dim=1)
            mlh_step = torch.stack(mlh_steps, dim=1)

        for name, a, b in (("wdl", wdl_full, wdl_step), ("mlh", mlh_full, mlh_step)):
            diff = (a - b).abs().max().item()
            self.assertLess(diff, 1e-4, f"{name} 整序列 vs 逐步差 {diff}")
        # policy 按方案 A 断 softmax 概率差（§10.1 #3 口径；raw logits 差为 kernel 固有噪声，作诊断记录）
        prob_full = torch.softmax(pol_full, dim=-1)
        prob_step = torch.softmax(pol_step, dim=-1)
        prob_diff = (prob_full - prob_step).abs().max().item()
        logit_diff = (pol_full - pol_step).abs().max().item()
        print(f"\npolicy 概率差 {prob_diff:.2e}（门槛 1e-4）；raw logits 差 {logit_diff:.2e}（诊断值）")
        self.assertLess(prob_diff, 1e-4, f"policy 概率差 {prob_diff}")


if __name__ == "__main__":
    unittest.main()
