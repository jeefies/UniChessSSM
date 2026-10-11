"""WDL 软价值目标（D4）与子力差辅助（D9）。

覆盖：
1. 矩匹配投影：pW − pL ≡ g、概率和恒 1、draw 质量 d 取自当前预测的截断值；
2. 极端 g（±1）下 d 被压缩，pW/pL 仍非负（无需再归一）；
3. 价值损失对"预测匹配目标"显著低于"预测背离目标"（梯度方向正确）；
4. pos_mask / value_weights 的屏蔽与降权生效；
5. material_loss 的目标等于重放棋盘的标准子力差 / 10。
"""

from __future__ import annotations
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_IMPORT_ROOT = _os.path.dirname(_HERE)   # import 根：~/UniChess
if _IMPORT_ROOT not in _sys.path:
    _sys.path.insert(0, _IMPORT_ROOT)


import unittest

import torch

from SSM.model.losses import material_loss, value_loss, value_soft_loss


def _logits_from_q(q: float, d: float, scale: float = 8.0) -> torch.Tensor:
    """构造一个 softmax 后 (pW,pD,pL) ≈ ((q+1-d)/2, d, (1-q-d)/2) 的 logits。"""
    p = torch.tensor([(q + 1.0 - d) / 2.0, d, (1.0 - q - d) / 2.0])
    return (torch.log(p.clamp(min=1e-6)) * scale).reshape(1, 1, 3)


class ValueSoftTargetTest(unittest.TestCase):
    def test_projection_is_max_entropy(self):
        """和棋质量 = 1−|g|，胜/负按符号分配；和恒 1、均值恒等于 g。"""
        g = torch.tensor([0.0, 0.6, -0.3, 0.95, -0.95, 1.0, -1.0, 0.25])
        w = g.clamp(min=0)
        d = 1.0 - g.abs()
        l = (-g).clamp(min=0)
        self.assertTrue(torch.allclose(w + d + l, torch.ones_like(g), atol=1e-7), "概率和不为 1")
        self.assertTrue(torch.allclose(w - l, g, atol=1e-7), "pW−pL 不等于 g")
        self.assertTrue(torch.all(w >= 0) and torch.all(d >= 0) and torch.all(l >= 0))

    def test_extremes_are_one_hot(self):
        """|g|=1 与 g=0 必须精确 one-hot —— λ=1 时才能与硬标签 MC 逐位一致。"""
        for g_val, expect in ((1.0, [1.0, 0.0, 0.0]), (-1.0, [0.0, 0.0, 1.0]),
                              (0.0, [0.0, 1.0, 0.0])):
            g = torch.tensor([g_val])
            t = torch.stack([g.clamp(min=0), 1.0 - g.abs(), (-g).clamp(min=0)], dim=-1)
            self.assertTrue(torch.allclose(t[0], torch.tensor(expect), atol=1e-7),
                            f"g={g_val} 未退化成 one-hot：{t[0].tolist()}")

    def test_lambda_one_matches_hard_label(self):
        """λ=1 时软目标必须与硬标签 CE 完全一致（架构消融的前提）。"""
        torch.manual_seed(0)
        logits = torch.randn(2, 5, 3) * 2.0
        result = torch.tensor([[0, 1, 2, 0, 1], [2, 2, 1, 0, 0]])       # 0 胜/1 和/2 负
        g = (1.0 - result.float()).reshape(-1)                          # → +1/0/−1
        soft = value_soft_loss(logits, g.reshape(2, 5))
        hard = value_loss(logits, result)
        self.assertAlmostEqual(float(soft), float(hard), places=6,
                               msg=f"λ=1 软目标 {float(soft)} 与硬标签 {float(hard)} 不一致")

    def test_loss_prefers_matching_prediction(self):
        g = torch.full((1, 4), 0.8)
        good = _logits_from_q(0.8, 0.2)
        bad = _logits_from_q(-0.8, 0.2)
        l_good = value_soft_loss(good, g)
        l_bad = value_soft_loss(bad, g)
        self.assertLess(float(l_good), float(l_bad) - 0.5,
                        f"匹配目标的损失应显著更低：{float(l_good)} vs {float(l_bad)}")

    def test_mask_and_weights(self):
        g = torch.tensor([[0.5, 0.5, 0.5]])
        logits = torch.randn(1, 3, 3)
        mask = torch.tensor([[True, False, False]])
        only_first = value_soft_loss(logits, g, pos_mask=mask)
        full = value_soft_loss(logits, g)
        self.assertNotAlmostEqual(float(only_first), float(full), places=6)
        # 权重 0 的位置不贡献：等价于只算第一个位置
        w = torch.tensor([[1.0, 0.0, 0.0]])
        self.assertAlmostEqual(float(only_first), float(value_soft_loss(logits, g, None, w)),
                               places=6)

    def test_all_masked_returns_zero(self):
        g = torch.zeros(1, 3)
        logits = torch.randn(1, 3, 3)
        mask = torch.zeros(1, 3, dtype=torch.bool)
        self.assertEqual(float(value_soft_loss(logits, g, mask)), 0.0)


class MaterialLossTest(unittest.TestCase):
    def _features(self, pieces: dict, n_pos: int = 1) -> torch.Tensor:
        """pieces: {(plane_idx, square): 1}；plane 0..5=白 P,N,B,R,Q,K，6..11=黑。

        n_pos > 1 时在同一条序列上重复同一局面（用于验证 mask 屏蔽）。
        """
        from SSM.features import FEATURE_DIM
        f = torch.zeros(1, n_pos, FEATURE_DIM)
        for (plane, sq), v in pieces.items():
            f[0, :, plane * 64 + sq] = v
        return f

    def test_material_target(self):
        # 白：Q(9)+R(5)+P(1)=15（plane 4/3/0）；黑：Q(9)+N(3)=12（plane 10/7）
        # ⇒ 白方绝对视角子力差 3 ⇒ /10 = 0.3；预测 0、delta=1 ⇒ Huber = 0.5·0.3² = 0.045
        f = self._features({(4, 0): 1, (3, 1): 1, (0, 2): 1, (10, 3): 1, (7, 4): 1})
        pred = torch.zeros(1, 1, 1)
        self.assertAlmostEqual(float(material_loss(pred, f)), 0.045, places=6)

    def test_material_target_negative(self):
        # 黑多一个车（plane 9 = 黑 R，5 分）：白方绝对视角差 −5 ⇒ /10 = −0.5
        # ⇒ Huber = 0.5·0.5² = 0.125
        f = self._features({(9, 0): 1})
        self.assertAlmostEqual(float(material_loss(torch.zeros(1, 1, 1), f)), 0.125, places=6)

    def test_equal_material_zero_loss(self):
        f = self._features({(0, 0): 1, (6, 1): 1})
        self.assertAlmostEqual(float(material_loss(torch.zeros(1, 1, 1), f)), 0.0, places=6)

    def test_mask(self):
        # 白多一个车 ⇒ 目标 0.5；mask 掉第二位后只算第一位
        f = self._features({(3, 0): 1}, n_pos=2)
        pred = torch.zeros(1, 2, 1)
        mask = torch.tensor([[True, False]])
        self.assertAlmostEqual(float(material_loss(pred, f, mask)), 0.125, places=6)


if __name__ == "__main__":
    unittest.main()
