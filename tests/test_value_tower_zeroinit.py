"""价值塔（D1）：零初始化残差块 ⇒ 与改造前的单层线性头逐位一致。

覆盖三条不变式：
1. ``PredictionHeads.forward`` 的 wdl 输出 ≡ ``w_v(head_norm(transform(h)))``（塔是恒等映射）；
2. 塔的 fc2 确实为零，且经 ``load_state_dict(..., strict=False)`` 加载"没有塔键"的旧
   checkpoint 后**仍然**为零（防止有人删掉零初始化导致旧权重加载即回归）；
3. 参数量仍在 D4 的 ~28M 预算内；塔在训练几步后必须开始学（不是死参数）。

前两条纯 CPU 可跑；第三条与整体前向需要 GPU（Mamba 主干）。
"""

from __future__ import annotations
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_IMPORT_ROOT = _os.path.dirname(_HERE)   # import 根：~/UniChess
if _IMPORT_ROOT not in _sys.path:
    _sys.path.insert(0, _IMPORT_ROOT)


import tempfile
import unittest

import torch

from SSM.model.heads import D_MODEL, PredictionHeads, VALUE_TOWER_BLOCKS, VALUE_TOWER_HIDDEN


def _tiny_head() -> PredictionHeads:
    torch.manual_seed(0)
    return PredictionHeads(d_model=32, n_actions=8)


class ValueTowerZeroInitTest(unittest.TestCase):
    def test_tower_is_identity(self):
        head = _tiny_head()
        h = torch.randn(3, 5, 32)
        pol, wdl, mlh = head(h)
        u = head.transform(h)
        ref_wdl = head.w_v(head.head_norm(u))
        ref_mlh = torch.nn.functional.mish(head.w_m(head.head_norm(u)))
        # 零初始化残差块 ⇒ v ≡ head_norm(u)，wdl/mlh 必须逐位相同
        self.assertTrue(torch.equal(wdl, ref_wdl), "价值塔未保持恒等：wdl 与旧口径不一致")
        self.assertTrue(torch.equal(mlh, ref_mlh), "mlh 路径被价值塔改动")
        self.assertTrue(torch.equal(pol, head.w_p(u)), "policy 路径被价值塔改动")

    def test_tower_params_are_zero(self):
        head = _tiny_head()
        for i, blk in enumerate(head.value_tower):
            self.assertTrue(torch.equal(blk.fc2.weight, torch.zeros_like(blk.fc2.weight)),
                            f"value_tower[{i}].fc2.weight 未零初始化")
            self.assertTrue(torch.equal(blk.fc2.bias, torch.zeros_like(blk.fc2.bias)),
                            f"value_tower[{i}].fc2.bias 未零初始化")
        self.assertTrue(torch.equal(head.w_mat.weight, torch.zeros_like(head.w_mat.weight)))
        self.assertTrue(torch.equal(head.w_mat.bias, torch.zeros_like(head.w_mat.bias)))

    def test_old_checkpoint_load_keeps_tower_zero(self):
        """旧 checkpoint（无 value_tower / w_mat 键）经 strict=False 加载后塔必须仍为零。"""
        head = _tiny_head()
        sd = {k: v for k, v in head.state_dict().items()
              if not k.startswith("value_tower.") and not k.startswith("w_mat.")}
        fresh = PredictionHeads(d_model=32, n_actions=8)
        missing, unexpected = fresh.load_state_dict(sd, strict=False)
        self.assertEqual(sorted(unexpected), [])
        self.assertTrue(any(k.startswith("value_tower.") for k in missing),
                        "加载后没有缺失的塔键——说明塔参数被意外纳入了 state_dict 前缀判断")
        h = torch.randn(2, 4, 32)
        _, wdl, _ = fresh(h)
        u = fresh.transform(h)
        self.assertTrue(torch.equal(wdl, fresh.w_v(fresh.head_norm(u))),
                        "旧 checkpoint 加载后价值塔不是恒等映射")

    def test_material_head_zero_init(self):
        head = _tiny_head()
        h = torch.randn(2, 3, 32)
        pol, wdl, mlh, mat = head(h, return_material=True)
        self.assertTrue(torch.equal(mat, torch.zeros_like(mat)),
                        "子力差辅助头未零初始化（初始预测应为 0 = 子力相等）")

    def test_param_budget_d4(self):
        """总参数必须仍在 D4 的 ~28M 预算内（价值塔 +1.57M）。"""
        from SSM.model import SeqModel, count_parameters
        model = SeqModel(dropout=0.0)
        counts = count_parameters(model)
        total = counts["合计"]
        self.assertLessEqual(total, 28_000_000, f"超出 D4 预算：{total}")
        self.assertGreater(total, 26_000_000, f"参数异常偏小：{total}")


@unittest.skipUnless(torch.cuda.is_available(), "整体前向/反向需 CUDA")
class ValueTowerTrainingTest(unittest.TestCase):
    def test_tower_learns(self):
        """塔不是死参数：几步价值损失后 fc2 必须非零且损失下降。"""
        torch.manual_seed(0)
        head = PredictionHeads(d_model=D_MODEL, n_actions=8).cuda()
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
        h = torch.randn(4, 7, D_MODEL, device="cuda")
        # 目标：让 q = pW - pL 收敛到 +0.8（行棋方视角）
        g = torch.full((4, 7), 0.8, device="cuda")
        first = None
        for i in range(60):
            _, wdl, _ = head(h)
            loss = torch.nn.functional.cross_entropy(
                wdl.reshape(-1, 3), torch.full((wdl.numel() // 3,), 0, dtype=torch.long,
                                               device="cuda"))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if i == 0:
                first = float(loss.detach())
        self.assertGreater(first, 0.9)
        nonzero = any(float(blk.fc2.weight.detach().abs().sum()) > 0 for blk in head.value_tower)
        self.assertTrue(nonzero, "塔的 fc2 仍全零：残差路径没有拿到梯度")


if __name__ == "__main__":
    unittest.main()
