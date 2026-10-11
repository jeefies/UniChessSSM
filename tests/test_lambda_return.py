"""λ-return 价值目标（D2）：向后递推的正确性与边界。

不变式：
1. λ=1 ⇒ g ≡ z（= 现 MC 基线，可逐位对拍）；
2. λ=0 ⇒ g[t] = q̂[t+1]、g[last] = 锚点（纯一步自举，明确要避免的端点）；
3. 一般 λ 与 O(T²) 朴素参考实现一致；
4. 整局无 q̂（老分片）⇒ 任意 λ 都退化为纯 MC；
5. 截断局 / 被 t_max 截断（anchor_last=True）⇒ 锚点用 q̂[last] 而非 z；
6.  padding 不污染有效位；同一批次里"有/无 q̂"的局互不影响。
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

from SSM.model.losses import lambda_return


def _naive(q, qv, z, lam, valid, anchor_last):
    """O(T²) 参考实现：直接按定义求和，便于对照向量化递推。"""
    b, t = q.shape
    out = torch.zeros_like(q)
    for i in range(b):
        idx = [k for k in range(t) if bool(valid[i, k])]
        if not idx:
            continue
        last = idx[-1]
        anchor = float(q[i, last]) if bool(anchor_last[i]) and bool(qv[i, last]) else float(z[i])
        g = [0.0] * t
        g[last] = anchor
        for k in range(last - 1, -1, -1):
            if bool(qv[i, k + 1]):
                q_eff = float(q[i, k + 1])
            else:
                q_eff = g[k + 1]
            g[k] = (1.0 - lam) * q_eff + lam * g[k + 1]
        for k in idx:
            out[i, k] = g[k]
    return out


class LambdaReturnTest(unittest.TestCase):
    def setUp(self):
        # 白方绝对视角：q 递增，z = +1（白胜）
        self.q = torch.tensor([[0.10, 0.20, 0.30, 0.40, 0.50]], dtype=torch.float32)
        self.qv = torch.ones_like(self.q, dtype=torch.bool)
        self.z = torch.tensor([1.0])
        self.valid = torch.ones_like(self.q, dtype=torch.bool)

    def test_lambda_one_equals_mc(self):
        g = lambda_return(self.q, self.qv, self.z, 1.0, self.valid)
        for t in range(5):
            self.assertAlmostEqual(float(g[0, t]), 1.0, places=6,
                                   msg=f"λ=1 时 g[{t}] 应等于终局 z")

    def test_lambda_zero_is_one_step_bootstrap(self):
        g = lambda_return(self.q, self.qv, self.z, 0.0, self.valid)
        for t in range(4):
            self.assertAlmostEqual(float(g[0, t]), float(self.q[0, t + 1]), places=6)
        self.assertAlmostEqual(float(g[0, 4]), 1.0, places=6)   # 锚点 = z

    def test_matches_naive_reference(self):
        for lam in (0.2, 0.5, 0.8):
            g = lambda_return(self.q, self.qv, self.z, lam, self.valid)
            ref = _naive(self.q, self.qv, self.z, lam, self.valid, torch.zeros(1, dtype=torch.bool))
            self.assertTrue(torch.allclose(g, ref, atol=1e-6),
                            f"λ={lam} 与参考实现不一致:\n{g}\n{ref}")

    def test_no_qsearch_degrades_to_mc(self):
        """老分片（整局无 q̂）：任意 λ 都必须是纯 MC。"""
        qv = torch.zeros_like(self.q, dtype=torch.bool)
        for lam in (0.0, 0.3, 0.5, 0.9):
            g = lambda_return(self.q, qv, self.z, lam, self.valid)
            for t in range(5):
                self.assertAlmostEqual(float(g[0, t]), 1.0, places=6,
                                       msg=f"λ={lam} 无 q̂ 时应退化为 MC")

    def test_partial_qsearch_uses_later_g(self):
        """中间 ply 缺 q̂：该步不引入自举，权重自动归到 g[t+1]。"""
        qv = torch.tensor([[True, False, True, True, True]])
        g = lambda_return(self.q, qv, self.z, 0.5, self.valid)
        ref = _naive(self.q, qv, self.z, 0.5, self.valid, torch.zeros(1, dtype=torch.bool))
        self.assertTrue(torch.allclose(g, ref, atol=1e-6))

    def test_truncated_anchor_uses_last_q(self):
        """截断局：z=0 是约定值，锚点必须改用 q̂[last]。"""
        anchor = torch.tensor([True])
        g = lambda_return(self.q, self.qv, self.z, 0.5, self.valid, anchor)
        self.assertAlmostEqual(float(g[0, 4]), 0.50, places=6)
        ref = _naive(self.q, self.qv, self.z, 0.5, self.valid, anchor)
        self.assertTrue(torch.allclose(g, ref, atol=1e-6))

    def test_padding_does_not_contaminate(self):
        q = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0],
                          [0.4, 0.5, 0.0, 0.0, 0.0]])
        qv = torch.ones_like(q, dtype=torch.bool)
        valid = torch.tensor([[True, True, True, False, False],
                              [True, True, False, False, False]])
        z = torch.tensor([1.0, -1.0])
        g = lambda_return(q, qv, z, 0.5, valid)
        ref = _naive(q, qv, z, 0.5, valid, torch.zeros(2, dtype=torch.bool))
        self.assertTrue(torch.allclose(g, ref, atol=1e-6))
        # 短行的有效位不能受长行/填充位影响
        self.assertAlmostEqual(float(g[1, 1]), -1.0, places=6)

    def test_mixed_rows(self):
        """同一批里一行有 q̂、一行没有：互不影响。"""
        q = torch.tensor([[0.1, 0.2, 0.3], [9.9, 9.9, 9.9]])
        qv = torch.tensor([[True, True, True], [False, False, False]])
        z = torch.tensor([1.0, -1.0])
        valid = torch.ones_like(q, dtype=torch.bool)
        g = lambda_return(q, qv, z, 0.5, valid)
        self.assertAlmostEqual(float(g[1, 0]), -1.0, places=6)
        self.assertAlmostEqual(float(g[1, 2]), -1.0, places=6)
        ref = _naive(q, qv, z, 0.5, valid, torch.zeros(2, dtype=torch.bool))
        self.assertTrue(torch.allclose(g, ref, atol=1e-6))

    def test_single_ply(self):
        q = torch.tensor([[0.7]])
        g = lambda_return(q, torch.ones_like(q, dtype=torch.bool), torch.tensor([1.0]), 0.5,
                          torch.ones_like(q, dtype=torch.bool))
        self.assertAlmostEqual(float(g[0, 0]), 1.0, places=6)


# ------------------------- 常数权重的终局锚（价值尺度防坍缩） -------------------------

class AnchorWeightTest(unittest.TestCase):
    """β>0 时终局真值在每个位置都有固定话语权，杜绝纯自举导致的尺度代际坍缩。"""

    def setUp(self):
        # 长序列：中位位置离终局 ~90 步，λ^90 早已衰减到 0
        self.T = 181
        self.q = (torch.rand(1, self.T) * 2 - 1) * 0.5          # |q| ≤ 0.5
        self.qv = torch.ones_like(self.q, dtype=torch.bool)
        self.z = torch.tensor([1.0])
        self.valid = torch.ones_like(self.q, dtype=torch.bool)

    def test_beta_zero_is_old_behaviour(self):
        """β=0 必须与旧行为逐位一致（递推自带的 λ^k 锚）。"""
        g0 = lambda_return(self.q, self.qv, self.z, 0.5, self.valid, None, 0.0)
        g_none = lambda_return(self.q, self.qv, self.z, 0.5, self.valid)
        self.assertTrue(torch.equal(g0, g_none), "β=0 与不传 β 不一致")

    def test_beta_one_is_pure_mc(self):
        g = lambda_return(self.q, self.qv, self.z, 0.5, self.valid, None, 1.0)
        for t in range(self.T):
            self.assertAlmostEqual(float(g[0, t]), 1.0, places=6, msg=f"β=1 时 g[{t}] 应等于 z")

    def test_anchor_weight_is_constant_across_distance(self):
        """关键不变式：终局锚的权重不随剩余步数衰减（旧实现在远端位置恒等于 0）。"""
        beta = 0.5
        g_beta = lambda_return(self.q, self.qv, self.z, 0.5, self.valid, None, beta)
        g_plain = lambda_return(self.q, self.qv, self.z, 0.5, self.valid, None, 0.0)
        # g_beta = (1-β)·g_plain + β·z  ⇒ 残差应恒为 β·z
        resid = g_beta - (1.0 - beta) * g_plain
        expect = torch.full_like(resid, beta * 1.0)
        self.assertTrue(torch.allclose(resid, expect, atol=1e-6),
                        f"锚权重不是常数：残差范围 [{resid.min():.6f}, {resid.max():.6f}]")
        # 远端位置（递推锚已衰减到 0）也必须拿到 β·z
        far = self.T - 2
        self.assertAlmostEqual(float(resid[0, far]), beta, places=6)

    def test_target_scale_does_not_collapse(self):
        """坍缩回归：目标里必须保留终局真值的尺度，而不是被 q̂ 的尺度拖着走。"""
        beta = 0.5
        g = lambda_return(self.q, self.qv, self.z, 0.5, self.valid, None, beta)
        # q̂ 全压到 0.05（模拟已坍缩的价值头），目标仍应离 z 有 beta 的量级
        q_flat = torch.full_like(self.q, 0.05)
        g_flat = lambda_return(q_flat, self.qv, self.z, 0.5, self.valid, None, beta)
        self.assertGreater(float(g_flat[0, self.T // 2]), 0.4,
                           "q̂ 被压平时目标也跟着塌——终局锚没起作用")
        g_flat_old = lambda_return(q_flat, self.qv, self.z, 0.5, self.valid, None, 0.0)
        self.assertLess(abs(float(g_flat_old[0, self.T // 2])), 0.1,
                        "β=0 时目标应几乎完全跟随 q̂（这正是坍缩的机制）")

    def test_padding_stays_zero(self):
        valid = torch.ones_like(self.q, dtype=torch.bool)
        valid[0, self.T // 2:] = False
        g = lambda_return(self.q, self.qv, self.z, 0.5, valid, None, 0.5)
        self.assertEqual(float(g[0, self.T // 2:].abs().max()), 0.0,
                         "β 混合把 padding 位也填上了")


if __name__ == "__main__":
    unittest.main()
