"""A#6 补充（集成级）：train/stage_b2.py 实际使用的 forward_train(policy_soft_target=...)
路径——含 padding、终局无合法着、bf16 autocast——前向+反向全部有限。

tests/test_gumbel.py::SoftCESafetyTest 只验证 stateseq.gumbel.pi_prime 这一纯数学函数；
本测试覆盖的是训练器真正调用的集成路径（model.forward_train 新增的软目标分支 +
policy_soft_loss + bf16 autocast），防止两者实现不同步。
需要 CUDA（R 主干前向/反向仅 GPU 可行）；无 GPU 时跳过。
"""

from __future__ import annotations

import math
import unittest

import torch

from stateseq.conditions import TimeControlBucket


@unittest.skipUnless(torch.cuda.is_available(), "R 主干前向/反向需 CUDA")
class SoftPolicyIntegrationSafetyTest(unittest.TestCase):
    def test_forward_backward_finite_with_soft_target(self):
        from stateseq import losses
        from stateseq.actions import NUM_ACTIONS
        from stateseq.features import FEATURE_DIM
        from stateseq.model import SeqModel, TrainBatch

        torch.manual_seed(0)
        device = "cuda"
        b, t = 2, 6
        model = SeqModel(dropout=0.0).float().to(device)

        features = torch.randn(b, t, FEATURE_DIM, device=device)
        legal = torch.zeros(b, t, NUM_ACTIONS, dtype=torch.bool, device=device)
        legal[:, :, :10] = True
        actions = torch.randint(0, 10, (b, t), device=device)
        results = torch.randint(0, 3, (b, t), device=device)
        moves_left = torch.zeros(b, t, device=device)
        color = torch.zeros(b, t, dtype=torch.long, device=device)
        elo_w = torch.ones(b, device=device)
        tc = torch.full((b,), int(TimeControlBucket.RAPID), dtype=torch.long, device=device)
        elo = torch.zeros(b, device=device)
        batch = TrainBatch(features, actions, legal, results, moves_left, elo_w, tc, elo, color)

        # π′ 软目标：合法位给概率；局 0 末步视为 padding（截断），局 1 末步视为终局无合法着。
        target = torch.zeros(b, t, NUM_ACTIONS, device=device)
        target[:, :, :10] = torch.softmax(torch.randn(b, t, 10, device=device), dim=-1)
        valid = torch.ones(b, t, dtype=torch.bool, device=device)
        valid[0, -1] = False
        legal[1, -1] = False
        target[1, -1] = 0.0

        # 局 1 视为封顶截断局：mlh 整局剔除（§2.5）。
        mlh_valid = valid.clone()
        mlh_valid[1, :] = False

        weights = losses.LossWeights(w_v=1.0)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.1)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total, metrics = model.forward_train(batch, weights, step=0, total_steps=100,
                                                 valid_mask=valid, policy_soft_target=target,
                                                 mlh_valid_mask=mlh_valid)
        total.backward()

        for k, v in metrics.items():
            self.assertTrue(math.isfinite(v), f"{k}={v} 非有限")
        n_checked = 0
        for name, p in model.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(), f"{name} 梯度含 NaN/Inf")
                n_checked += 1
        self.assertGreater(n_checked, 0, "没有任何参数收到梯度，测试未覆盖到目标路径")


if __name__ == "__main__":
    unittest.main()
