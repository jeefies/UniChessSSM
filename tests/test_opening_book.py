"""开局注入与 π′ 跨局共享（P1-1 / P1-2）+ book_mask 降权（P1-3）单元测试。

覆盖：
1. ``book_pipol_rng``：同 (seed, opening_idx, ply) 逐位一致、跨 ply/开局不同；
2. ``build_book_mask``：book 段掩码（含超长 book_counts、局长约束、旧分片 flags=0）；
3. 生成器配置默认值（book_plies 6 等）。
开局文件解析（裁至 book_plies、非法线丢弃、不去重）在 kit：``Kit/tests/test_selfplay.py``；
book ply 的 π′ 跨局共享与原生成器逐字节一致：``tests/test_kit_selfplay.py``。
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


import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np


try:  # 工具模块与 dataset_selfplay 均依赖 torch；远端全量单测环境具备
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - 本机（Windows）无 torch
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "需要 torch（远端全量单测环境）")
class TestBookPipolRng(unittest.TestCase):
    """π′ 开局段噪声只取决于 (seed, opening_idx, ply)——跨局共享的前提。"""

    @classmethod
    def setUpClass(cls):
        from SSM.kit import book_pipol_rng

        cls.rng = staticmethod(book_pipol_rng)

    def test_deterministic_same_key(self):
        a = self.rng(42, 3, 1)
        b = self.rng(42, 3, 1)
        self.assertTrue(np.array_equal(a.random(16), b.random(16)))

    def test_differs_across_ply_and_opening(self):
        base = self.rng(42, 3, 1).random(16)
        self.assertFalse(np.array_equal(base, self.rng(42, 3, 2).random(16)))
        self.assertFalse(np.array_equal(base, self.rng(42, 4, 1).random(16)))
        self.assertFalse(np.array_equal(base, self.rng(43, 3, 1).random(16)))

    def test_stable_across_instances(self):
        """跨进程/跨次调用一致（SeedSequence 熵确定性）——跨局共享的前提。"""
        first = self.rng(7, 11, 4).random(8).copy()
        for _ in range(3):
            self.assertTrue(np.array_equal(first, self.rng(7, 11, 4).random(8)))


@unittest.skipUnless(_HAS_TORCH, "需要 torch（远端全量单测环境）")
class TestBuildBookMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from SSM.dataset.dataset_selfplay import build_book_mask

        cls.fn = staticmethod(build_book_mask)

    def test_basic(self):
        m = self.fn([10, 8], [6, 0], t=10)
        self.assertEqual(m.shape, (2, 10))
        self.assertTrue(m[0, :6].all())
        self.assertFalse(m[0, 6:].any())
        self.assertFalse(m[1].any())  # flags=0（旧分片）⇒ 无 book 段

    def test_book_count_clamped_by_length(self):
        m = self.fn([4, 10], [6, 3], t=10)
        self.assertEqual(int(m[0].sum()), 4)   # 局长 4 < book 6 ⇒ 只掩有效段
        self.assertEqual(int(m[1].sum()), 3)

    def test_book_count_exceeds_t(self):
        m = self.fn([10], [20], t=10)
        self.assertTrue(m.all())  # t_max 截断内全部为 book 段


@unittest.skipUnless(_HAS_TORCH, "需要 torch（远端全量单测环境）")
class TestSelfPlayConfigDefaults(unittest.TestCase):
    """自对弈配置默认值（生成器在 kit 的 selfplay pipeline）。"""

    @classmethod
    def setUpClass(cls):
        from Kit.pipelines.selfplay import SelfPlayConfig

        cls.SelfPlayConfig = SelfPlayConfig

    def test_config_defaults(self):
        cfg = self.SelfPlayConfig(games=2)
        self.assertEqual(cfg.book_plies, 6)
        self.assertEqual(cfg.max_plies, 300)
        self.assertEqual(cfg.first_game, 0)
        self.assertIsNone(cfg.openings)

    def test_game_index_is_global(self):
        """多进程分片的局序号是全局的（first_game 起），各进程区间不相交。"""
        cfg = self.SelfPlayConfig(games=10, first_game=5)
        self.assertEqual([g for g in range(cfg.first_game, cfg.first_game + cfg.games)],
                         list(range(5, 15)))


@unittest.skipUnless(_HAS_TORCH, "需要 torch（远端全量单测环境）")
class TestOpeningLossWeightPlumbing(unittest.TestCase):
    """P1-3：policy_weights 经 forward_train 正确传入 policy 软 CE。"""

    def _make_batch(self, b: int, t: int, n_legal: int = 5):
        import torch

        from SSM.actions import NUM_ACTIONS
        from SSM.features import FEATURE_DIM
        from SSM.model import TrainBatch

        torch.manual_seed(0)
        feats = torch.randn(b, t, FEATURE_DIM) * 0.1
        legal = torch.zeros(b, t, NUM_ACTIONS, dtype=torch.bool)
        # 每步固定取前 n_legal 个动作作合法着（确定性，便于构造软目标）
        for i in range(n_legal):
            legal[:, :, i] = True
        actions = torch.zeros(b, t, dtype=torch.int64)
        results = torch.ones(b, t, dtype=torch.int64)  # 全和棋
        moves_left = torch.arange(t, dtype=torch.float32).unsqueeze(0).expand(b, t).contiguous()
        elo_w = torch.ones(b)
        tc = torch.zeros(b, dtype=torch.int64)
        elo_std = torch.zeros(b)
        color = torch.ones(b, t, dtype=torch.int64)
        batch = TrainBatch(feats, actions, legal, results, moves_left, elo_w, tc, elo_std, color)
        # 均匀软目标（支持集 = 合法着）
        soft = torch.zeros(b, t, NUM_ACTIONS)
        soft[:, :, :n_legal] = 1.0 / n_legal
        return batch, soft

    def _cpu_model(self):
        """构造可跑 CPU 的模型：Mamba trunk（causal_conv1d）需 CUDA，替换为零张量假实现。

        只关心 policy 损失权重管线，trunk 输出恒零不影响该验证。
        """
        import torch

        from SSM.model import SeqModel

        model = SeqModel(dropout=0.0).eval()
        model.trunk = lambda x: torch.zeros_like(x)
        return model

    def test_weights_reach_policy_soft_loss(self):
        import torch
        from unittest.mock import patch

        from SSM.model import losses

        model = self._cpu_model()
        batch, soft = self._make_batch(2, 8)
        valid = torch.ones(2, 8, dtype=torch.bool)
        book_mask = torch.zeros(2, 8, dtype=torch.bool)
        book_mask[:, :3] = True  # 前 3 ply 为 book 段
        pw = torch.where(book_mask, torch.full((2, 8), 0.25), torch.ones(2, 8))

        captured = {}
        real = losses.policy_soft_loss

        def spy(logits, target_probs, weights, pos_mask=None, eps=1e-8):
            captured["weights"] = weights.detach().clone()
            return real(logits, target_probs, weights, pos_mask, eps)

        with patch.object(losses, "policy_soft_loss", side_effect=spy):
            with torch.no_grad():
                total, metrics = model.forward_train(
                    batch, losses.LossWeights(), step=0, total_steps=10,
                    valid_mask=valid, policy_soft_target=soft, policy_weights=pw)

        w = captured["weights"]
        self.assertEqual(tuple(w.shape), (2, 8))
        self.assertTrue(torch.allclose(w[:, :3], torch.full((2, 3), 0.25)))
        self.assertTrue(torch.allclose(w[:, 3:], torch.ones(2, 5)))
        self.assertIn("loss_policy", metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss_policy"])))

    def test_no_weights_means_all_ones(self):
        import torch
        from unittest.mock import patch

        from SSM.model import losses

        model = self._cpu_model()
        batch, soft = self._make_batch(1, 6)
        valid = torch.ones(1, 6, dtype=torch.bool)
        captured = {}
        real = losses.policy_soft_loss

        def spy(logits, target_probs, weights, pos_mask=None, eps=1e-8):
            captured["weights"] = weights.detach().clone()
            return real(logits, target_probs, weights, pos_mask, eps)

        with patch.object(losses, "policy_soft_loss", side_effect=spy):
            with torch.no_grad():
                model.forward_train(batch, losses.LossWeights(), step=0, total_steps=10,
                                    valid_mask=valid, policy_soft_target=soft)
        self.assertTrue(torch.allclose(captured["weights"], torch.ones(1, 6)))


if __name__ == "__main__":
    unittest.main()
