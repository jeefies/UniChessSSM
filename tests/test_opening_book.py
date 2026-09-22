"""开局注入与 π′ 跨局共享（P1-1 / P1-2）+ book_mask 降权（P1-3）单元测试。

覆盖：
1. ``load_openings``：裁至 book_plies、非法线丢弃、空/缺失路径返回 []；
2. ``book_pipol_rng``：同 (seed, opening_idx, ply) 逐位一致、跨 ply/开局不同；
3. ``build_book_mask``：book 段掩码（含超长 book_counts、局长约束、旧分片 flags=0）；
4. 生成器 book 阶段不变量：``SelfPlayConfig.book_plies`` 默认 6、
   ``GameState`` 新字段默认值（轻量，不建模型）。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

_TOOL_PATH = os.path.join(HERE, "tools", "ssm_gumbel_selfplay.py")

try:  # 工具模块与 dataset_selfplay 均依赖 torch；远端全量单测环境具备
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - 本机（Windows）无 torch
    _HAS_TORCH = False


def _load_tool():
    spec = importlib.util.spec_from_file_location("ssm_gumbel_selfplay_under_test", _TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    # dataclass 处理需要通过 sys.modules[cls.__module__] 反查命名空间，必须先注册
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(_HAS_TORCH, "需要 torch（远端全量单测环境）")
class TestLoadOpenings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = _load_tool()

    def _write(self, text: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        self.addCleanup(os.remove, path)
        return path

    def test_slice_to_book_plies(self):
        path = self._write("e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6\n"
                           "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7\n")
        ops = self.tool.load_openings(path, book_plies=6)
        self.assertEqual(len(ops), 2)
        for line in ops:
            self.assertEqual(len(line), 6)
        self.assertEqual(ops[0], ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"])
        self.assertEqual(ops[1], ["d4", "d5", "c4", "e6", "Nc3", "Nf6"])

    def test_short_line_kept_as_is(self):
        path = self._write("e4 e5 Nf3\n")
        ops = self.tool.load_openings(path, book_plies=6)
        self.assertEqual(ops, [["e4", "e5", "Nf3"]])

    def test_illegal_line_dropped(self):
        path = self._write("e4 e5 Nf3\n"          # 合法
                           "e4 e5 Nf6 Ke2\n"       # Nf6 后 Ke2 不合法（黑方未走）
                           "garbage\n")            # 完全非法
        ops = self.tool.load_openings(path, book_plies=6)
        self.assertEqual(ops, [["e4", "e5", "Nf3"]])

    def test_missing_or_empty_path(self):
        self.assertEqual(self.tool.load_openings("", book_plies=6), [])
        self.assertEqual(self.tool.load_openings("/nonexistent/x.txt", book_plies=6), [])

    def test_blank_lines_ignored(self):
        path = self._write("\n\ne4 e5\n\n   \n")
        ops = self.tool.load_openings(path, book_plies=6)
        self.assertEqual(ops, [["e4", "e5"]])


@unittest.skipUnless(_HAS_TORCH, "需要 torch（远端全量单测环境）")
class TestBookPipolRng(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = _load_tool()

    def test_deterministic_same_key(self):
        a = self.tool.book_pipol_rng(42, 3, 1)
        b = self.tool.book_pipol_rng(42, 3, 1)
        self.assertTrue(np.array_equal(a.random(16), b.random(16)))

    def test_differs_across_ply_and_opening(self):
        base = self.tool.book_pipol_rng(42, 3, 1).random(16)
        self.assertFalse(np.array_equal(base, self.tool.book_pipol_rng(42, 3, 2).random(16)))
        self.assertFalse(np.array_equal(base, self.tool.book_pipol_rng(42, 4, 1).random(16)))
        self.assertFalse(np.array_equal(base, self.tool.book_pipol_rng(43, 3, 1).random(16)))

    def test_stable_across_instances(self):
        """跨进程/跨次调用一致（SeedSequence 熵确定性）——跨局共享的前提。"""
        first = self.tool.book_pipol_rng(7, 11, 4).random(8).copy()
        for _ in range(3):
            self.assertTrue(np.array_equal(first, self.tool.book_pipol_rng(7, 11, 4).random(8)))


@unittest.skipUnless(_HAS_TORCH, "需要 torch（远端全量单测环境）")
class TestBuildBookMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from stateseq.data.dataset_selfplay import build_book_mask

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
class TestGameStateBookFields(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = _load_tool()

    def test_config_default_book_plies(self):
        cfg = self.tool.SelfPlayConfig(ckpt="x", out_dir="y", tag="z")
        self.assertEqual(cfg.book_plies, 6)
        self.assertEqual(cfg.n_sims, 256)
        self.assertEqual(cfg.gumbel_g, 1.0)

    def test_game_state_has_memo_fields(self):
        import chess

        class _FakeModel:
            def initial_cache(self, batch_size: int = 1):
                return None

        cfg = self.tool.SelfPlayConfig(ckpt="x", out_dir="y", tag="z", book_plies=6)
        gs = self.tool.GameState(0, _FakeModel(), cfg, np.random.SeedSequence(0).spawn(1)[0],
                                 opening_moves=["e4", "e5"], opening_idx=3,
                                 pipol_memo={("k",): "v"})
        self.assertEqual(gs.n_book_plies, 0)
        self.assertEqual(gs.opening_idx, 3)
        self.assertEqual(gs.pipol_memo, {("k",): "v"})
        self.assertEqual(gs.opening_moves, ["e4", "e5"])
        self.assertIsInstance(gs.board, chess.Board)


@unittest.skipUnless(_HAS_TORCH, "需要 torch（远端全量单测环境）")
class TestOpeningLossWeightPlumbing(unittest.TestCase):
    """P1-3：policy_weights 经 forward_train 正确传入 policy 软 CE。"""

    def _make_batch(self, b: int, t: int, n_legal: int = 5):
        import torch

        from stateseq.actions import NUM_ACTIONS
        from stateseq.features import FEATURE_DIM
        from stateseq.model import TrainBatch

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

        from stateseq.model import SeqModel

        model = SeqModel(dropout=0.0).eval()
        model.trunk = lambda x: torch.zeros_like(x)
        return model

    def test_weights_reach_policy_soft_loss(self):
        import torch
        from unittest.mock import patch

        from stateseq import losses

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

        from stateseq import losses

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
