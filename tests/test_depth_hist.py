"""扩展深度直方图（P3 埋点）单元测试。

1. 纯函数：hist_add / hist_merge / hist_summary 的计数、分位与重放前向数；
2. （需 CUDA）随机初始化模型上，串行 ``play_one_game`` 与批量 ``BatchedArenaGame`` 的
   直方图逐位相等、PGN 一致，且驱动器前向数与直方图推出的前向数吻合。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from stateseq.depth_hist import hist_add, hist_merge, hist_summary  # noqa: E402

try:
    import torch  # noqa: F401
    import chess  # noqa: F401

    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover - 本机（Windows）无 torch
    _HAS_TORCH = False
    _HAS_CUDA = False


class TestHistPure(unittest.TestCase):
    def test_add_grows_and_counts(self):
        h: list[int] = []
        for d in (1, 1, 3, 2, 1):
            hist_add(h, d)
        self.assertEqual(h, [0, 3, 1, 1])

    def test_merge(self):
        self.assertEqual(hist_merge([0, 2], [0, 1, 4]), [0, 3, 4])
        self.assertEqual(hist_merge([0, 2, 5], None), [0, 2, 5])
        a = [0, 1]
        hist_merge(a, [0, 1])
        self.assertEqual(a, [0, 1], "merge 不得原地修改入参")

    def test_summary(self):
        h = [0] * 11
        h[1], h[2], h[10] = 50, 40, 10
        s = hist_summary(h)
        self.assertEqual(s["expansions"], 100)
        self.assertAlmostEqual(s["mean"], (50 + 80 + 100) / 100)
        self.assertEqual((s["p50"], s["p90"], s["p99"], s["max"]), (1, 2, 10, 10))
        self.assertEqual(s["replay_forwards"], 40 * 1 + 10 * 9)

    def test_summary_empty(self):
        s = hist_summary([])
        self.assertEqual((s["expansions"], s["mean"], s["max"]), (0, 0.0, 0))


def _load_tool():
    path = os.path.join(HERE, "tools", "ssm_gumbel_arena.py")
    spec = importlib.util.spec_from_file_location("ssm_gumbel_arena_hist", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(_HAS_CUDA, "需要 CUDA（mamba/causal_conv1d 步进核只有 GPU 版）")
class TestArenaHist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = _load_tool()

    def _model(self, seed: int):
        from stateseq.model import SeqModel

        torch.manual_seed(seed)
        m = self.tool.ArenaModel.__new__(self.tool.ArenaModel)
        m.ckpt_path = f"rand{seed}"
        m.device = "cuda"
        m.seq = SeqModel(dropout=0.0).to("cuda").eval()
        m._tc_cache, m._elo_cache = {}, {}
        m.c_visit, m.c_scale = 50.0, 0.1
        return m

    def _cfg(self):
        cfg = lambda: None  # noqa: E731
        cfg.n_sims, cfg.m0, cfg.max_plies, cfg.c_visit, cfg.c_scale = 12, 4, 3, 50.0, 0.1
        return cfg

    def test_serial_equals_batched(self):
        a, b = self._model(1), self._model(2)
        cfg = self._cfg()
        opening = "e4 e5"
        gd_s = self.tool.play_one_game(a, b, cfg, opening_san=opening, opening_id=0, seed=7)
        game = self.tool.BatchedArenaGame(0, 0, [a, b], True, cfg, opening, 0, seed=7)
        args = self.tool.argparse.Namespace(sprt=False, games=1, sprt_alpha=0.05, sprt_beta=0.05,
                                            sprt_min_games=64)
        driver = self.tool.BatchedArenaDriver([game], concurrency=1, args=args,
                                              internal_sprt=False)
        driver.run()
        gd_b = driver.results[0]

        self.assertEqual(gd_s["pgn"], gd_b["pgn"])
        self.assertEqual(gd_s["expand_depth_hist"], gd_b["expand_depth_hist"])
        hist = gd_b["expand_depth_hist"]
        s = hist_summary(hist)
        n_plies = gd_b["n_plies"]
        self.assertEqual(n_plies, cfg.max_plies)
        self.assertLessEqual(s["expansions"], cfg.n_sims * n_plies)
        self.assertGreater(s["max"], 1, "12 sims / m0=4 应当展开到第 2 层以下")
        # 前向数 = 每 ply（含开局）双方各 1 次推进 + 重放 + 非终局评估
        advances = 2 * (2 + n_plies)
        lo = advances + s["replay_forwards"]
        self.assertLessEqual(lo, driver.n_forwards)
        self.assertLessEqual(driver.n_forwards, lo + s["expansions"])

        agg = self.tool._aggregate_results([gd_b, dict(gd_s, arena_result=1)], 1,
                                           self.tool.argparse.Namespace(
                                               ckpt_a="a", ckpt_b="b", c_visit=50.0,
                                               c_scale_a=0.1, c_scale_b=0.1, n_sims=12, m0=4))
        self.assertEqual(agg["expand_depth"]["hist"], hist_merge(hist, hist))


if __name__ == "__main__":
    unittest.main()
