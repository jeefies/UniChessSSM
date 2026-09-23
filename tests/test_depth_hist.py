"""扩展深度直方图（P3 埋点）单元测试。

1. 纯函数：hist_add / hist_merge / hist_summary 的计数、分位与重放前向数；
2. arena 的 kit 记录 → games.jsonl 转换与聚合（逐局直方图合并进 arena.json）。
端到端（真实搜索产出直方图、多进程转发、续跑读回）见 ``tests/test_arena_kit.py``；
自对弈直方图与原生成器逐位一致见 ``tests/test_kit_selfplay.py``。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
KIT_ROOT = os.environ.get("UNICHESS_KIT_ROOT", os.path.join(os.path.dirname(HERE), "Kit"))
if os.path.isdir(KIT_ROOT) and KIT_ROOT not in sys.path:
    sys.path.append(KIT_ROOT)  # 追加而非前插：Kit 的 tests 包不得遮蔽本仓库的 tests

from stateseq.depth_hist import hist_add, hist_merge, hist_summary  # noqa: E402

try:
    import chess  # noqa: F401
    import unichess_kit  # noqa: F401

    _HAS_KIT = True
except ImportError:  # pragma: no cover - 缺 python-chess 或兄弟仓库 Kit
    _HAS_KIT = False


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


def _rec(game, white, a_score, moves, termination):
    return {"type": "game", "game": game, "pair": game // 2, "white": white,
            "opening": ["e2e4", "e7e5"], "moves": moves, "result": "*",
            "termination": termination, "plies": 2 + len(moves), "a_score": a_score,
            "sources": {"A": {}, "B": {}}, "elapsed_s": 0.1}


@unittest.skipUnless(_HAS_KIT, "需要 python-chess 与兄弟仓库 Kit")
class TestArenaAggregate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = _load_tool()

    def test_game_dict_and_aggregate(self):
        ckpts = {"A": "a.pt", "B": "b.pt"}
        h1, h2 = [0, 3, 2], [0, 1, 0, 4]
        g0 = self.tool.game_dict(_rec(0, "A", 1.0, ["g1f3", "b8c6"], "truncated"),
                                 ckpts, {0: 7}, h1)
        g1 = self.tool.game_dict(_rec(1, "B", 0.5, ["g1f3"], "truncated"), ckpts, {0: 7}, h2)
        self.assertEqual((g0["arena_result"], g1["arena_result"]), (0, 1))
        self.assertEqual((g0["ckpt_white"], g0["ckpt_black"]), ("a.pt", "b.pt"))
        self.assertEqual((g1["ckpt_white"], g1["ckpt_black"]), ("b.pt", "a.pt"))
        self.assertEqual((g0["n_plies"], g0["opening_id"], g0["is_truncated"]), (2, 7, True))
        self.assertIn("1. e4 e5 2. Nf3 Nc6", g0["pgn"])
        self.assertEqual(g0["expand_depth_hist"], h1)

        args = argparse.Namespace(ckpt_a="a.pt", ckpt_b="b.pt", c_visit=50.0, c_scale_a=0.1,
                                  c_scale_b=0.1, n_sims=12, m0=4, workers=1)
        agg = self.tool._aggregate_results([g0, g1], args)
        self.assertEqual((agg["wins_a"], agg["draws"], agg["wins_b"]), (1, 1, 0))
        self.assertEqual(agg["score_a"], 1.5)
        self.assertEqual(agg["termination"], {"truncated": 2})
        self.assertEqual(agg["distinct_games"], 2)
        self.assertEqual(agg["expand_depth"], hist_summary(hist_merge(h1, h2)))
        self.assertNotIn("sprt", agg)


if __name__ == "__main__":
    unittest.main()
