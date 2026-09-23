"""kit 驱动的 arena（``tools/ssm_gumbel_arena.py``）端到端测试。需要 CUDA 与兄弟仓库 Kit。

随机初始化的两个检查点（写到临时目录），小预算跑满整条链：
1. 单进程与 2 进程（并发 1）逐局相同——棋谱、结果、扩展深度直方图（多进程时直方图经
   worker 事件转发，验证它在该局结果之前完整到达）；
2. 断点续跑：截掉一半结果后重跑，最终 games.jsonl 与一次跑完逐局相同（直方图从
   expand_hist.jsonl 读回）；
3. ``main`` 写出 arena.json / games.jsonl / model_ids.json，聚合口径（计分、终止分布、
   distinct_games、扩展深度）与逐局记录一致。
逐局与切换前实现一致的证据见 git 历史中的 ``tools/kit_arena_parity.py``（真实权重 16/16）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
KIT_ROOT = os.environ.get("UNICHESS_KIT_ROOT", os.path.join(os.path.dirname(HERE), "Kit"))
if os.path.isdir(KIT_ROOT) and KIT_ROOT not in sys.path:
    sys.path.append(KIT_ROOT)  # 追加而非前插：Kit 的 tests 包不得遮蔽本仓库的 tests

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover - 本机（Windows）无 torch
    _HAS_CUDA = False

try:
    import unichess_kit  # noqa: F401

    _HAS_KIT = True
except ImportError:  # pragma: no cover
    _HAS_KIT = False

from stateseq.depth_hist import hist_merge, hist_summary  # noqa: E402


def _load_tool():
    path = os.path.join(HERE, "tools", "ssm_gumbel_arena.py")
    spec = importlib.util.spec_from_file_location("ssm_gumbel_arena_e2e", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _strip(games):
    return [{k: v for k, v in g.items() if k != "elapsed_s"} for g in games]


@unittest.skipUnless(_HAS_CUDA and _HAS_KIT, "需要 CUDA（mamba 步进核）与兄弟仓库 Kit")
class TestArenaKit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from stateseq.model import SeqModel

        cls.tool = _load_tool()
        cls.tmp = tempfile.mkdtemp(prefix="arena_kit_")
        cls.ckpts = []
        for seed in (1, 2):
            torch.manual_seed(seed)
            path = os.path.join(cls.tmp, f"rand{seed}.pt")
            torch.save({"model": SeqModel(dropout=0.0).state_dict()}, path)
            cls.ckpts.append(path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _args(self, out, **kw):
        d = dict(ckpt_a=self.ckpts[0], ckpt_b=self.ckpts[1], out=out, games=4,
                 openings_file="", n_sims=8, m0=4, max_plies=6, seed=5, workers=1,
                 concurrency=1, c_visit=50.0, c_scale_a=0.1, c_scale_b=0.1, sprt=False,
                 sprt_min_games=64, sprt_alpha=0.05, sprt_beta=0.05, pairs=8)
        d.update(kw)
        return argparse.Namespace(**d)

    def _run(self, name, **kw):
        out = os.path.join(self.tmp, name)
        games, _, summary = self.tool.run_arena(self._args(out, **kw))
        return out, games, summary

    def test_workers_equal_single_process(self):
        _, g1, s1 = self._run("w1")
        _, g2, _ = self._run("w2", workers=2)
        self.assertEqual(len(g1), 4)
        self.assertEqual(_strip(g1), _strip(g2))
        for g in g1:
            self.assertTrue(g["expand_depth_hist"], "每局都应有扩展深度直方图")
            s = hist_summary(g["expand_depth_hist"])
            self.assertLessEqual(s["expansions"], 8 * g["n_plies"])
            self.assertLessEqual(g["n_plies"], 6, "--max_plies 只计开局之后")
            if g["termination_reason"] == "truncated":
                self.assertEqual(g["n_plies"], 6)
            self.assertIsNone(g["anomaly"])
        self.assertEqual(len({g["pgn"] for g in g1}), 4)
        self.assertEqual(s1["games"], 4)

    def test_resume_matches_full_run(self):
        # 续跑在同一目录进行（配置哈希含开局文件路径，内置库就写在输出目录里）
        part, full, _ = self._run("resume")
        kit_path = os.path.join(part, "kit_results.jsonl")
        with open(kit_path, encoding="utf-8") as fh:
            lines = [ln for ln in fh if ln.strip()]
        kept = [ln for ln in lines if json.loads(ln).get("type") != "game"]
        games = [ln for ln in lines if json.loads(ln).get("type") == "game"][:2]
        with open(kit_path, "w", encoding="utf-8") as fh:
            fh.writelines(kept + games)
        keep_ids = {json.loads(ln)["game"] for ln in games}
        hist_path = os.path.join(part, "expand_hist.jsonl")
        with open(hist_path, encoding="utf-8") as fh:
            rows = [ln for ln in fh if ln.strip() and json.loads(ln)["game"] in keep_ids]
        with open(hist_path, "w", encoding="utf-8") as fh:
            fh.writelines(rows)
        resumed, _, _ = self.tool.run_arena(self._args(part))
        self.assertEqual(_strip(resumed), _strip(full))

    def test_main_outputs(self):
        out = os.path.join(self.tmp, "main")
        argv = ["ssm_gumbel_arena.py", "--ckpt-a", self.ckpts[0], "--ckpt-b", self.ckpts[1],
                "--out", out, "--games", "4", "--n_sims", "8", "--m0", "4", "--max_plies", "6",
                "--seed", "5", "--concurrency", "4", "--openings-file", ""]
        with mock.patch.object(sys, "argv", argv):
            self.tool.main()
        with open(os.path.join(out, "arena.json")) as fh:
            arena = json.load(fh)
        with open(os.path.join(out, "games.jsonl")) as fh:
            games = [json.loads(ln) for ln in fh]
        with open(os.path.join(out, "model_ids.json")) as fh:
            ids = json.load(fh)
        self.assertFalse(ids["same_hash"])
        self.assertTrue(ids["forward_comparison"]["models_differ_functionally"])
        self.assertEqual(arena["total_games"], 4)
        self.assertEqual(len(games), 4)
        self.assertEqual(arena["wins_a"] + arena["wins_b"] + arena["draws"], 4)
        self.assertEqual(arena["score_a"],
                         sum({0: 1.0, 1: 0.5, 2: 0.0}[g["arena_result"]] for g in games))
        self.assertEqual(sum(arena["termination"].values()), 4)
        self.assertEqual(arena["distinct_games"], len({self.tool._pgn_fingerprint(g)
                                                       for g in games}))
        hist: list = []
        for g in games:
            hist = hist_merge(hist, g["expand_depth_hist"])
        self.assertEqual(arena["expand_depth"], hist_summary(hist))
        self.assertIn("elo", arena["kit"])
        # A/B 执白交替：同一对两局同开局、执色互换
        for p in range(2):
            a, b = games[2 * p], games[2 * p + 1]
            self.assertEqual((a["white_ckpt_side"], b["white_ckpt_side"]), ("A", "B"))
            self.assertEqual(a["opening_id"], b["opening_id"])


if __name__ == "__main__":
    unittest.main()
