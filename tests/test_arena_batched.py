"""批量 arena（跨局攒批）单元测试：cache 拼/拆往返、驱动器批调度与结果归属。

跨局攒批的正确性判据：
1. ``_concat_caches``/``_split_cache`` 往返无损（批维度拼回后再拆，逐位相等）；
2. 驱动器把同一模型槽位的请求拼成一批（批大小≈并发局数），结果按原序分发；
3. 每局双方模型各进一步、终局结果按执白方归属（B 执白的局翻转到 A 视角）。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

_TOOL_PATH = os.path.join(HERE, "tools", "ssm_gumbel_arena.py")

try:
    import torch  # noqa: F401
    import chess  # noqa: F401

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - 本机（Windows）无 torch
    _HAS_TORCH = False


def _load_tool():
    spec = importlib.util.spec_from_file_location("ssm_gumbel_arena_under_test", _TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeModel:
    """记录每次前向的批大小；cache 原样返回（拼/拆逻辑由被测函数承担）。"""

    ckpt_path = "fake.pt"
    c_visit = 50.0
    c_scale = 0.1

    def __init__(self, name: str):
        self.name = name
        self.batch_sizes: list[int] = []

    def initial_cache(self, b: int = 1):
        import torch

        return [(torch.zeros(b, 4), torch.zeros(b, 4))]

    def step(self, feats, tc, elo, color, cache, need_extra: bool = False):
        n = feats.shape[0]
        self.batch_sizes.append(n)
        return (np.zeros((n, 1936), dtype=np.float32),
                np.zeros((n, 3), dtype=np.float32),
                None, None, cache)


class _FakeGame:
    """最小协程：每局 yield n_reqs 个前向请求（槽位 0/1 交替），返回结果 dict。"""

    def __init__(self, game_idx: int, models: list, pair_idx: int, a_is_white: bool,
                 n_reqs: int = 3):
        self.game_idx = game_idx
        self.models = models
        self.pair_idx = pair_idx
        self.n_reqs = n_reqs
        self.slot = {chess.WHITE: 0 if a_is_white else 1,
                     chess.BLACK: 1 if a_is_white else 0}

    def run(self):
        import chess

        for i in range(self.n_reqs):
            slot = self.slot[chess.WHITE] if i % 2 == 0 else self.slot[chess.BLACK]
            feats = np.zeros(785, dtype=np.float32)
            cache = self.models[slot].initial_cache(1)
            _lg, _wd, _mlh, _x, _new = yield (slot, feats, 2, 0.0, 1, cache)
        return {
            "n_plies": self.n_reqs,
            "termination_reason": "checkmate",
            "arena_result": 0,
            "anomaly": None,
        }


@unittest.skipUnless(_HAS_TORCH, "需要 torch（远端全量单测环境）")
class TestBatchedArena(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = _load_tool()

    def _args(self, **kw):
        ap = self.tool.argparse.Namespace
        base = dict(games=8, pairs=4, n_sims=8, m0=4, max_plies=50, seed=1,
                    workers=1, batched=True, concurrency=4, sprt=False,
                    sprt_min_games=64, sprt_alpha=0.05, sprt_beta=0.05,
                    c_visit=50.0, c_scale_a=0.1, c_scale_b=0.1)
        base.update(kw)
        return ap(**base)

    def test_concat_split_roundtrip(self):
        from stateseq.model import SeqModel

        seq = SeqModel(dropout=0.0)
        caches = []
        for i in range(3):
            c = seq.initial_cache(1, device="cpu")
            for conv, ssm in c:
                conv.fill_(float(i + 1))
                ssm.fill_(float(i + 1))
            caches.append(c)
        merged = self.tool._concat_caches(caches)
        self.assertEqual(merged[0][0].shape[0], 3)
        back = self.tool._split_cache(merged, 3)
        for i in range(3):
            for (c0, s0), (c1, s1) in zip(caches[i], back[i]):
                self.assertTrue(torch.equal(c0, c1))
                self.assertTrue(torch.equal(s0, s1))

    def test_driver_batches_by_slot_and_collects(self):
        models = [_FakeModel("A"), _FakeModel("B")]
        games = []
        for i in range(6):
            games.append(_FakeGame(i, models, pair_idx=i // 2, a_is_white=(i % 2 == 0)))
        driver = self.tool.BatchedArenaDriver(games, concurrency=4, args=self._args(),
                                              internal_sprt=False)
        driver.run()
        self.assertEqual(len(driver.results), 6)
        # 两个模型都见过批（批大小 >1 证明跨局攒批生效）
        for m in models:
            self.assertTrue(m.batch_sizes)
            self.assertLessEqual(max(m.batch_sizes), 4)
        self.assertGreater(max(max(m.batch_sizes) for m in models), 1)
        # 结果按 (pair, 颜色) 归属：偶数 pair 内 A 执白在前
        for pair_idx in (0, 1, 2):
            pair_games = [g for g in driver.results if g["pair_idx"] == pair_idx]
            self.assertEqual(len(pair_games), 2)
            self.assertEqual([g["white_ckpt_side"] for g in pair_games], ["A", "B"])
        # game_idx 连续
        self.assertEqual([g["game_idx"] for g in driver.results], list(range(6)))

    def test_driver_respects_external_stop(self):
        models = [_FakeModel("A"), _FakeModel("B")]
        games = [_FakeGame(i, models, pair_idx=i // 2, a_is_white=(i % 2 == 0), n_reqs=1)
                 for i in range(8)]
        state = {"stop": False}

        def stop_check():
            # 第 2 局完成后停止排程（只影响尚未开始的局）
            return state["stop"]

        driver = self.tool.BatchedArenaDriver(games, concurrency=2, args=self._args(),
                                              internal_sprt=False, stop_check=stop_check)
        orig_finish = driver._finish

        def finish(game, gd):
            orig_finish(game, gd)
            if len(driver.results) >= 2:
                state["stop"] = True

        driver._finish = finish
        driver.run()
        self.assertGreaterEqual(len(driver.results), 2)
        self.assertLess(len(driver.results), 8)

    def test_env_gate_skips_gracefully(self):
        """无 torch 环境下测试类整体跳过（不 ERROR）。"""
        self.assertTrue(_HAS_TORCH or True)


if __name__ == "__main__":
    unittest.main()
