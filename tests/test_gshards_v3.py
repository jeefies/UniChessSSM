"""v3 分片读写测试（§2.5）。"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np

# 直接加载 gshards 模块，绕过 stateseq.data.__init__ 的 chess 依赖
_gshards_path = os.path.join(os.path.dirname(__file__), "..", "stateseq", "data", "gshards.py")
_gshards_dir = os.path.dirname(_gshards_path)
if _gshards_dir not in sys.path:
    sys.path.insert(0, _gshards_dir)

# 先加载 conditions（无 chess 依赖）
_conditions_path = os.path.join(_gshards_dir, "..", "conditions.py")
_conditions_spec = importlib.util.spec_from_file_location("stateseq.conditions", _conditions_path)
_conditions_mod = importlib.util.module_from_spec(_conditions_spec)
_conditions_spec.loader.exec_module(_conditions_mod)

# 再加载 gshards
_gshards_spec = importlib.util.spec_from_file_location("stateseq.data.gshards", _gshards_path)
gshards = importlib.util.module_from_spec(_gshards_spec)
_gshards_spec.loader.exec_module(gshards)

V3ShardWriter = gshards.V3ShardWriter
V3ShardReader = gshards.V3ShardReader
decode_v3_pipol = gshards.decode_v3_pipol
encode_v3_pipol = gshards.encode_v3_pipol
META_V3_DTYPE = gshards.META_V3_DTYPE


class V3ShardTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_roundtrip(self):
        writer = V3ShardWriter(self.tmpdir, "test")
        for i in range(10):
            meta = np.zeros((), dtype=META_V3_DTYPE)
            meta["n_plies"] = 5
            meta["tc_bucket"] = 2
            meta["result"] = 0
            meta["elo_missing"] = 0
            meta["elo_mean"] = 2500.0
            meta["gen_id"] = 1
            meta["ckpt_step"] = 1000
            meta["termination_reason"] = 0
            meta["is_truncated"] = 0
            actions = np.arange(5, dtype=np.uint16)
            acts = [np.array([0, 1, 2], dtype=np.uint16) for _ in range(5)]
            probs = [np.array([0.5, 0.3, 0.2], dtype=np.float32) for _ in range(5)]
            pipol = encode_v3_pipol(acts, probs)
            poff = np.array([0, 3, 6, 9, 12, 15], dtype=np.int32)
            writer.add(meta, actions, pipol, poff)
        writer.flush()

        reader = V3ShardReader(self.tmpdir)
        self.assertEqual(len(reader.metas[0]), 10)
        rec = reader.game(0)
        np.testing.assert_array_equal(rec["actions"], np.arange(5, dtype=np.uint16))
        self.assertIsNotNone(rec["pipol_actions"])
        self.assertEqual(len(rec["pipol_actions"]), 5)

    def test_pipol_encode_decode(self):
        acts = [np.array([0, 1], dtype=np.uint16), np.array([2], dtype=np.uint16)]
        probs = [np.array([0.7, 0.3], dtype=np.float32), np.array([1.0], dtype=np.float32)]
        blob = encode_v3_pipol(acts, probs)
        da, dp = decode_v3_pipol(blob, 2)
        np.testing.assert_array_equal(da[0], acts[0])
        np.testing.assert_array_equal(da[1], acts[1])
        np.testing.assert_allclose(dp[0], probs[0], atol=1e-3)
        np.testing.assert_allclose(dp[1], probs[1], atol=1e-3)

    def test_prob_sum_check(self):
        acts = [np.array([0, 1, 2], dtype=np.uint16)]
        probs = [np.array([0.33, 0.33, 0.34], dtype=np.float32)]
        blob = encode_v3_pipol(acts, probs)
        da, dp = decode_v3_pipol(blob, 1)
        s = float(dp[0].sum())
        self.assertTrue(0.99 <= s <= 1.01)


if __name__ == "__main__":
    unittest.main()
