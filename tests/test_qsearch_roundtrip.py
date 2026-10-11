"""v3 分片 qsearch（逐 ply 搜索价值，D2/D3）编解码与落盘/读回。

覆盖：
1. ``encode_v3_qsearch`` / ``decode_v3_qsearch`` 定长往返（含 sims=0 哨兵、255 截断）；
2. ``V3ShardWriter(record qsearch)`` → ``V3ShardReader`` 整片读回，偏移对齐（混合"有/无
   记录"的局也不能错位）；
3. 未开记录时**不产生** qsearch 文件，读取端 ``has_qsearch=False``、``game()["qsearch"]=None``
   （老分片/老行为逐位不变）；
4. manifest 的 ``qsearch`` 标记与文件存在性一致。
"""

from __future__ import annotations
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_IMPORT_ROOT = _os.path.dirname(_HERE)   # import 根：~/UniChess
if _IMPORT_ROOT not in _sys.path:
    _sys.path.insert(0, _IMPORT_ROOT)


import json
import os
import tempfile
import unittest

import numpy as np

from SSM.dataset.gshards import (META_V3_DTYPE, QSEARCH_DTYPE, V3ShardReader, V3ShardWriter,
                                 decode_v3_qsearch, encode_v3_qsearch, encode_v3_pipol)


def _meta(n_plies: int, gen_id: int = 7) -> np.ndarray:
    m = np.zeros((), dtype=META_V3_DTYPE)
    m["n_plies"] = n_plies
    m["result"] = 0
    m["game_key"] = "0123456789abcdef"
    m["gen_id"] = gen_id
    m["flags"] = 0
    return m


def _pipol(n_plies: int) -> tuple:
    acts = [np.array([0, 1], dtype=np.uint16)] * n_plies
    probs = [np.array([0.75, 0.25], dtype=np.float32)] * n_plies
    # 每 ply = u16 合法数 + 2×(u16 id + f16 prob) = 2 + 8 = 10 字节
    return encode_v3_pipol(acts, probs), np.array([0, 10 * n_plies], dtype=np.int32)


class QsearchCodecTest(unittest.TestCase):
    def test_roundtrip(self):
        q = [0.0, 0.5, -0.5, 1.0, -1.0, 0.03125]
        sims = [0, 16, 64, 256, 300, 32]
        blob = encode_v3_qsearch(q, sims)
        self.assertEqual(len(blob), len(q) * QSEARCH_DTYPE.itemsize)
        q_back, s_back = decode_v3_qsearch(blob, len(q))
        np.testing.assert_allclose(q_back, np.asarray(q, dtype=np.float32), rtol=0, atol=1e-3)
        # sims 截断到 u8 量程（256 → 255；仅用于"低预算降权"判定，不影响阈值比较）；
        # 0 是无记录哨兵
        np.testing.assert_array_equal(s_back, [0, 16, 64, 255, 255, 32])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            encode_v3_qsearch([0.1, 0.2], [64])

    def test_decode_short_blob(self):
        """blob 比 n_plies 短时读取端按无记录处理（由调用方判长度，这里只验证不越界）。"""
        blob = encode_v3_qsearch([0.5], [64])
        q, s = decode_v3_qsearch(blob, 1)
        self.assertEqual(len(q), 1)
        self.assertEqual(int(s[0]), 64)


class QsearchShardTest(unittest.TestCase):
    def _write(self, out_dir: str, records, record_qsearch: bool):
        w = V3ShardWriter(out_dir, tag="selfplay", shard_size=100)
        for n, q, sims in records:
            pipol, off = _pipol(n)
            qs = encode_v3_qsearch(q, sims) if (record_qsearch and q is not None) else None
            w.add(_meta(n), np.zeros(n, dtype=np.uint16), pipol, off, qsearch=qs)
        w.flush()

    def test_write_read_roundtrip(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            recs = [(3, [0.1, -0.2, 0.3], [64, 64, 16]),
                    (2, [0.0, 0.0], [0, 0]),           # 整局无记录（哨兵）
                    (4, [0.5, 0.5, 0.5, 0.5], [256, 256, 256, 256])]
            self._write(d, recs, record_qsearch=True)
            self.assertTrue(os.path.exists(os.path.join(d, "shard-selfplay-00000.qsearch.bin")))
            with open(os.path.join(d, "manifest.json"), encoding="utf-8") as fh:
                self.assertTrue(json.load(fh)["qsearch"])
            r = V3ShardReader(d)
            self.assertTrue(r.has_qsearch)
            for i, (n, q, sims) in enumerate(recs):
                g = r.game(i)
                self.assertIsNotNone(g["qsearch"], f"第 {i} 局丢了 qsearch")
                q_back, s_back = g["qsearch"]
                self.assertEqual(len(q_back), n)
                np.testing.assert_allclose(q_back, np.asarray(q, dtype=np.float32),
                                           rtol=0, atol=1e-3)
                # sims 截断到 u8 量程
                np.testing.assert_array_equal(s_back, [min(s, 255) for s in sims])
                # 新文件不能破坏既有 pipol / actions 读回
                self.assertEqual(len(g["actions"]), n)
                self.assertIsNotNone(g["pipol_actions"])
                self.assertEqual(len(g["pipol_actions"]), n)
                self.assertEqual(len(g["pipol_actions"][0]), 2)
                self.assertAlmostEqual(float(g["pipol_probs"][0].sum()), 1.0, places=5)

    def test_mixed_presence_keeps_offsets(self):
        """同一片里有的局有记录、有的没有：偏移必须仍然对齐（不能串局）。"""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            recs = [(3, [0.1, 0.2, 0.3], [64, 64, 64]),
                    (2, None, None),                  # 无记录 ⇒ 补等长哨兵
                    (4, [0.4, 0.4, 0.4, 0.4], [64, 64, 64, 64])]
            self._write(d, recs, record_qsearch=True)
            r = V3ShardReader(d)
            q0, s0 = r.game(0)["qsearch"]
            np.testing.assert_allclose(q0, [0.1, 0.2, 0.3], rtol=0, atol=1e-3)
            q1, s1 = r.game(1)["qsearch"]
            np.testing.assert_array_equal(s1, [0, 0])
            self.assertEqual(float(np.abs(q1).max()), 0.0)
            q2, s2 = r.game(2)["qsearch"]
            np.testing.assert_allclose(q2, [0.4] * 4, rtol=0, atol=1e-3)
            np.testing.assert_array_equal(s2, [64] * 4)

    def test_disabled_produces_no_file(self):
        """未开记录：不产生 qsearch 文件，读取端视为老分片（行为不变）。"""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            recs = [(3, [0.1, 0.2, 0.3], [64, 64, 64])]
            self._write(d, recs, record_qsearch=False)
            self.assertFalse(os.path.exists(os.path.join(d, "shard-selfplay-00000.qsearch.bin")))
            with open(os.path.join(d, "manifest.json"), encoding="utf-8") as fh:
                self.assertFalse(json.load(fh)["qsearch"])
            r = V3ShardReader(d)
            self.assertFalse(r.has_qsearch)
            self.assertIsNone(r.game(0)["qsearch"])


if __name__ == "__main__":
    unittest.main()
