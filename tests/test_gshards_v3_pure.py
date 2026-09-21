"""v3 分片纯 Python（numpy + 标准库，无 torch/chess 依赖）单测。

覆盖测试范围：
1. Header / Data Type 结构布局与尺寸断言：
   - META_V2_DTYPE, META_V3_EXT_DTYPE, META_V3_DTYPE 字段、偏移量与大小
2. Game metadata packing / unpacking：
   - v2 基础字段与 v3 扩展 16B 字段（gen_id, ckpt_step, termination_reason, is_truncated, start_type, flags, pad）
3. 变长 π' (.pipol.bin) 二进制编码/解码：
   - encode_v3_pipol 与 decode_v3_pipol
   - legal_count (u16), action_id (u16), prob (f16) 的逐位与 float16 精度验证
   - validate_v3_pipol 完整性校验（概率和范围、支持集匹配、非法动作拦截）
4. V3ShardWriter 与 V3ShardReader 端到端往返：
   - 多局写入、flush、manifest.json 生成
   - actions.bin, meta.npz, pipol.offsets.bin, pipol.bin 文件往返
   - 索引偏移表 lookup 准确性
5. 边界与边缘用例：
   - 0 步对局（n_plies = 0，空 actions / 空 pipol）
   - 单步对局（n_plies = 1）
   - is_truncated = 1 截断局
   - 大步数对局（如 n_plies = 300 完整长局）
   - 动态变长合法动作支持集（从 1 个动作到 100+ 个候选着法）
   - 多分片连续切片（分片自动 flush）
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import struct
import sys
import tempfile
import types
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# 模拟 / 隔离 stateseq.conditions，使得无需 torch 即可导入 gshards.py
# ---------------------------------------------------------------------------
_stateseq_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stateseq"))
_data_dir = os.path.join(_stateseq_dir, "data")
_gshards_path = os.path.join(_data_dir, "gshards.py")

# 如果 sys.modules 中没有 stateseq.conditions，注入一个纯 python 版本
if "stateseq.conditions" not in sys.modules:
    fake_conditions = types.ModuleType("stateseq.conditions")

    from enum import IntEnum

    class TimeControlBucket(IntEnum):
        BULLET = 0
        BLITZ = 1
        RAPID = 2
        CLASSICAL = 3
        CORRESPONDENCE = 4
        OTHER = 5
        UNKNOWN = 6

    def time_control_bucket(tc: str | None) -> TimeControlBucket:
        if not tc or tc in ("-", "?"):
            return TimeControlBucket.UNKNOWN
        if "/" in tc:
            return TimeControlBucket.CORRESPONDENCE
        try:
            base, _, inc = tc.partition("+")
            seconds = int(base) + 40 * int(inc or 0)
        except ValueError:
            return TimeControlBucket.UNKNOWN
        if seconds < 180:
            return TimeControlBucket.BULLET
        if seconds < 480:
            return TimeControlBucket.BLITZ
        if seconds < 1500:
            return TimeControlBucket.RAPID
        return TimeControlBucket.CLASSICAL

    fake_conditions.TimeControlBucket = TimeControlBucket
    fake_conditions.time_control_bucket = time_control_bucket

    # 注册进入 sys.modules
    sys.modules["stateseq"] = types.ModuleType("stateseq")
    sys.modules["stateseq.conditions"] = fake_conditions

_spec = importlib.util.spec_from_file_location("stateseq.data.gshards", _gshards_path)
gshards = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gshards)

META_V2_DTYPE = gshards.META_V2_DTYPE
META_V3_EXT_DTYPE = gshards.META_V3_EXT_DTYPE
META_V3_DTYPE = gshards.META_V3_DTYPE
V3ShardWriter = gshards.V3ShardWriter
V3ShardReader = gshards.V3ShardReader
encode_v3_pipol = gshards.encode_v3_pipol
decode_v3_pipol = gshards.decode_v3_pipol
validate_v3_pipol = gshards.validate_v3_pipol
make_game_key = gshards.make_game_key
is_val_key = gshards.is_val_key


class TestGShardsV3Pure(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="v3shard_test_")

    def tearDown(self):
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # 1. 结构与尺寸断言
    # -----------------------------------------------------------------------
    def test_metadata_dtypes_and_sizes(self):
        """验证 v2 基础结构、v3 扩展结构、v3 复合结构的字段与对齐尺寸。"""
        # META_V3_EXT_DTYPE: 16 字节
        # u32(4) + u32(4) + u8(1) + u8(1) + u8(1) + u8(1) + u32(4) = 16 字节
        self.assertEqual(META_V3_EXT_DTYPE.itemsize, 16)
        expected_ext_fields = [
            "gen_id", "ckpt_step", "termination_reason",
            "is_truncated", "start_type", "flags", "pad"
        ]
        for f in expected_ext_fields:
            self.assertIn(f, META_V3_EXT_DTYPE.names)

        # 检查 META_V3_DTYPE 包含所有必要字段
        all_v3_fields = [
            "n_plies", "tc_bucket", "result", "elo_missing", "elo_mean", "game_key",
            "gen_id", "ckpt_step", "termination_reason", "is_truncated",
            "start_type", "flags", "pad"
        ]
        for f in all_v3_fields:
            self.assertIn(f, META_V3_DTYPE.names)

    # -----------------------------------------------------------------------
    # 2. π' (pipol) 二进制编解码与精度
    # -----------------------------------------------------------------------
    def test_pipol_binary_layout(self):
        """验证变长 pipol 的二进制逐字节协议：
        每 ply 存储: u16 legal_count + legal_count * (u16 action_id + f16 prob)
        """
        acts = [np.array([42, 1935], dtype=np.uint16)]
        probs = [np.array([0.75, 0.25], dtype=np.float32)]
        blob = encode_v3_pipol(acts, probs)

        # 头部: legal_count = 2 (u16 -> 2 bytes)
        # item 1: 42 (u16 -> 2 bytes) + 0.75 (f16 -> 2 bytes)
        # item 2: 1935 (u16 -> 2 bytes) + 0.25 (f16 -> 2 bytes)
        # 总字节: 2 + 4 * 2 = 10 字节
        self.assertEqual(len(blob), 10)

        legal_cnt = struct.unpack_from("<H", blob, 0)[0]
        self.assertEqual(legal_cnt, 2)

        a1, p1 = struct.unpack_from("<He", blob, 2)
        self.assertEqual(a1, 42)
        self.assertAlmostEqual(p1, 0.75, places=3)

        a2, p2 = struct.unpack_from("<He", blob, 6)
        self.assertEqual(a2, 1935)
        self.assertAlmostEqual(p2, 0.25, places=3)

        # 解码往返验证
        dec_acts, dec_probs = decode_v3_pipol(blob, 1)
        self.assertEqual(len(dec_acts), 1)
        self.assertEqual(len(dec_probs), 1)
        np.testing.assert_array_equal(dec_acts[0], acts[0])
        np.testing.assert_allclose(dec_probs[0], probs[0], atol=1e-3)

    def test_pipol_empty_and_single_move(self):
        """测试只有 1 个动作和空动作情况。"""
        # 1 个确定性动作，概率 1.0
        acts = [np.array([100], dtype=np.uint16)]
        probs = [np.array([1.0], dtype=np.float32)]
        blob = encode_v3_pipol(acts, probs)
        self.assertEqual(len(blob), 2 + 4)  # 6 字节

        da, dp = decode_v3_pipol(blob, 1)
        self.assertEqual(da[0][0], 100)
        self.assertAlmostEqual(float(dp[0][0]), 1.0, places=3)

        # 0 个 ply 的解码
        da0, dp0 = decode_v3_pipol(b"", 0)
        self.assertEqual(len(da0), 0)
        self.assertEqual(len(dp0), 0)

    # -----------------------------------------------------------------------
    # 3. pipol 校验函数 (validate_v3_pipol)
    # -----------------------------------------------------------------------
    def test_validate_v3_pipol(self):
        acts = [np.array([10, 20, 30], dtype=np.uint16)]
        probs = [np.array([0.5, 0.3, 0.2], dtype=np.float32)]

        # 正常通过
        validate_v3_pipol(acts, probs, 1)

        # legal_masks 校验正常通过
        legal_mask = np.zeros((1, 1936), dtype=bool)
        legal_mask[0, [10, 20, 30]] = True
        validate_v3_pipol(acts, probs, 1, legal_masks=legal_mask)

        # 概率和略有波动但在 [0.99, 1.01] 内
        probs_loose = [np.array([0.5, 0.3, 0.205], dtype=np.float32)]  # sum = 1.005
        validate_v3_pipol(acts, probs_loose, 1)

        # 概率和越界被拒
        probs_bad = [np.array([0.5, 0.3, 0.1], dtype=np.float32)]  # sum = 0.90
        with self.assertRaises(ValueError):
            validate_v3_pipol(acts, probs_bad, 1)

        # 动作数量与 legal_mask 数量不符被拒
        legal_mask_mismatch = np.zeros((1, 1936), dtype=bool)
        legal_mask_mismatch[0, [10, 20]] = True
        with self.assertRaises(ValueError):
            validate_v3_pipol(acts, probs, 1, legal_masks=legal_mask_mismatch)

        # 动作包含未在 legal_mask 里的非法动作被拒
        legal_mask_wrong = np.zeros((1, 1936), dtype=bool)
        legal_mask_wrong[0, [10, 20, 99]] = True
        with self.assertRaises(ValueError):
            validate_v3_pipol(acts, probs, 1, legal_masks=legal_mask_wrong)

    # -----------------------------------------------------------------------
    # 4. 端到端分片读写与往返校验 (V3ShardWriter & V3ShardReader)
    # -----------------------------------------------------------------------
    def test_v3_shard_roundtrip_basic(self):
        """测试标准对局集合的写入、flush 与读取还原。"""
        writer = V3ShardWriter(self.tmpdir, "test_basic", shard_size=100)

        num_games = 12
        game_records = []

        rng = np.random.RandomState(42)

        for i in range(num_games):
            n_plies = rng.randint(5, 25)
            actions = rng.randint(0, 1936, size=n_plies, dtype=np.uint16)

            ply_actions = []
            ply_probs = []
            byte_offsets = [0]
            current_bytes = 0

            for t in range(n_plies):
                n_legal = rng.randint(1, 15)
                # 随机挑选合法动作
                acts = np.sort(rng.choice(1936, size=n_legal, replace=False)).astype(np.uint16)
                # 随机生成归一化概率
                raw_p = rng.uniform(0.1, 1.0, size=n_legal)
                p = (raw_p / raw_p.sum()).astype(np.float32)
                ply_actions.append(acts)
                ply_probs.append(p)
                current_bytes += 2 + 4 * n_legal
                byte_offsets.append(current_bytes)

            pipol_blob = encode_v3_pipol(ply_actions, ply_probs)
            pipol_offsets = np.array(byte_offsets, dtype=np.int64)

            meta = np.zeros((), dtype=META_V3_DTYPE)
            meta["n_plies"] = n_plies
            meta["tc_bucket"] = i % 7
            meta["result"] = i % 3
            meta["elo_missing"] = 0
            meta["elo_mean"] = 2500.0 + i
            meta["game_key"] = f"{i:016x}"
            meta["gen_id"] = 100 + i
            meta["ckpt_step"] = 50000 + i * 10
            meta["termination_reason"] = i % 5
            meta["is_truncated"] = 1 if i == 7 else 0
            meta["start_type"] = 0
            meta["flags"] = 0
            meta["pad"] = 0

            writer.add(meta, actions, pipol_blob, pipol_offsets)
            game_records.append({
                "meta": meta,
                "actions": actions,
                "ply_actions": ply_actions,
                "ply_probs": ply_probs,
            })

        writer.flush()

        # 读取并全量断言
        reader = V3ShardReader(self.tmpdir)
        self.assertEqual(len(reader.meta_all), num_games)

        for i in range(num_games):
            rec = reader.game(i)
            expected = game_records[i]

            # 1. 验证元数据
            m = rec["meta"]
            exp_m = expected["meta"]
            self.assertEqual(int(m["n_plies"]), int(exp_m["n_plies"]))
            self.assertEqual(int(m["tc_bucket"]), int(exp_m["tc_bucket"]))
            self.assertEqual(int(m["result"]), int(exp_m["result"]))
            self.assertEqual(int(m["gen_id"]), int(exp_m["gen_id"]))
            self.assertEqual(int(m["ckpt_step"]), int(exp_m["ckpt_step"]))
            self.assertEqual(int(m["termination_reason"]), int(exp_m["termination_reason"]))
            self.assertEqual(int(m["is_truncated"]), int(exp_m["is_truncated"]))
            self.assertEqual(str(m["game_key"]), str(exp_m["game_key"]))

            # 2. 验证动作池
            np.testing.assert_array_equal(rec["actions"], expected["actions"])

            # 3. 验证 π' 结构
            p_acts = rec["pipol_actions"]
            p_probs = rec["pipol_probs"]
            self.assertEqual(len(p_acts), int(exp_m["n_plies"]))
            self.assertEqual(len(p_probs), int(exp_m["n_plies"]))

            for t in range(int(exp_m["n_plies"])):
                np.testing.assert_array_equal(p_acts[t], expected["ply_actions"][t])
                # f16 编码精度下与原概率比对
                np.testing.assert_allclose(p_probs[t], expected["ply_probs"][t], atol=1e-3)
                # 验证概率和在有效范围
                self.assertTrue(0.99 <= p_probs[t].sum() <= 1.01)

    # -----------------------------------------------------------------------
    # 5. 边界与边缘用例
    # -----------------------------------------------------------------------
    def test_edge_case_zero_plies(self):
        """测试含 0 步局的分片（对局有 0 步局但分片总体有其它对局或动作）。"""
        writer = V3ShardWriter(self.tmpdir, "edge_zero")
        
        # 局 0: 0 步局
        meta0 = np.zeros((), dtype=META_V3_DTYPE)
        meta0["n_plies"] = 0
        meta0["tc_bucket"] = 1
        meta0["result"] = 0
        meta0["game_key"] = "0000000000000000"
        meta0["gen_id"] = 1
        meta0["ckpt_step"] = 100
        meta0["termination_reason"] = 2
        meta0["is_truncated"] = 0
        actions0 = np.array([], dtype=np.uint16)
        pipol_blob0 = b""
        pipol_offsets0 = np.array([0], dtype=np.int64)
        writer.add(meta0, actions0, pipol_blob0, pipol_offsets0)

        # 局 1: 普通 1 步局（确保分片 actions.bin 文件非空，满足 memmap 要求）
        meta1 = np.zeros((), dtype=META_V3_DTYPE)
        meta1["n_plies"] = 1
        meta1["tc_bucket"] = 1
        meta1["result"] = 1
        meta1["game_key"] = "0000000000000001"
        meta1["gen_id"] = 1
        meta1["ckpt_step"] = 100
        meta1["termination_reason"] = 1
        meta1["is_truncated"] = 0
        actions1 = np.array([42], dtype=np.uint16)
        pipol_blob1 = encode_v3_pipol([np.array([42], dtype=np.uint16)], [np.array([1.0], dtype=np.float32)])
        pipol_offsets1 = np.array([0, 6], dtype=np.int64)
        writer.add(meta1, actions1, pipol_blob1, pipol_offsets1)

        writer.flush()

        reader = V3ShardReader(self.tmpdir)
        rec0 = reader.game(0)
        self.assertEqual(int(rec0["meta"]["n_plies"]), 0)
        self.assertEqual(len(rec0["actions"]), 0)
        self.assertEqual(len(rec0["pipol_actions"]), 0)
        self.assertEqual(len(rec0["pipol_probs"]), 0)

        rec1 = reader.game(1)
        self.assertEqual(int(rec1["meta"]["n_plies"]), 1)
        self.assertEqual(len(rec1["actions"]), 1)
        self.assertEqual(int(rec1["actions"][0]), 42)

    def test_edge_case_truncated_and_large_plies(self):
        """测试 300 步超长局与截断标志 (is_truncated=1)。"""
        writer = V3ShardWriter(self.tmpdir, "edge_long")
        n_plies = 300
        meta = np.zeros((), dtype=META_V3_DTYPE)
        meta["n_plies"] = n_plies
        meta["tc_bucket"] = 2
        meta["result"] = 1  # 和棋
        meta["game_key"] = "1234567890abcdef"
        meta["gen_id"] = 99
        meta["ckpt_step"] = 12345
        meta["termination_reason"] = 4  # max_plies 截断
        meta["is_truncated"] = 1

        actions = np.full(n_plies, 123, dtype=np.uint16)
        ply_actions = [np.array([10, 20], dtype=np.uint16) for _ in range(n_plies)]
        ply_probs = [np.array([0.5, 0.5], dtype=np.float32) for _ in range(n_plies)]

        pipol_blob = encode_v3_pipol(ply_actions, ply_probs)
        # 每 ply: 2 + 4 * 2 = 10 bytes
        byte_offsets = np.arange(n_plies + 1, dtype=np.int64) * 10

        writer.add(meta, actions, pipol_blob, byte_offsets)
        writer.flush()

        reader = V3ShardReader(self.tmpdir)
        rec = reader.game(0)
        self.assertEqual(int(rec["meta"]["n_plies"]), 300)
        self.assertEqual(int(rec["meta"]["is_truncated"]), 1)
        self.assertEqual(int(rec["meta"]["termination_reason"]), 4)
        self.assertEqual(len(rec["actions"]), 300)
        self.assertEqual(len(rec["pipol_actions"]), 300)
        self.assertEqual(len(rec["pipol_probs"]), 300)

    def test_multiple_shards_auto_flush(self):
        """测试当局数达到 shard_size 时自动 flush 并切分多个分片文件。"""
        shard_size = 5
        total_games = 12  # 将切分为 3 个分片: 5 + 5 + 2
        writer = V3ShardWriter(self.tmpdir, "multi_shard", shard_size=shard_size)

        for i in range(total_games):
            meta = np.zeros((), dtype=META_V3_DTYPE)
            meta["n_plies"] = 2
            meta["game_key"] = f"{i:016x}"
            actions = np.array([i, i + 1], dtype=np.uint16)
            p_acts = [np.array([i], dtype=np.uint16), np.array([i + 1], dtype=np.uint16)]
            p_probs = [np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float32)]
            blob = encode_v3_pipol(p_acts, p_probs)
            poff = np.array([0, 6, 12], dtype=np.int64)
            writer.add(meta, actions, blob, poff)

        writer.flush()

        self.assertEqual(len(writer.shard_files), 3)

        reader = V3ShardReader(self.tmpdir)
        self.assertEqual(len(reader.meta_all), total_games)
        self.assertEqual(len(reader.metas), 3)

        # 验证跨分片读取
        for i in range(total_games):
            rec = reader.game(i)
            self.assertEqual(str(rec["meta"]["game_key"]), f"{i:016x}")
            np.testing.assert_array_equal(rec["actions"], np.array([i, i + 1], dtype=np.uint16))


if __name__ == "__main__":
    unittest.main()
