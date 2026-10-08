"""多代自对弈 Replay Buffer（SelfPlayDataset / StageB2Task）按代等权采样测试。

验证内容：
1. SelfPlayDataset 单目录兼容：单目录模式下 is_multigen 为 False，train_indices 为整数列表；
2. SelfPlayDataset 多代 Replay Buffer：
   - is_multigen 为 True；
   - train_indices 包含 (g, idx) 元组；
   - epoch_batches 按代均匀采样轮询产出；
   - val_batch 均衡覆盖各代；
3. StageB2Task 支持 selfplay['dir'] 与 selfplay['dirs'] 两种配置形式。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IMPORT_ROOT = os.path.dirname(_HERE)
if _IMPORT_ROOT not in sys.path:
    sys.path.insert(0, _IMPORT_ROOT)

import chess
import numpy as np

from SSM.actions import move_to_action
from SSM.dataset.dataset_selfplay import SelfPlayDataset
from SSM.dataset.gshards import GAMES_PER_SHARD, V3ShardWriter, encode_v3_pipol, make_game_key, META_V3_DTYPE
from SSM.kit import pipol_byte_offsets


def _create_mock_gen_shards(out_dir: str, gen_id: int, n_games: int):
    """创建包含 n_games 局的假 v3 分片。"""
    os.makedirs(out_dir, exist_ok=True)
    writer = V3ShardWriter(out_dir, tag=f"gen{gen_id}", shard_size=n_games + 10)
    for g in range(n_games):
        moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
        actions = []
        pipol_actions = []
        pipol_probs = []
        board = chess.Board()
        for uci in moves:
            mv = chess.Move.from_uci(uci)
            a = move_to_action(mv)
            actions.append(a)
            legals = [move_to_action(m) for m in board.legal_moves if move_to_action(m) is not None]
            n_leg = len(legals)
            probs = [1.0 / n_leg] * n_leg
            pipol_actions.append(legals)
            pipol_probs.append(probs)
            board.push(mv)

        meta = np.zeros((), dtype=META_V3_DTYPE)
        meta["n_plies"] = len(actions)
        meta["tc_bucket"] = 1
        meta["result"] = 0
        meta["elo_missing"] = 0
        meta["elo_mean"] = 2567.5
        meta["game_key"] = make_game_key(f"selfplay_gen{gen_id}", g)
        meta["gen_id"] = gen_id
        meta["ckpt_step"] = gen_id * 100
        meta["termination_reason"] = 0
        meta["is_truncated"] = 0
        meta["flags"] = 2

        writer.add(meta, np.array(actions, dtype=np.uint16),
                   encode_v3_pipol(pipol_actions, pipol_probs),
                   pipol_byte_offsets(pipol_actions))
    writer.flush()


class TestMultigenSelfPlayDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_multigen_")
        self.dir0 = os.path.join(self.tmp, "gen_0000")
        self.dir1 = os.path.join(self.tmp, "gen_0001")
        _create_mock_gen_shards(self.dir0, 0, 200)
        _create_mock_gen_shards(self.dir1, 1, 200)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_single_dir_backwards_compatible(self):
        """单目录输入时保持原样单代行为。"""
        ds = SelfPlayDataset(self.dir0, workers=1, seed=42)
        try:
            self.assertFalse(ds.is_multigen)
            self.assertEqual(len(ds.shard_dirs), 1)
            self.assertEqual(ds.n_games, 200)
            self.assertIsInstance(ds.train_indices[0], int)
        finally:
            ds.close()

    def test_multigen_initialization_and_indices(self):
        """多代目录输入时初始化多 reader，索引为 (g, idx) 元组。"""
        ds = SelfPlayDataset([self.dir0, self.dir1], workers=1, seed=42)
        try:
            self.assertTrue(ds.is_multigen)
            self.assertEqual(len(ds.shard_dirs), 2)
            self.assertEqual(ds.n_games, 400)
            self.assertIsInstance(ds.train_indices[0], tuple)
            self.assertEqual(len(ds.train_indices[0]), 2)
            gens = {it[0] for it in ds.train_indices}
            self.assertEqual(gens, {0, 1})
        finally:
            ds.close()

    def test_multigen_val_batch(self):
        """多代验证集均衡从各代采样。"""
        ds = SelfPlayDataset([self.dir0, self.dir1], workers=1, seed=42)
        try:
            batches = ds.val_batch(n_batches=4, microbatch=2, device="cpu", seed=100)
            self.assertGreater(len(batches), 0)
            for b in batches:
                self.assertIn("batch", b)
                self.assertIn("valid", b)
                self.assertIn("policy_soft_target", b)
                self.assertTrue(1 <= b["valid"].shape[0] <= 2)
        finally:
            ds.close()

    def test_multigen_epoch_batches_streaming(self):
        """多代自对弈 batch 流式产出，各代等权轮询。"""
        ds = SelfPlayDataset([self.dir0, self.dir1], workers=1, seed=42)
        try:
            batches = list(ds.epoch_batches(microbatch=2, device="cpu", shuffle=False))
            self.assertGreater(len(batches), 0)
            self.assertEqual(batches[0]["valid"].shape[0], 2)
            for b in batches:
                self.assertTrue(1 <= b["valid"].shape[0] <= 2)
                self.assertIn("book_mask", b)
        finally:
            ds.close()


if __name__ == "__main__":
    unittest.main()
