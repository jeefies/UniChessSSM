"""v3 分片 自对弈 Sink（make_v3_sink / V3Sink）单元测试。

验证内容：
1. make_v3_sink 工厂接口：支持 out_dir 与 path，支持自定义 tag / shard_size / elo；
2. V3Sink 写入与 .games.jsonl 追踪；
3. done_games() 局号恢复（断点续跑）；
4. close() 刷新与自动 flush；
5. V3ShardWriter 在同一目录下追加写入时不覆盖旧分片。
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
from SSM.dataset.gshards import GAMES_PER_SHARD, V3ShardReader, V3ShardWriter
from SSM.kit import V3Sink, make_v3_sink
from Kit.api import MoveDecision


class DummyDecision:
    def __init__(self, move: chess.Move, pi_ids: list[int], pi_probs: list[float]):
        self.move = move
        self.info = {
            "pi_ids": np.array(pi_ids, dtype=np.uint16),
            "pi": np.array(pi_probs, dtype=np.float32),
        }


class TestV3Sink(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_v3_sink_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_game(self, game_id: int, moves: list[str]) -> tuple[dict, chess.Board, list]:
        board = chess.Board()
        decisions = []
        for uci in moves:
            mv = chess.Move.from_uci(uci)
            a = move_to_action(mv)
            self.assertIsNotNone(a, f"着法 {uci} 无法映射到 action")
            # 简单的 1-hot 合法着概率
            decisions.append(DummyDecision(mv, [a], [1.0]))
            board.push(mv)
        record = {
            "game": game_id,
            "book_plies": 2,
            "moves": moves,
        }
        return record, board, decisions

    def test_factory_creation(self):
        """测试 make_v3_sink 工厂参数解析。"""
        out1 = os.path.join(self.tmp, "s1")
        sink1 = make_v3_sink(out_dir=out1, gen_id=5, shard_size=10)
        self.assertIsInstance(sink1, V3Sink)
        self.assertEqual(sink1.gen_id, 5)
        self.assertEqual(sink1.writer.shard_size, 10)
        self.assertEqual(sink1.done_games(), set())

        out2 = os.path.join(self.tmp, "s2")
        sink2 = make_v3_sink(path=out2, gen_id=2)
        self.assertEqual(sink2.gen_id, 2)
        self.assertEqual(sink2.out_dir, Path(out2))

        with self.assertRaises(ValueError):
            make_v3_sink()

    def test_game_recording_and_done_games(self):
        """测试 on_game_end 写入、.games.jsonl 追踪及 done_games 集合。"""
        out = os.path.join(self.tmp, "rec")
        sink = make_v3_sink(out_dir=out, gen_id=1, shard_size=100)

        # 记录第 0 局（学者将死）
        rec0, b0, dec0 = self._make_game(0, ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"])
        sink.on_game_end(rec0, b0, dec0)

        self.assertEqual(sink.games, 1)
        self.assertEqual(sink.done_games(), {0})

        # 记录第 1 局
        rec1, b1, dec1 = self._make_game(1, ["d2d4", "d7d5", "c2c4"])
        sink.on_game_end(rec1, b1, dec1)

        self.assertEqual(sink.games, 2)
        self.assertEqual(sink.done_games(), {0, 1})

        # 验证 .games.jsonl 写入内容
        meta_file = Path(out) / ".games.jsonl"
        self.assertTrue(meta_file.exists())
        lines = [json.loads(line) for line in meta_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["game"], 0)
        self.assertEqual(lines[0]["plies"], 7)
        self.assertEqual(lines[1]["game"], 1)
        self.assertEqual(lines[1]["plies"], 3)

        # 刷新并关闭
        sink.close()

        # 断点续跑模拟：重新用相同目录创建 Sink
        sink_resumed = make_v3_sink(out_dir=out, gen_id=1, shard_size=100)
        self.assertEqual(sink_resumed.done_games(), {0, 1})
        self.assertEqual(sink_resumed.games, 2)

    def test_close_and_shards_flush(self):
        """测试 close() 自动 flush 并生成分片和 manifest.json。"""
        out = os.path.join(self.tmp, "shards")
        sink = make_v3_sink(out_dir=out, gen_id=3, shard_size=10)

        for g in range(5):
            rec, b, dec = self._make_game(g, ["e2e4", "e7e5"])
            sink.on_game_end(rec, b, dec)

        sink.close()

        manifest_path = Path(out) / "manifest.json"
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["games"], 5)
        self.assertEqual(len(manifest["shards"]), 1)

        # 验证 reader 能够正确读取
        reader = V3ShardReader(out)
        self.assertEqual(len(reader.meta_all), 5)
        self.assertEqual(reader.meta_all["gen_id"][0], 3)
        self.assertEqual(reader.meta_all["n_plies"][0], 2)

    def test_writer_append_does_not_overwrite(self):
        """测试 V3ShardWriter 在同一目录中继续追加写时不覆盖已有分片。"""
        out = os.path.join(self.tmp, "multi_shards")
        sink1 = make_v3_sink(out_dir=out, gen_id=1, shard_size=2)
        for g in range(2):
            rec, b, dec = self._make_game(g, ["e2e4", "e7e5"])
            sink1.on_game_end(rec, b, dec)
        sink1.close()

        manifest1 = json.loads((Path(out) / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest1["shards"]), 1)
        self.assertEqual(manifest1["games"], 2)

        # 续写第 2 批
        sink2 = make_v3_sink(out_dir=out, gen_id=1, shard_size=2)
        for g in range(2, 4):
            rec, b, dec = self._make_game(g, ["d2d4", "d7d5"])
            sink2.on_game_end(rec, b, dec)
        sink2.close()

        manifest2 = json.loads((Path(out) / "manifest.json").read_text(encoding="utf-8"))
        # 应该拥有 2 个分片，共 4 局
        self.assertEqual(len(manifest2["shards"]), 2)
        self.assertEqual(manifest2["games"], 4)

        reader = V3ShardReader(out)
        self.assertEqual(len(reader.meta_all), 4)


if __name__ == "__main__":
    unittest.main()
