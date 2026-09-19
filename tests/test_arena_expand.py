"""Review round-3 回归：arena 展开必须携带完整 path（否则深度≥2 节点在错误局面上评估）。

覆盖 `tools/ssm_gumbel_arena.py::_expand_child`：
1. 子节点 path/depth 正确传播；
2. 被评估局面特征 == 根局面 + node.path + action（逐位相等，规则引擎给出权威局面）；
3. 深度≥2 的节点从根重放 path（而不是把深层动作错套在根局面上）；
4. occurrence 采用 encode-before-increment（与生成器/训练重放一致）；
5. 路径中的非法动作必须抛错（不得伪造 q=0 终局）；
6. 将杀叶子 q 取终局真值。

不需要 GPU：fake model 只实现 ArenaModel 的 step 接口。
"""

from __future__ import annotations

import unittest

import chess
import numpy as np
import torch

from stateseq.actions import move_to_action
from stateseq.adapter import encode_board
from stateseq.data.sequences import _board_key
from stateseq.gumbel import Node
from tools.ssm_gumbel_arena import _expand_child


class _FakeModel:
    """最小 ArenaModel 替身：记录被评估特征，cache 为可克隆的张量。"""

    device = "cpu"

    def __init__(self):
        self.calls: list[np.ndarray] = []

    def initial_cache(self, b: int = 1):
        return [(torch.zeros(1), torch.zeros(1))]

    def step(self, feats, tc, elo, color, cache):
        self.calls.append(np.asarray(feats, dtype=np.float32).copy())
        logits = np.zeros((1, 1936), dtype=np.float32)
        wdl = np.zeros((1, 3), dtype=np.float32)
        mlh = np.zeros((1, 1), dtype=np.float32)
        x = np.zeros((1, 512), dtype=np.float32)
        n = cache[0][0] + 1.0
        return logits, wdl, mlh, x, [(n, n)]


def _aid(uci: str) -> int:
    return move_to_action(chess.Move.from_uci(uci))


class ArenaExpandTest(unittest.TestCase):
    def setUp(self):
        self.model = _FakeModel()
        self.board = chess.Board()
        self.cache = self.model.initial_cache(1)
        self.occur = {_board_key(self.board): 1}  # 模拟根局面已计过一次

    def _root(self):
        return Node(legal=np.array([0, 1], dtype=np.int64),
                    logits=np.zeros(2, dtype=np.float32), q=0.0)

    def test_path_propagates_and_position_correct(self):
        a1 = _aid("e2e4")
        child = _expand_child(self.model, self.board, self.cache, self.occur, self._root(), a1)
        self.assertEqual(child.path, (a1,))
        self.assertEqual(child.depth, 1)
        self.assertFalse(child.terminal)

        b_after = chess.Board()
        b_after.push_uci("e2e4")
        exp_feats, _, _, _ = encode_board(b_after, 0)
        np.testing.assert_array_equal(self.model.calls[-1], exp_feats.reshape(1, -1))

        # 深度 2：必须从根重放 a1，再评估 a2 —— 特征等于 根+a1+a2
        a2 = _aid("e7e5")
        grand = _expand_child(self.model, self.board, self.cache, self.occur, child, a2)
        self.assertEqual(grand.path, (a1, a2))
        self.assertEqual(grand.depth, 2)
        b2 = chess.Board()
        b2.push_uci("e2e4")
        b2.push_uci("e7e5")
        exp2, _, _, _ = encode_board(b2, 0)
        np.testing.assert_array_equal(self.model.calls[-1], exp2.reshape(1, -1))

    def test_occurrence_encode_before_increment(self):
        # 根局面已计 1 次；4 步循环回到该局面 → occurrence=1 → is1=1, is2=0
        a1 = _aid("g1f3")
        child = _expand_child(self.model, self.board, self.cache, self.occur, self._root(), a1)
        node = child
        for uci in ("g8f6", "f3g1"):
            node = _expand_child(self.model, self.board, self.cache, self.occur, node, _aid(uci))
        leaf = _expand_child(self.model, self.board, self.cache, self.occur, node, _aid("f6g8"))
        self.assertEqual(leaf.path, (_aid("g1f3"), _aid("g8f6"), _aid("f3g1"), _aid("f6g8")))
        feats = self.model.calls[-1][0]
        self.assertEqual(float(feats[783]), 1.0, "is1 应为 1（此前出现 1 次）")
        self.assertEqual(float(feats[784]), 0.0, "is2 应为 0（此前出现 <2 次）")

        # 根局面已计 2 次 → 回到该局面时 occurrence=2 → is2=1（post-increment 实现会错发 is2）
        self.model.calls.clear()
        occur2 = {_board_key(self.board): 2}
        child = _expand_child(self.model, self.board, self.cache, occur2, self._root(), a1)
        node = child
        for uci in ("g8f6", "f3g1"):
            node = _expand_child(self.model, self.board, self.cache, occur2, node, _aid(uci))
        _expand_child(self.model, self.board, self.cache, occur2, node, _aid("f6g8"))
        feats2 = self.model.calls[-1][0]
        self.assertEqual(float(feats2[783]), 0.0, "is1 应为 0（此前出现 ≥2 次）")
        self.assertEqual(float(feats2[784]), 1.0, "is2 应为 1（此前出现 ≥2 次）")

    def test_illegal_path_action_raises(self):
        bad_root = Node(legal=np.array([0], dtype=np.int64),
                        logits=np.zeros(1, dtype=np.float32), q=0.0,
                        path=(999999,))
        with self.assertRaises(RuntimeError):
            _expand_child(self.model, self.board, self.cache, self.occur, bad_root, _aid("e2e4"))

    def test_checkmate_child_terminal_q(self):
        board = chess.Board("7k/8/6K1/8/8/8/8/R7 w - - 0 1")
        occur = {_board_key(board): 1}
        a = _aid("a1a8")
        child = _expand_child(self.model, board, self.cache, occur, self._root(), a)
        self.assertTrue(child.terminal)
        self.assertEqual(child.path, (a,))
        self.assertAlmostEqual(child.q, -1.0)  # 将杀后行棋方（黑）视角 -1


if __name__ == "__main__":
    unittest.main()