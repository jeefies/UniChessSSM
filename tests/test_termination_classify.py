"""回归：终局裁决口径必须与对局循环退出条件（claim_draw=True）一致。

背景 bug（2026-09-19 实测）：生成器用 `is_repetition(3)` / `is_fifty_moves()`（严格判定）
分类，而对局循环用 `is_game_over(claim_draw=True)`（含"下一着可申和"）退出，差一 ply →
规则申和局全部掉进兜底分支被记为"300 ply 封顶截断"。gen2k 前 400 局中 173/218 条如此。

arena/生成器开局的 encode-before-move 口径（B₀ 入 R、无重复步进）见 ``tests/test_kit_replay.py``。
"""

from __future__ import annotations

import os
import sys
import unittest

import chess
import numpy as np

from stateseq.adapter import classify_final_board
from stateseq.gumbel import TERM_CODES

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT_ROOT = os.environ.get("UNICHESS_KIT_ROOT", os.path.join(os.path.dirname(HERE), "Kit"))
if os.path.isdir(KIT_ROOT) and KIT_ROOT not in sys.path:
    sys.path.append(KIT_ROOT)  # 追加而非前插：Kit 的 tests 包不得遮蔽本仓库的 tests


def _claimable_threefold_board() -> chess.Board:
    """马来回走 7 个半回合：此时"下一着可申和"成立，但严格三次重复尚未成立。"""
    board = chess.Board()
    for san in ("Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1"):
        board.push_san(san)
    return board


class TerminationClassifyTest(unittest.TestCase):
    def test_claimable_threefold_is_not_truncated(self):
        board = _claimable_threefold_board()
        # 这正是 bug 触发条件：循环认为终局，严格判定认为没有
        self.assertTrue(board.is_game_over(claim_draw=True))
        self.assertFalse(board.is_repetition(3))

        result, reason, is_truncated = classify_final_board(board)
        self.assertEqual(reason, "threefold")
        self.assertFalse(is_truncated)
        self.assertEqual(result, 1)  # 和棋（白视角）
        self.assertIn(reason, TERM_CODES)

    def test_unfinished_game_is_truncated(self):
        board = chess.Board()
        board.push_san("e4")
        result, reason, is_truncated = classify_final_board(board)
        self.assertEqual(reason, "truncated")
        self.assertTrue(is_truncated)
        self.assertEqual(result, 1)

    def test_checkmate_result_is_white_perspective(self):
        board = chess.Board()
        for san in ("f3", "e5", "g4", "Qh4#"):
            board.push_san(san)
        result, reason, is_truncated = classify_final_board(board)
        self.assertEqual(reason, "checkmate")
        self.assertFalse(is_truncated)
        self.assertEqual(result, 2)  # 黑胜

    def test_stalemate_and_insufficient_material(self):
        stale = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual(classify_final_board(stale)[1], "stalemate")
        bare = chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 0 1")
        self.assertEqual(classify_final_board(bare)[1], "insufficient_material")

    def test_generator_sink_uses_same_verdict(self):
        """生成器落盘（kit_adapter.V3Sink）的终局字段必须与 classify_final_board 同口径。"""
        try:
            from stateseq.kit_adapter import V3Sink
            from unichess_kit.api import MoveDecision
        except ImportError:
            self.skipTest("需要 torch 与兄弟仓库 Kit")

        class _Writer:
            def add(self, meta, actions, pipol, poff):
                self.meta, self.actions = meta, actions

        board = _claimable_threefold_board()
        decisions = [MoveDecision(m, info={"pi_ids": np.zeros(1, np.uint16),
                                           "pi": np.ones(1, np.float32)})
                     for m in board.move_stack]
        writer = _Writer()
        sink = V3Sink(writer, gen_id=2)
        sink.on_game_end({"game": 0, "book_plies": 0}, board, decisions)
        self.assertEqual(TERM_CODES[int(writer.meta["termination_reason"])], "threefold")
        self.assertEqual(int(writer.meta["is_truncated"]), 0)
        self.assertEqual(int(writer.meta["result"]), 1)
        self.assertEqual(int(writer.meta["n_plies"]), len(board.move_stack))
        self.assertEqual(sink.truncated_games, 0)


if __name__ == "__main__":
    unittest.main()
