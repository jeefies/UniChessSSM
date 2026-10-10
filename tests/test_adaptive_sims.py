"""测试 KataGo 式自适应模拟预算（Adaptive Simulations）触发逻辑。
"""
import unittest
import chess
import numpy as np

from SSM.kit import check_adaptive_deep_trigger, _board_mat


class TestAdaptiveSims(unittest.TestCase):
    def test_check_trigger(self):
        # 1. 处于被将军局面 (黑王被白车在 e8 将军)
        board = chess.Board("4k3/8/8/8/8/8/8/4R2K b - - 0 1")
        self.assertTrue(board.is_check())
        triggered, reason = check_adaptive_deep_trigger(board, ply=40)
        self.assertTrue(triggered)
        self.assertEqual(reason, "check")

    def test_endgame_advantage_trigger(self):
        # 2. 残局局面：总子力 <= 16 (单王白单兵 vs 黑单王, mat=1 <= 16, 白方多 1 兵还不够，需要多 2 兵)
        # 白方多双兵 (mat=2 <= 16)
        board = chess.Board("4k3/8/8/8/8/4PP2/8/7K w - - 0 50")
        triggered, reason = check_adaptive_deep_trigger(board, ply=50)
        self.assertTrue(triggered)
        self.assertEqual(reason, "endgame_advantage")

        # 劣势方走子：黑方总子力 <= 16 但黑方落后 2 兵，黑方不触发优势残局深算
        board_black = chess.Board("4k3/8/8/8/8/4PP2/8/7K b - - 0 50")
        triggered, reason = check_adaptive_deep_trigger(board_black, ply=50)
        self.assertFalse(triggered)

    def test_opening_explore_trigger(self):
        # 3. 开局探索 (ply <= 12)
        board = chess.Board()
        class MockRng:
            def __init__(self, val): self.val = val
            def random(self): return self.val

        # 掷骰命中 (< 0.15)
        triggered, reason = check_adaptive_deep_trigger(board, ply=4, rng=MockRng(0.05), p_explore=0.15)
        self.assertTrue(triggered)
        self.assertEqual(reason, "opening_explore")

        # 掷骰未命中 (>= 0.15)
        triggered, reason = check_adaptive_deep_trigger(board, ply=4, rng=MockRng(0.50), p_explore=0.15)
        self.assertFalse(triggered)
        self.assertEqual(reason, "normal")

        # 超过 12 步不再进行开局随机探索
        triggered, reason = check_adaptive_deep_trigger(board, ply=15, rng=MockRng(0.05), p_explore=0.15)
        self.assertFalse(triggered)

    def test_normal_quiet_middlegame(self):
        # 4. 正常均势满盘中局 (mat > 16, 未被将军)
        board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5")
        triggered, reason = check_adaptive_deep_trigger(board, ply=15)
        self.assertFalse(triggered)
        self.assertEqual(reason, "normal")


if __name__ == "__main__":
    unittest.main()
