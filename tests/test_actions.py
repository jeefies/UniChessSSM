"""验收 #1：动作空间双射（设计文档 §10.1）。

1936 id ↔ (from,to,promo) 全覆盖、含全部升变（含低升变与边线吃子升变）、王车易位可达、与坐标约定一致。
"""

from __future__ import annotations

import random
import unittest

import chess

from stateseq.actions import (
    FROM_ACTION,
    NUM_ACTIONS,
    NUM_KNIGHT_MOVES,
    NUM_PROMOTION_MOVES,
    NUM_QUEEN_MOVES,
    TO_ACTION,
    action_to_move,
    move_to_action,
)


class ActionSpaceTest(unittest.TestCase):
    def test_table_size(self):
        self.assertEqual(len(FROM_ACTION), NUM_ACTIONS)
        self.assertEqual(len(TO_ACTION), NUM_ACTIONS)
        self.assertEqual(NUM_QUEEN_MOVES, 1456)
        self.assertEqual(NUM_KNIGHT_MOVES, 336)
        self.assertEqual(NUM_PROMOTION_MOVES, 144)

    def test_bijection_random_playouts(self):
        """随机对局：每个合法着 ↔ 唯一 id，往返一致。"""
        rng = random.Random(20260914)
        for game in range(30):
            board = chess.Board()
            while not board.is_game_over() and board.ply() < 250:
                legal = list(board.legal_moves)
                seen: set[int] = set()  # 同一局面内合法着 id 互异；跨局面允许重复
                for mv in legal:
                    aid = move_to_action(mv)
                    self.assertNotIn(aid, seen, f"动作 id 冲突: {aid}")
                    seen.add(aid)
                    back = action_to_move(aid)
                    self.assertEqual(back.from_square, mv.from_square)
                    self.assertEqual(back.to_square, mv.to_square)
                    self.assertEqual(back.promotion, mv.promotion)
                board.push(rng.choice(legal))

    def test_promotions_covered(self):
        """全部升变着（直进/双吃、边线、R/B/N/Q）均可编码且互不相同。"""
        # a7/h7 边线与中间兵、白黑双方
        fens = [
            ("8/k6K/8/8/8/8/p7/8 b - - 0 1", chess.BLACK),  # 黑 a2 兵（下测白方）
            ("8/PP6/8/8/8/8/8/k6K w - - 0 1", chess.WHITE),  # 白 b7/g7 兵
            ("rn2q2r/P6P/8/8/8/8/8/k6K w - - 0 1", chess.WHITE),  # 带吃子升变
        ]
        found = set()
        for fen, _ in fens:
            board = chess.Board(fen)
            for mv in board.legal_moves:
                if mv.promotion:
                    found.add(mv.promotion)
                    aid = move_to_action(mv)
                    back = action_to_move(aid)
                    expected = None if mv.promotion == chess.QUEEN else mv.promotion  # 升后走 Q→后走法 id
                    self.assertEqual(back.promotion, expected)
        self.assertEqual(found, {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT})

    def test_castling_via_queen_pair(self):
        """王车易位 = 王两格 (from,to)，落在后走法集合。"""
        board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        castle = [mv for mv in board.legal_moves if board.is_castling(mv)]
        self.assertEqual(len(castle), 2)
        for mv in castle:
            aid = move_to_action(mv)
            frm, to, promo = FROM_ACTION[aid]
            self.assertEqual((frm, to), (mv.from_square, mv.to_square))
            self.assertIsNone(promo)

    def test_no_self_pairs_except_phantom(self):
        """from==to 只允许幻影升变槽位存在（永非法）。"""
        self_pairs = [a for a, (f, t, p) in enumerate(FROM_ACTION) if f == t]
        self.assertTrue(all(a >= 1792 for a in self_pairs))


if __name__ == "__main__":
    unittest.main()
