"""Review round-3 回归：生成器/arena 与训练重放必须使用同一局面键（规格 §1.2/§2.4）。

`stateseq/data/sequences.py::_board_key` 是训练侧 occurrence 计数的权威口径
（棋子布置 + 走子方 + 易位权 + 合法过路兵）。生成器与 arena 必须复用同一函数，
不得再用"仅棋子布置"的 FEN 前缀哈希（会造成生成/训练重复位不一致）。
"""

from __future__ import annotations

import os
import sys
import unittest

import chess

from stateseq.data.sequences import _board_key

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT_ROOT = os.environ.get("UNICHESS_KIT_ROOT", os.path.join(os.path.dirname(HERE), "Kit"))
if os.path.isdir(KIT_ROOT) and KIT_ROOT not in sys.path:
    sys.path.append(KIT_ROOT)  # 追加而非前插：Kit 的 tests 包不得遮蔽本仓库的 tests


class BoardKeyConsistencyTest(unittest.TestCase):
    def test_tool_modules_reuse_sequences_key(self):
        try:  # 自对弈与 arena 的 Player 都在 kit_adapter（需要 torch 与兄弟仓库 Kit）
            import stateseq.kit_adapter as ka
        except ImportError:
            self.skipTest("需要 torch 与兄弟仓库 Kit")
        self.assertIs(ka._board_key, _board_key)

    def test_equivalent_to_piece_map_key(self):
        """位棋盘键与原 piece_map 键的相等关系完全相同（随机对局 + 来回走子）。"""
        import random

        def slow(board):
            return (tuple(sorted(board.piece_map().items())), board.turn, board.castling_rights,
                    board.ep_square if board.has_legal_en_passant() else None)

        rng = random.Random(0)
        fast_to_slow: dict = {}
        slow_to_fast: dict = {}
        n = 0
        for g in range(60):
            board = chess.Board()
            for _ in range(120):
                f, s = _board_key(board), slow(board)
                self.assertEqual(fast_to_slow.setdefault(f, s), s)
                self.assertEqual(slow_to_fast.setdefault(s, f), f)
                n += 1
                moves = list(board.legal_moves)
                if not moves:
                    break
                if g % 2 and len(board.move_stack) >= 2 and rng.random() < 0.7:
                    last = board.move_stack[-2]
                    back = chess.Move(last.to_square, last.from_square)
                    if back in moves:
                        board.push(back)
                        continue
                board.push(rng.choice(moves))
        self.assertGreater(n, 5000)
        self.assertLess(len(fast_to_slow), n)       # 确有重复局面参与比较

    def test_cycle_returns_same_key(self):
        start = chess.Board()
        cycle = chess.Board()
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
            cycle.push_uci(uci)
        self.assertEqual(_board_key(start), _board_key(cycle))

    def test_turn_differences_split_key(self):
        start = chess.Board()
        black_to_move = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
        self.assertNotEqual(_board_key(start), _board_key(black_to_move))

    def test_castling_rights_split_key(self):
        start = chess.Board()
        no_castle = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")
        self.assertNotEqual(_board_key(start), _board_key(no_castle))

    def test_legal_ep_included(self):
        with_ep = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
        without_ep = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
        self.assertTrue(with_ep.has_legal_en_passant())
        self.assertNotEqual(_board_key(with_ep), _board_key(without_ep))


if __name__ == "__main__":
    unittest.main()