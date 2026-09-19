"""Review round-3 回归：生成器/arena 与训练重放必须使用同一局面键（规格 §1.2/§2.4）。

`stateseq/data/sequences.py::_board_key` 是训练侧 occurrence 计数的权威口径
（棋子布置 + 走子方 + 易位权 + 合法过路兵）。生成器与 arena 必须复用同一函数，
不得再用"仅棋子布置"的 FEN 前缀哈希（会造成生成/训练重复位不一致）。
"""

from __future__ import annotations

import unittest

import chess

from stateseq.data.sequences import _board_key


class BoardKeyConsistencyTest(unittest.TestCase):
    def test_tool_modules_reuse_sequences_key(self):
        import tools.ssm_gumbel_arena as ar
        import tools.ssm_gumbel_selfplay as sp

        self.assertIs(sp._board_key, _board_key)
        self.assertIs(ar._board_key, _board_key)

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