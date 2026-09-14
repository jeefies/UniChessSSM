"""验收 #2：特征往返（设计文档 §10.1）。

随机合法局面 B → 785 维特征 → 手工解码一致（棋子布置/走子方/易位权/过路兵/计数/重复标记）。
"""

from __future__ import annotations

import random
import unittest

import chess

from stateseq.features import FEATURE_DIM, decode, encode


def random_position(rng: random.Random) -> chess.Board:
    board = chess.Board()
    for _ in range(rng.randrange(0, 120)):
        if board.is_game_over():
            break
        board.push(rng.choice(list(board.legal_moves)))
    return board


class FeatureRoundtripTest(unittest.TestCase):
    def test_dim(self):
        self.assertEqual(FEATURE_DIM, 785)

    def test_roundtrip_random(self):
        rng = random.Random(42)
        for i in range(200):
            board = random_position(rng)
            occ = rng.choice([0, 1, 2, 3])
            fields = decode(encode(board, occurrence=occ))
            # 棋子布置
            self.assertEqual(
                {sq: (p.color == chess.WHITE, p.piece_type) for sq, p in board.piece_map().items()},
                {sq: (w, pt) for sq, (w, pt) in fields.pieces.items()},
            )
            # 走子方 / 易位权
            self.assertEqual(fields.side_white, board.turn == chess.WHITE)
            self.assertEqual(
                fields.castling,
                (
                    board.has_kingside_castling_rights(chess.WHITE),
                    board.has_queenside_castling_rights(chess.WHITE),
                    board.has_kingside_castling_rights(chess.BLACK),
                    board.has_queenside_castling_rights(chess.BLACK),
                ),
            )
            # 半回合计数（/100 精度内；>100 截断）
            self.assertAlmostEqual(fields.halfmove, min(board.halfmove_clock, 100), delta=1)
            self.assertGreaterEqual(fields.fullmove, 1)

    def test_ep_file(self):
        """过路兵 file one-hot 仅在存在合法吃过路兵时非零。"""
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
        fields = decode(encode(board))
        self.assertEqual(fields.ep_file, 3)  # d 列
        # 无过路兵局面全零
        board2 = chess.Board()
        self.assertIsNone(decode(encode(board2)).ep_file)

    def test_repetition_flags(self):
        board = chess.Board()
        self.assertFalse(decode(encode(board, 0)).rep_is1)
        self.assertTrue(decode(encode(board, 1)).rep_is1)
        self.assertTrue(decode(encode(board, 2)).rep_is2)


if __name__ == "__main__":
    unittest.main()
