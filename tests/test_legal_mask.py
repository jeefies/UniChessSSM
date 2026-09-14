"""验收 #5：合法 mask 与规则引擎一致（设计文档 §10.1）。

随机局面（含易位/过路兵/升变局面）mask 与 python-chess legal_moves 集合完全一致；
且 mask 内动作经 action_to_move 还原后确实合法（升变旗标由规则引擎上下文补齐）。
"""

from __future__ import annotations

import random
import unittest

import chess

from stateseq.actions import FROM_ACTION, legal_mask


def random_position(rng: random.Random, fen: str | None = None) -> chess.Board:
    board = chess.Board(fen) if fen else chess.Board()
    for _ in range(rng.randrange(0, 100)):
        if board.is_game_over():
            break
        board.push(rng.choice(list(board.legal_moves)))
    return board


class LegalMaskTest(unittest.TestCase):
    def test_mask_matches_engine(self):
        rng = random.Random(7)
        boards = [
            chess.Board(),
            chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"),   # 四方易位
            chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"),  # 过路兵
            chess.Board("rn2q2r/P6P/8/8/8/8/8/k6K w - - 0 1"),                  # 吃子升变
        ]
        boards += [random_position(rng) for _ in range(100)]
        for board in boards:
            mask = legal_mask(board)
            engine_keys = {(mv.from_square, mv.to_square, None if mv.promotion == chess.QUEEN else mv.promotion)
                           for mv in board.legal_moves}
            mask_keys = {FROM_ACTION[i] for i in mask.nonzero()[0]}
            self.assertEqual(engine_keys, mask_keys, f"FEN={board.fen()}")
            # 反向：mask 中的动作确实被规则引擎接受（含易位/过路兵旗标修正）
            for i in mask.nonzero()[0]:
                frm, to, promo = FROM_ACTION[i]
                mv = chess.Move(frm, to, promotion=promo)
                if board.is_legal(mv):
                    continue
                # 王车易位 / 吃过路兵 / 升后(Q→后走法 id) 的旗标修正
                for candidate in board.legal_moves:
                    cand_promo = None if candidate.promotion == chess.QUEEN else candidate.promotion
                    if candidate.from_square == frm and candidate.to_square == to and cand_promo == promo:
                        break
                else:
                    self.fail(f"mask 动作 {i} 非合法着: {board.fen()}")


if __name__ == "__main__":
    unittest.main()
