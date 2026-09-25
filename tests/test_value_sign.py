"""验收 #7：价值符号与 moves_left 方向的单元测试（设计文档 §10.1）。

- 结果对行棋方归一：白胜棋中白走步 label=胜(0)，同一局面黑走步 label=负(2)；
- moves_left 沿棋谱单调递减（ply 口径，截断 200）；
- 和棋全部为 DRAW(1)。
"""

from __future__ import annotations
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_IMPORT_ROOT = _os.path.dirname(_HERE)   # import 根：~/UniChess：SSM 与 Kit 都是它的顶层包
HERE = _HERE
KIT_ROOT = _os.environ.get("UNICHESS_KIT_ROOT", _os.path.join(_IMPORT_ROOT, "Kit"))
if _IMPORT_ROOT not in _sys.path:
    _sys.path.insert(0, _IMPORT_ROOT)
if _os.path.isdir(KIT_ROOT) and KIT_ROOT not in _sys.path:
    _sys.path.append(KIT_ROOT)   # 追加而非前插：Kit 的 tests 包不得遮蔽本仓库的 tests


import io
import unittest

import chess.pgn

from SSM.dataset.sequences import T_MAX, game_to_sequence
from SSM.model.losses import RESULT_DRAW, RESULT_LOSS, RESULT_WIN

PGN_DECISIVE = """[Event "t"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "1900"]
[TimeControl "300+3"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
"""

PGN_DRAW = """[Event "t"]
[Result "1/2-1/2"]

1. e4 e5 2. Nf3 Nc6 3. Nc3 Nf6 1/2-1/2
"""


def _seq(pgn_text: str):
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    meta = {
        "elo_mean": 1950.0,
        "time_control": "300+3",
        "result": game.headers["Result"],
        "variant": "Standard",
    }
    return game_to_sequence(game, meta)


class ValueSignTest(unittest.TestCase):
    def test_result_normalization(self):
        seq = _seq(PGN_DECISIVE)
        self.assertEqual(len(seq), 6)
        # 白走步（偶数 ply）：胜=0；黑走步：负=2
        for t, rec in enumerate(seq):
            if t % 2 == 0:
                self.assertEqual(rec.result, RESULT_WIN)
                self.assertEqual(rec.color, 1)
            else:
                self.assertEqual(rec.result, RESULT_LOSS)
                self.assertEqual(rec.color, 0)

    def test_draw(self):
        seq = _seq(PGN_DRAW)
        self.assertTrue(all(rec.result == RESULT_DRAW for rec in seq))

    def test_moves_left_direction(self):
        seq = _seq(PGN_DECISIVE)
        ml = [rec.moves_left for rec in seq]
        self.assertEqual(ml, [6, 5, 4, 3, 2, 1])
        self.assertTrue(all(m <= T_MAX for m in ml))

    def test_mlh_loss_log_target(self):
        """验证 mlh_loss 的 log_target=True 支持及其变换行为。"""
        import torch
        import torch.nn.functional as F
        from SSM.model.losses import mlh_loss

        pred = torch.tensor([0.0, 5.0, 10.0, 50.0], dtype=torch.float32)
        target = torch.tensor([0.0, 4.0, 15.0, 60.0], dtype=torch.float32)
        # 默认模式 delta=1.0
        l_default = mlh_loss(pred, target, log_target=False)
        expected_default = F.huber_loss(pred, target, delta=1.0, reduction="mean")
        self.assertAlmostEqual(float(l_default), float(expected_default), places=6)

        # log_target 模式：log1p(relu), delta=0.5
        l_log = mlh_loss(pred, target, log_target=True)
        pred_log = torch.log1p(F.relu(pred))
        target_log = torch.log1p(F.relu(target))
        expected_log = F.huber_loss(pred_log, target_log, delta=0.5, reduction="mean")
        self.assertAlmostEqual(float(l_log), float(expected_log), places=6)

    def test_long_game_cap(self):
        """构造 250 ply 长局：moves_left 截断 200。"""
        import random

        rng = random.Random(1)
        for _attempt in range(50):
            board = chess.Board()
            sans = []
            while len(sans) < 240 and not board.is_game_over():
                mv = rng.choice(list(board.legal_moves))
                sans.append(board.san(mv))
                board.push(mv)
            if len(sans) >= 201:
                break
        else:  # pragma: no cover
            self.fail("未能随机出 201+ ply 长局")
        # 未到自然终局但按实际计为和棋（abandoned 口径，§4.1）
        pgn_text = '[Result "1/2-1/2"]\n\n' + " ".join(sans) + " 1/2-1/2"
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        meta = {"elo_mean": None, "time_control": None, "result": "1/2-1/2", "variant": "Standard"}
        seq = game_to_sequence(game, meta)
        self.assertEqual(seq[0].moves_left, T_MAX)
        self.assertTrue(all(r.elo_missing for r in seq))


if __name__ == "__main__":
    unittest.main()
