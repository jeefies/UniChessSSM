"""回归：终局裁决口径必须与对局循环退出条件（claim_draw=True）一致。

背景 bug（2026-09-19 实测）：生成器用 `is_repetition(3)` / `is_fifty_moves()`（严格判定）
分类，而对局循环用 `is_game_over(claim_draw=True)`（含"下一着可申和"）退出，差一 ply →
规则申和局全部掉进兜底分支被记为"300 ply 封顶截断"。gen2k 前 400 局中 173/218 条如此。

另覆盖 arena 开局库的 encode-before-move 口径（B₀ 必须入 R、且无局面重复步进）。
"""

from __future__ import annotations

import unittest

import chess
import numpy as np
import torch

from stateseq.adapter import classify_final_board
from stateseq.gumbel import TERM_CODES


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

    def test_generator_result_uses_same_verdict(self):
        """生成器 GameState.result() 必须与 classify_final_board 同口径。"""
        from tools.ssm_gumbel_selfplay import GameState

        game = GameState.__new__(GameState)  # 不触发模型加载
        game.board = _claimable_threefold_board()
        result, code, is_truncated = game.result()
        self.assertEqual(TERM_CODES[code], "threefold")
        self.assertFalse(is_truncated)
        self.assertEqual(result, 1)


class _RecordingModel:
    """最小 ArenaModel 替身：记录每次 step 的 785 维特征。"""

    device = "cpu"

    def __init__(self):
        self.ckpt_path = "fake"
        self.feats: list[np.ndarray] = []

    def initial_cache(self, b: int = 1):
        return [(torch.zeros(1), torch.zeros(1))]

    def step(self, feats, tc, elo, color, cache):
        self.feats.append(np.asarray(feats, dtype=np.float32).reshape(-1).copy())
        logits = np.zeros((1, 1936), dtype=np.float32)
        wdl = np.zeros((1, 3), dtype=np.float32)
        n = cache[0][0] + 1.0
        return logits, wdl, np.zeros((1, 1), np.float32), np.zeros((1, 512), np.float32), [(n, n)]


class ArenaOpeningHistoryTest(unittest.TestCase):
    """开局库必须 encode-before-move：B₀ 入 R，且没有局面被重复步进。"""

    def test_opening_positions_each_encoded_once(self):
        from stateseq.adapter import encode_board
        from tools.ssm_gumbel_arena import _board_key, play_one_game

        cfg = type("cfg", (), {"n_sims": 2, "m0": 2, "max_plies": 0,
                               "c_visit": 50.0, "c_scale": 1.0})()
        mw, mb = _RecordingModel(), _RecordingModel()
        opening = "e4 e5 Nf3"
        play_one_game(mw, mb, cfg, opening_san=opening, opening_id=0)

        # max_plies=0 → 只跑开局；双方各步进 len(tokens) 次
        tokens = opening.split()
        self.assertEqual(len(mw.feats), len(tokens))

        expected = []
        board = chess.Board()
        occ: dict = {}
        for san in tokens:
            key = _board_key(board)
            expected.append(encode_board(board, occ.get(key, 0))[0])
            occ[key] = occ.get(key, 0) + 1
            board.push_san(san)
        for got, want in zip(mw.feats, expected):
            np.testing.assert_array_equal(got, want)


if __name__ == "__main__":
    unittest.main()
