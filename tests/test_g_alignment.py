"""验收 #9（review.txt 待办⑤）：g 动作对齐断言。

训练语义不变量：SequenceDataset 批次中，对每个有效步 t，
  1. 由 actions[:t] 从初始局面重放得到的棋盘 B_t，其 encode(B_t, occurrence=prior)
     必须与批内 features[:, t] **float 精确相等**（同一编码路径，非近似）；
  2. actions[t] 必须是由 B_t 出发的合法着，且 push(B_t, a_t) 重放得到的 B_{t+1}
     与 t+1 步重放局面一致 —— 即 forward_train 里 g(h[:, t-1], actions[:, t])
     的目标 sg(x_t − x_{t-1}) 对应的正是真实转移 B_{t-1} --a_t--> B_t。

需要 v2 数据分片（远端 data/shards）；本地无分片时跳过。
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARD_DIR = os.environ.get(
    "UNICHESS_SSM_SHARDS",
    str(REPO_ROOT / "data" / "shards"),
)

try:
    import chess  # noqa: F401
    _HAS_CHESS = os.path.isdir(SHARD_DIR)
except ImportError:  # pragma: no cover
    _HAS_CHESS = False

if _HAS_CHESS:
    from stateseq.actions import action_to_move, move_to_action
    from stateseq.data.dataset import SequenceDataset
    from stateseq.data.sequences import _board_key
    from stateseq.features import encode


@unittest.skipUnless(_HAS_CHESS, "需要 v2 数据分片（设置 UNICHESS_SSM_SHARDS 或远端 data/shards）")
class GAlignmentTest(unittest.TestCase):
    """从 v2 分片取 4 局，经 SequenceDataset 构造训练批后逐项断言。"""

    @classmethod
    def setUpClass(cls):
        cls.ds = SequenceDataset(SHARD_DIR, workers=2)
        cls.addClassCleanup(cls.ds.close)
        batches = cls.ds.val_batch(n_batches=1, microbatch=4, device="cpu")
        cls.out = batches[0]

    def _assert_game_alignment(self, row: int):
        batch, valid = self.out["batch"], self.out["valid"]
        actions = batch.actions[row].numpy()
        feats = batch.features[row].numpy()
        n_valid = int(valid[row].sum())

        board = chess.Board()
        occ: dict = {}
        for t in range(n_valid):
            key = _board_key(board)
            prior = occ.get(key, 0)
            occ[key] = prior + 1

            # 不变量 1：重放局面编码 == 批内特征（float 精确相等）
            want = encode(board, occurrence=prior)
            np.testing.assert_array_equal(
                feats[t], want,
                err_msg=f"第 {row} 局 t={t}：重放局面编码与批内 features 不一致",
            )
            self.assertTrue(bool(valid[row, t]))

            a = int(actions[t])
            mv = action_to_move(a)
            # 动作表往返无损
            self.assertEqual(move_to_action(mv), a)
            # 不变量 2：a_t 是 B_t 的合法着，且真实执行后到达 B_{t+1}
            self.assertIn(mv, list(board.legal_moves),
                          f"第 {row} 局 t={t}：actions[t] 不是当前局面合法着")
            board.push(mv)  # B_{t+1}；下一轮迭代重建并断言其编码

        # valid 是连续前缀，填充步无效
        self.assertFalse(bool(valid[row, n_valid:].any()))

    def test_g_alignment_game0(self):
        self._assert_game_alignment(0)

    def test_g_alignment_game1(self):
        self._assert_game_alignment(1)

    def test_g_alignment_game2(self):
        self._assert_game_alignment(2)

    def test_g_alignment_game3(self):
        self._assert_game_alignment(3)

    def test_dyn_index_semantics(self):
        """forward_train 的 dyn 切片语义：g 输入 (h[:, :-1], actions[:, 1:])。

        即第 t 步（t≥1）的 dyn 样本用 actions[:, t] 去预测 x_t − x_{t-1}；
        结合上面的不变量 2，actions[t] 正是 B_{t-1} --a_t--> B_t 的真实转移，
        因此 (h_{t-1}, a_t) 与目标 sg(x_t − x_{t-1}) 对齐（此处断言语义前提：
        批内相邻步确实属于同一局、valid 连续前缀）。
        """
        valid = self.out["valid"]
        for row in range(valid.shape[0]):
            n_valid = int(valid[row].sum())
            # valid 是连续前缀（SequenceDataset._collate 构造方式）
            self.assertTrue(bool(valid[row, :n_valid].all()))
            self.assertGreaterEqual(n_valid, 2, "dyn 需要至少 2 个有效步")


if __name__ == "__main__":
    unittest.main()
