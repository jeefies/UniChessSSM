"""S 的 ReplayStore 不变量（``stateseq.kit_adapter``）：CPU 上用假 evaluator 验证送进网络的局面。

移植自原 arena 的 ``test_arena_expand`` / ``ArenaOpeningHistoryTest``（Review round-3 回归）：
1. 深度 ≥2 的叶子从根 cache 重放路径（不是把深层动作错套在根局面上），被评估局面
   == 根局面 + 路径，逐位相等；
2. occurrence 采用 encode-before-increment（与生成器/训练重放一致），根的计数计入；
3. 展开不修改根状态；
4. 开局 encode-before-move：B₀ 入 R，每个局面恰好步进一次；再次 choose 只补新局面。
原测试里「路径含非法动作要抛错」「将杀叶子取终局真值」两条在 kit 下由构造保证：
叶子路径来自 ``chess.Board.move_stack``（只能是合法着），终局在 kit Gumbel 调 Expander
之前判定（``Kit/tests/test_gumbel.py``）。
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
KIT_ROOT = os.environ.get("UNICHESS_KIT_ROOT", os.path.join(os.path.dirname(HERE), "Kit"))
if os.path.isdir(KIT_ROOT) and KIT_ROOT not in sys.path:
    sys.path.insert(0, KIT_ROOT)

try:
    import chess
    import numpy as np

    from stateseq import kit_adapter as ka
    from stateseq.adapter import encode_board
    from stateseq.data.sequences import _board_key
    from unichess_kit.api import GameStart, Leaf, SearchBudget
    from unichess_kit.runtime import run_sync
    from unichess_kit.search.gumbel import GumbelConfig

    _OK = True
except ImportError:  # pragma: no cover - 本机无 torch 或缺兄弟仓库 Kit
    _OK = False


class _FakeEvaluator:
    """记录每个负载（局面特征 + 输入 cache）；cache 是步进计数，便于核对重放链。"""

    model_key = "fake"

    def __init__(self):
        self.calls: list = []   # 每次 evaluate 一项：[(feats, cache_in), ...]

    def initial_cache(self):
        return 0

    def evaluate(self, payloads):
        self.calls.append([(p[0].copy(), p[4]) for p in payloads])
        return [(np.zeros(1936, np.float32), np.zeros(3, np.float32), p[4] + 1) for p in payloads]


def _feats(board: chess.Board, occurrence: int) -> "np.ndarray":
    return np.asarray(encode_board(board, occurrence)[0], dtype=np.float32).reshape(-1)


def _board(*ucis) -> "chess.Board":
    b = chess.Board()
    for u in ucis:
        b.push_uci(u)
    return b


@unittest.skipUnless(_OK, "需要 torch 与兄弟仓库 Kit")
class TestExpanderReplay(unittest.TestCase):
    def setUp(self):
        self.ev = _FakeEvaluator()
        self.exp = ka.SsmExpander(self.ev)

    def _root(self, board, count=1, cache=1):
        return ka.RootState(ply=len(board.move_stack), cache=cache,
                            occurrence={_board_key(board): count})

    def _expand(self, root, *ucis):
        leaf_board = _board(*ucis)
        return run_sync(self.exp.expand([Leaf(board=leaf_board, parent_handle=root,
                                              move=leaf_board.move_stack[-1])]))

    def test_depth1_and_depth2_positions(self):
        root = self._root(chess.Board(), cache=5)
        (ne,) = self._expand(root, "e2e4")
        self.assertEqual(len(self.ev.calls), 1)
        (feats, cache_in), = self.ev.calls[0]
        np.testing.assert_array_equal(feats, _feats(_board("e2e4"), 0))
        self.assertEqual(cache_in, 5, "叶子必须从根 cache 起步")
        self.assertIs(ne.handle, root)
        self.assertEqual(ne.moves, list(_board("e2e4").legal_moves))

        self.ev.calls.clear()
        self._expand(root, "e2e4", "e7e5")
        self.assertEqual(len(self.ev.calls), 2, "深度 2 = 重放 1 步 + 评估叶子")
        (f1, c1), = self.ev.calls[0]
        (f2, c2), = self.ev.calls[1]
        np.testing.assert_array_equal(f1, _feats(_board("e2e4"), 0))
        np.testing.assert_array_equal(f2, _feats(_board("e2e4", "e7e5"), 0))
        self.assertEqual((c1, c2), (5, 6), "重放必须沿路径链式推进 cache")

    def test_occurrence_encode_before_increment(self):
        cycle = ("g1f3", "g8f6", "f3g1", "f6g8")
        # 根局面已计 1 次；4 步循环回到该局面 → occurrence=1 → is1=1, is2=0
        root = self._root(chess.Board(), count=1)
        self._expand(root, *cycle)
        (feats, _), = self.ev.calls[-1]
        self.assertEqual(float(feats[783]), 1.0, "is1 应为 1（此前出现 1 次）")
        self.assertEqual(float(feats[784]), 0.0, "is2 应为 0（此前出现 <2 次）")
        np.testing.assert_array_equal(feats, _feats(_board(*cycle), 1))
        self.assertEqual(root.occurrence, {_board_key(chess.Board()): 1}, "展开不得修改根状态")

        # 根局面已计 2 次 → 回到该局面时 occurrence=2 → is2=1（post-increment 实现会错发 is2）
        self._expand(self._root(chess.Board(), count=2), *cycle)
        (feats2, _), = self.ev.calls[-1]
        self.assertEqual(float(feats2[783]), 0.0, "is1 应为 0（此前出现 ≥2 次）")
        self.assertEqual(float(feats2[784]), 1.0, "is2 应为 1（此前出现 ≥2 次）")

    def test_root_position_is_not_expandable(self):
        with self.assertRaises(ValueError):
            run_sync(self.exp.expand([Leaf(board=chess.Board(),
                                           parent_handle=self._root(chess.Board()))]))


@unittest.skipUnless(_OK, "需要 torch 与兄弟仓库 Kit")
class TestPlayerHistory(unittest.TestCase):
    def test_opening_positions_each_encoded_once(self):
        ev = _FakeEvaluator()
        player = ka.SsmPlayer("S", ev, GumbelConfig(simulations=4, m0=2, g=0.0))
        run_sync(player.new_game(GameStart(color=chess.BLACK, seed=3)))
        opening = ("e2e4", "e7e5", "g1f3")
        board = _board(*opening)
        run_sync(player.choose(board.copy(), SearchBudget()))

        # 追赶：B₀..B₃ 各一次单负载步进，按序、encode-before-increment、cache 链式
        replay, occ = chess.Board(), {}
        for k in range(len(opening) + 1):
            if k:
                replay.push_uci(opening[k - 1])
            (feats, cache_in), = ev.calls[k]
            key = _board_key(replay)
            np.testing.assert_array_equal(feats, _feats(replay, occ.get(key, 0)))
            self.assertEqual(cache_in, k)
            occ[key] = occ.get(key, 0) + 1
        self.assertEqual(player.stepped, len(opening) + 1)
        self.assertEqual(player.occurrence, occ)

        # 同一局面再 choose 不重复步进：评估序列 == 第一次去掉追赶部分（g=0、同 rng 种子）
        first_search = ev.calls[len(opening) + 1:]
        n = len(ev.calls)
        run_sync(player.choose(board.copy(), SearchBudget()))
        again = ev.calls[n:]
        self.assertEqual(len(again), len(first_search))
        for x, y in zip(again, first_search):
            self.assertEqual([c for _, c in x], [c for _, c in y])
            for (fx, _), (fy, _) in zip(x, y):
                np.testing.assert_array_equal(fx, fy)
        self.assertEqual(player.stepped, len(opening) + 1)
        # 前进两着后只补这两个新局面
        board.push_uci("b8c6")
        board.push_uci("f1b5")
        n = len(ev.calls)
        run_sync(player.choose(board.copy(), SearchBudget()))
        (f5, c5), = ev.calls[n]
        (f6, c6), = ev.calls[n + 1]
        np.testing.assert_array_equal(f5, _feats(_board(*opening, "b8c6"), 0))
        np.testing.assert_array_equal(f6, _feats(_board(*opening, "b8c6", "f1b5"), 0))
        self.assertEqual((c5, c6), (4, 5))
        self.assertEqual(player.stepped, len(opening) + 3)


if __name__ == "__main__":
    unittest.main()
