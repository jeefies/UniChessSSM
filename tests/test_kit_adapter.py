"""S 的 kit 接入（``SSM.kit``）单元测试。需要 CUDA 与兄弟仓库 Kit。

切换前与 S arena 原版 ``play_one_game`` 着法逐个相同、扩展深度直方图逐位相同（随机初始化
模型、并发 1；真实权重见 git 历史中的 ``tools/kit_arena_parity.py``）。原版已删除，这里锁死：
1. 同输入的对局完全可复现（g=0），逐步 info 里的直方图之和 == Player 累计直方图；
2. 前向数 = 追赶步进 + 重放 + 叶子评估，落在由直方图推出的区间内（kit 的 Gumbel 在调
   Expander 之前判终局，终局叶子不重放路径）；
3. 懒追赶与逐 ply 步进等价；多叶子同步重放与逐个展开等价；并发批量对弈可跑通。
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


import os
import sys
import unittest


try:
    import torch
    import chess
    import numpy as np

    _HAS_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover - 本机（Windows）无 torch
    _HAS_CUDA = False

from SSM.kit import hist_merge, hist_summary  # noqa: E402

try:
    import Kit  # noqa: F401

    _HAS_KIT = True
except ImportError:  # pragma: no cover
    _HAS_KIT = False


@unittest.skipUnless(_HAS_CUDA and _HAS_KIT, "需要 CUDA（mamba 步进核）与兄弟仓库 Kit")
class TestKitAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import SSM.kit as ka
        from SSM.model import SeqModel

        cls.ka = ka
        cls.seqs = {}
        for seed in (1, 2):
            torch.manual_seed(seed)
            cls.seqs[seed] = SeqModel(dropout=0.0).to("cuda").eval()

    # ---- 构造 ----

    def _evaluator(self, seed):
        return self.ka.SsmEvaluator(self.seqs[seed], "cuda", f"S:rand{seed}")

    def _factory(self, seed, n_sims, m0):
        from Kit.search.gumbel import GumbelConfig
        return self.ka.SsmPlayerFactory(f"S{seed}", self._evaluator(seed),
                                        GumbelConfig(simulations=n_sims, m0=m0, g=0.0))

    def _kit_game(self, fa, fb, opening_uci, a_is_white, max_plies, seed=7, observer=None):
        from Kit.api import SearchBudget
        from Kit.pipelines.match import GameTask, play_game
        from Kit.rules.referee import StandardReferee
        from Kit.runtime import run_sync

        task = GameTask(game=0, pair=0, a_is_white=a_is_white, opening=tuple(opening_uci),
                        seed_a=seed, seed_b=seed)
        referee = StandardReferee(max_plies=len(opening_uci) + max_plies)
        self._last_players = (fa(), fb())
        return run_sync(play_game(task, *self._last_players, referee, SearchBudget(),
                                  observer=observer))

    # ---- 测试 ----

    def test_game_reproducible_and_forward_accounting(self):
        cases = [(("e2e4", "e7e5"), True, 12, 4, 8),
                 (("d2d4", "g8f6", "c2c4"), False, 16, 16, 6),
                 ((), True, 8, 2, 5)]
        for uci, a_white, n_sims, m0, plies in cases:
            with self.subTest(opening=uci, a_is_white=a_white):
                runs = []
                for _ in range(2):
                    fa, fb = self._factory(1, n_sims, m0), self._factory(2, n_sims, m0)
                    events = []
                    rec = self._kit_game(fa, fb, uci, a_white, plies, observer=events.append)
                    hist = hist_merge(self._last_players[0].expand_hist,
                                      self._last_players[1].expand_hist)
                    n_fwd = fa.evaluator.n_forwards + fb.evaluator.n_forwards
                    runs.append((rec["moves"], hist, n_fwd))

                    from_info: list = []
                    for ev in events:
                        if ev["type"] == "move":
                            from_info = hist_merge(from_info, ev["info"]["expand_hist"])
                    self.assertEqual(from_info, hist)
                    self.assertEqual(len(rec["moves"]), plies)
                    total = len(uci) + plies
                    s = hist_summary(hist)
                    # 追赶：每方步进到自己最后一次行棋的局面（B_0..），合计 ≤ 2·total
                    self.assertGreaterEqual(n_fwd, total + s["replay_forwards"])
                    self.assertLessEqual(n_fwd, 2 * total + s["replay_forwards"] + s["expansions"])
                self.assertEqual(runs[0], runs[1], "g=0 下同输入对局必须完全可复现")

    def test_catch_up_equals_per_ply_stepping(self):
        """懒追赶（choose 时一次补齐）与逐 ply 步进得到同一 cache 与根评估。"""
        from Kit.api import GameStart, SearchBudget
        from Kit.runtime import run_sync

        ev = self._evaluator(1)
        board = chess.Board()
        for san in ("e4", "e5", "Nf3", "Nc6", "Bb5"):
            board.push_san(san)
        # 参考：逐局面步进（encode-before-increment）
        from SSM.dataset.sequences import _board_key
        cache, occ, out = ev.initial_cache(), {}, None
        replay = chess.Board()
        for k in range(len(board.move_stack) + 1):
            if k:
                replay.push(board.move_stack[k - 1])
            key = _board_key(replay)
            (out,) = ev.evaluate([self.ka.encode_payload(replay, occ.get(key, 0), cache)])
            cache = out[2]
            occ[key] = occ.get(key, 0) + 1

        p = self._factory(1, 4, 2)()
        run_sync(p.new_game(GameStart(color=chess.BLACK)))
        run_sync(p.choose(board.copy(), SearchBudget()))
        self.assertEqual(p.stepped, len(board.move_stack) + 1)
        self.assertEqual(p.occurrence, occ)
        np.testing.assert_array_equal(p.last[0], out[0])
        for (c1, s1), (c2, s2) in zip(p.cache, cache):
            self.assertTrue(torch.equal(c1, c2) and torch.equal(s1, s2))

    def test_expander_multi_leaf_lockstep(self):
        """多叶子同步重放与逐个展开一致（批大小不同，只要求浮点级接近）。"""
        from Kit.api import Leaf
        from Kit.runtime import run_sync

        ev = self._evaluator(2)
        exp = self.ka.SsmExpander(ev)
        root_board = chess.Board()
        (out,) = ev.evaluate([self.ka.encode_payload(root_board, 0, ev.initial_cache())])
        from SSM.dataset.sequences import _board_key
        root = self.ka.RootState(ply=0, cache=out[2], occurrence={_board_key(root_board): 1})
        lines = [("e2e4",), ("d2d4", "d7d5"), ("g1f3", "g8f6", "c2c4")]
        leaves = []
        for line in lines:
            b = root_board.copy()
            for u in line:
                b.push_uci(u)
            leaves.append(Leaf(board=b, parent_handle=root, move=b.move_stack[-1]))
        n0 = ev.n_forwards
        together = run_sync(exp.expand(leaves))
        self.assertEqual(ev.n_forwards - n0, sum(len(x) for x in lines))
        for leaf, ne in zip(leaves, together):
            (single,) = run_sync(exp.expand([leaf]))
            self.assertEqual(ne.moves, single.moves)
            self.assertEqual(ne.moves, list(leaf.board.legal_moves))
            np.testing.assert_allclose(ne.logits, single.logits, atol=1e-3)
            self.assertAlmostEqual(ne.value, single.value, places=3)
            self.assertIs(ne.handle, root)
        # 根状态不可被展开修改
        self.assertEqual(root.occurrence, {_board_key(root_board): 1})

    def test_expander_rejects_missing_root(self):
        from Kit.api import Leaf
        from Kit.runtime import run_sync

        exp = self.ka.SsmExpander(self._evaluator(1))
        with self.assertRaises(ValueError):
            run_sync(exp.expand([Leaf(board=chess.Board())]))

    def test_run_match_concurrent(self):
        """并发 4 的 kit 批量对弈能跑通（批大小变化 → 只验证合法与计数，不验逐位）。"""
        import tempfile
        from Kit.pipelines.match import MatchConfig, run_match

        fa, fb = self._factory(1, 8, 4), self._factory(2, 8, 4)
        with tempfile.TemporaryDirectory() as d:
            cfg = MatchConfig(pairs=2, seed=3, max_plies=14, concurrency=4, openings="bundled")
            summary = run_match(cfg, make_a=fa, make_b=fb, out_path=os.path.join(d, "r.jsonl"))
        self.assertEqual(summary["games"], 4)


if __name__ == "__main__":
    unittest.main()
