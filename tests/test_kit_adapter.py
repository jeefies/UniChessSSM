"""S 的 kit 接入（``stateseq.kit_adapter``）单元测试。需要 CUDA 与兄弟仓库 Kit。

核心断言：kit ``play_game`` 驱动两个 ``SsmPlayer`` 下出的棋，与 S arena 原版
``play_one_game`` **着法逐个相同**（随机初始化模型、并发 1），扩展深度直方图逐位相同。
前向数 ≤ 原版 − 1：懒追赶省掉最后一步非行棋方的 1 次步进；kit 的 Gumbel 在调 Expander
之前就判终局，终局叶子不再重放路径（原版先重放 d−1 步才发现终局），省下的正是这部分。
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
KIT_ROOT = os.environ.get("UNICHESS_KIT_ROOT", os.path.join(os.path.dirname(HERE), "Kit"))
if os.path.isdir(KIT_ROOT) and KIT_ROOT not in sys.path:
    sys.path.insert(0, KIT_ROOT)

try:
    import torch
    import chess
    import chess.pgn
    import numpy as np

    _HAS_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover - 本机（Windows）无 torch
    _HAS_CUDA = False

from stateseq.depth_hist import hist_merge, hist_summary  # noqa: E402

try:
    import unichess_kit  # noqa: F401

    _HAS_KIT = True
except ImportError:  # pragma: no cover
    _HAS_KIT = False


def _load_tool():
    path = os.path.join(HERE, "tools", "ssm_gumbel_arena.py")
    spec = importlib.util.spec_from_file_location("ssm_gumbel_arena_kit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pgn_ucis(pgn: str) -> list:
    game = chess.pgn.read_game(io.StringIO(pgn))
    return [m.uci() for m in game.mainline_moves()]


@unittest.skipUnless(_HAS_CUDA and _HAS_KIT, "需要 CUDA（mamba 步进核）与兄弟仓库 Kit")
class TestKitAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from stateseq import kit_adapter as ka
        from stateseq.model import SeqModel

        cls.ka = ka
        cls.tool = _load_tool()
        cls.seqs = {}
        for seed in (1, 2):
            torch.manual_seed(seed)
            cls.seqs[seed] = SeqModel(dropout=0.0).to("cuda").eval()

    # ---- 构造 ----

    def _arena_model(self, seed):
        m = self.tool.ArenaModel.__new__(self.tool.ArenaModel)
        m.ckpt_path = f"rand{seed}"
        m.device = "cuda"
        m.seq = self.seqs[seed]
        m._tc_cache, m._elo_cache = {}, {}
        m.c_visit, m.c_scale = 50.0, 0.1
        return m

    def _evaluator(self, seed):
        return self.ka.SsmEvaluator(self.seqs[seed], "cuda", f"S:rand{seed}")

    def _factory(self, seed, n_sims, m0):
        from unichess_kit.search.gumbel import GumbelConfig
        return self.ka.SsmPlayerFactory(f"S{seed}", self._evaluator(seed),
                                        GumbelConfig(simulations=n_sims, m0=m0, g=0.0))

    def _cfg(self, n_sims, m0, max_plies):
        cfg = lambda: None  # noqa: E731
        cfg.n_sims, cfg.m0, cfg.max_plies, cfg.c_visit, cfg.c_scale = n_sims, m0, max_plies, 50.0, 0.1
        return cfg

    def _kit_game(self, fa, fb, opening_uci, a_is_white, max_plies, seed=7):
        from unichess_kit.api import SearchBudget
        from unichess_kit.pipelines.match import GameTask, play_game
        from unichess_kit.rules.referee import StandardReferee
        from unichess_kit.runtime import run_sync

        task = GameTask(game=0, pair=0, a_is_white=a_is_white, opening=tuple(opening_uci),
                        seed_a=seed, seed_b=seed)
        referee = StandardReferee(max_plies=len(opening_uci) + max_plies)
        self._last_players = (fa(), fb())
        return run_sync(play_game(task, *self._last_players, referee, SearchBudget()))

    def _s_batched(self, ma, mb, a_is_white, cfg, opening_san, seed=7):
        game = self.tool.BatchedArenaGame(0, 0, [ma, mb], a_is_white, cfg, opening_san, 0, seed=seed)
        args = self.tool.argparse.Namespace(sprt=False, games=1, sprt_alpha=0.05, sprt_beta=0.05,
                                            sprt_min_games=64)
        driver = self.tool.BatchedArenaDriver([game], concurrency=1, args=args,
                                              internal_sprt=False)
        driver.run()
        return driver.results[0], driver.n_forwards

    # ---- 测试 ----

    def test_game_parity_with_s_arena(self):
        cases = [("e4 e5", ("e2e4", "e7e5"), True, 12, 4, 8),
                 ("d4 Nf6 c4", ("d2d4", "g8f6", "c2c4"), False, 16, 16, 6),
                 ("", (), True, 8, 2, 5)]
        for san, uci, a_white, n_sims, m0, plies in cases:
            with self.subTest(opening=san, a_is_white=a_white):
                cfg = self._cfg(n_sims, m0, plies)
                ma, mb = self._arena_model(1), self._arena_model(2)
                mw, mblk = (ma, mb) if a_white else (mb, ma)
                gd = self.tool.play_one_game(mw, mblk, cfg, opening_san=san or None, seed=7)
                gd_b, n_fwd_s = self._s_batched(ma, mb, a_white, cfg, san)
                self.assertEqual(gd["pgn"], gd_b["pgn"])

                fa, fb = self._factory(1, n_sims, m0), self._factory(2, n_sims, m0)
                rec = self._kit_game(fa, fb, uci, a_white, plies)
                self.assertEqual(rec["opening"] + rec["moves"], _pgn_ucis(gd["pgn"]))
                self.assertEqual(len(rec["moves"]), gd["n_plies"])
                hist_k = hist_merge(self._last_players[0].expand_hist,
                                    self._last_players[1].expand_hist)
                self.assertEqual(hist_k, gd["expand_depth_hist"])
                n_fwd_k = fa.evaluator.n_forwards + fb.evaluator.n_forwards
                self.assertLessEqual(n_fwd_k, n_fwd_s - 1)
                self.assertGreaterEqual(n_fwd_k, n_fwd_s - 1 - hist_summary(hist_k)["replay_forwards"])

    def test_catch_up_equals_per_ply_stepping(self):
        """懒追赶（choose 时一次补齐）与逐 ply 步进得到同一 cache 与根评估。"""
        from unichess_kit.api import GameStart, SearchBudget
        from unichess_kit.runtime import run_sync

        ev = self._evaluator(1)
        board = chess.Board()
        for san in ("e4", "e5", "Nf3", "Nc6", "Bb5"):
            board.push_san(san)
        # 参考：逐局面步进（encode-before-increment）
        from stateseq.data.sequences import _board_key
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
        from unichess_kit.api import Leaf
        from unichess_kit.runtime import run_sync

        ev = self._evaluator(2)
        exp = self.ka.SsmExpander(ev)
        root_board = chess.Board()
        (out,) = ev.evaluate([self.ka.encode_payload(root_board, 0, ev.initial_cache())])
        from stateseq.data.sequences import _board_key
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
        from unichess_kit.api import Leaf
        from unichess_kit.runtime import run_sync

        exp = self.ka.SsmExpander(self._evaluator(1))
        with self.assertRaises(ValueError):
            run_sync(exp.expand([Leaf(board=chess.Board())]))

    def test_run_match_concurrent(self):
        """并发 4 的 kit 批量对弈能跑通（批大小变化 → 只验证合法与计数，不验逐位）。"""
        import tempfile
        from unichess_kit.pipelines.match import MatchConfig, run_match

        fa, fb = self._factory(1, 8, 4), self._factory(2, 8, 4)
        with tempfile.TemporaryDirectory() as d:
            cfg = MatchConfig(pairs=2, seed=3, max_plies=14, concurrency=4, openings="bundled")
            summary = run_match(cfg, make_a=fa, make_b=fb, out_path=os.path.join(d, "r.jsonl"))
        self.assertEqual(summary["games"], 4)


if __name__ == "__main__":
    unittest.main()
