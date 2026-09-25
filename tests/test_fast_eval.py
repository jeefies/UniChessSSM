"""P4 快速前向（``SSM.infer.fast_eval``）对照参考实现。需要 CUDA 与兄弟仓库 Kit。

1. 单步：与 ``SsmEvaluator``（SeqModel.step + cat/split）同批大小下 logits / wdl / 状态逐位相同，
   eager 与 CUDA graph 均是；
2. 搜索：并发 1、串行模拟（parallel=False）时，快速实现（每叶 1 次前向）与参考实现
   （从根重放）整次搜索逐位相同——选着、根访问数、Q 累加、π′；
3. 槽池淘汰：极小的池（大量淘汰 + 重算）与大池结果逐位相同；
4. 轮内并发（parallel=True）只改拼批：与串行的 π′ 差在拼批噪声量级内；
   批不变模式（共享 GPU 服务所用）下行结果与同批其他行无关、逐位相同；
5. 对局级：快速实现的 SsmPlayer 与参考实现的着法序列一致（并发 1、串行）。
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


import gc
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

try:
    import Kit  # noqa: F401

    _HAS_KIT = True
except ImportError:  # pragma: no cover
    _HAS_KIT = False

FENS = [
    None,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
]


@unittest.skipUnless(_HAS_CUDA and _HAS_KIT, "需要 CUDA（mamba 步进核）与兄弟仓库 Kit")
class TestFastEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from SSM.infer import fast_eval as fe
        import SSM.kit as ka
        from SSM.model import SeqModel

        cls.ka, cls.fe = ka, fe
        torch.manual_seed(3)
        cls.seq = SeqModel(dropout=0.0).to("cuda").eval()
        cls.ref = ka.SsmEvaluator(cls.seq, "cuda", "S:ref")

    def _store(self, slots):
        fe = self.fe
        return (fe.StateTensors(*fe.model_dims(self.seq), fe.RESERVED_SLOTS + slots, "cuda"),
                fe.SlotPool(slots))

    def _fast(self, slots=512, graphs=True, invariant=False, store=None):
        return self.fe.SsmFastEvaluator.local(self.seq, "cuda", f"S:fast{slots}{graphs}",
                                              cuda_graphs=graphs, batch_invariant=invariant,
                                              store=store or self._store(slots))

    def _line(self, n=10):
        board = chess.Board()
        out = []
        for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6",
                    "e1g1", "f8e7")[:n]:
            board.push_uci(uci)
            out.append(board.copy())
        return out

    def _encode(self, board, occ=0):
        from SSM.kit import encode_board
        feats, tc, elo, color = encode_board(board, occ)
        return np.asarray(feats, dtype=np.float32).reshape(-1), tc, elo, color

    def _assert_state_equal(self, fast, node, cache):
        t = fast.backend.engine.tensors
        for li, (conv, ssm) in enumerate(cache):
            self.assertTrue(torch.equal(t.conv_views[li][node.slot], conv[0]), li)
            self.assertTrue(torch.equal(t.ssm_views[li][node.slot], ssm[0]), li)

    # ---- 1. 单步 ----

    def test_step_bitwise_batch1(self):
        for graphs in (False, True):
            with self.subTest(graphs=graphs):
                fast = self._fast(graphs=graphs)
                cache, node = self.ref.initial_cache(), fast.root_state()
                for board in self._line():
                    enc = self._encode(board)
                    (r_lg, r_wd, cache), = self.ref.evaluate([self.ref.child(cache, *enc)])
                    (f_lg, f_wd, new), = fast.evaluate([fast.child(node, *enc)])
                    fast.hold(new)
                    fast.release(node)
                    node = new
                    self.assertEqual(r_lg.tobytes(), f_lg.tobytes())
                    self.assertEqual(r_wd.tobytes(), f_wd.tobytes())
                    self._assert_state_equal(fast, node, cache)

    def test_step_bitwise_batch8_mixed_parents(self):
        """8 个不同父状态（含零状态）拼一批：与参考实现同批大小逐位相同。"""
        for graphs in (False, True):
            with self.subTest(graphs=graphs):
                fast = self._fast(graphs=graphs)
                boards = self._line(8)
                caches, nodes = [self.ref.initial_cache()], [fast.root_state()]
                for board in boards[:7]:       # 逐个步进出 7 个父状态
                    enc = self._encode(board)
                    (_, _, c), = self.ref.evaluate([self.ref.child(caches[-1], *enc)])
                    (_, _, n), = fast.evaluate([fast.child(nodes[-1], *enc)])
                    caches.append(c)
                    nodes.append(n)
                enc = self._encode(boards[7])
                r = self.ref.evaluate([self.ref.child(c, *enc) for c in caches])
                f = fast.evaluate([fast.child(n, *enc) for n in nodes])
                for (r_lg, r_wd, r_c), (f_lg, f_wd, f_n) in zip(r, f):
                    self.assertEqual(r_lg.tobytes(), f_lg.tobytes())
                    self.assertEqual(r_wd.tobytes(), f_wd.tobytes())
                    self._assert_state_equal(fast, f_n, r_c)
                # 父状态未被改写（copy-on-write）
                for n, c in zip(nodes[1:], caches[1:]):
                    self._assert_state_equal(fast, n, c)

    def test_padded_bucket_matches_eager_rows(self):
        """批 5 → 桶 8（3 行填充）：有效行与同批 eager 前向在 1e-5 内一致，且填充不污染父槽。"""
        fast_g, fast_e = self._fast(graphs=True), self._fast(graphs=False)
        boards = self._line(5)
        outs = []
        for fast in (fast_g, fast_e):
            root = fast.root_state()
            res = fast.evaluate([fast.child(root, *self._encode(b)) for b in boards])
            outs.append(np.stack([r[0] for r in res]))
            t = fast.backend.engine.tensors
            self.assertTrue(torch.count_nonzero(t.conv[:, 0]) == 0, "零槽被写")
            self.assertTrue(torch.count_nonzero(t.ssm[:, 0]) == 0, "零槽被写")
        np.testing.assert_allclose(outs[0], outs[1], rtol=0, atol=1e-4)

    def test_ssm_update_src_dst_bitwise(self):
        """源/目标槽内核 == 先把源槽复制到目标槽、再用原 selective_state_update 就地更新（逐位）。
        含零状态源、tie_hdim 形态、同批多行、源槽不被改写。"""
        from einops import repeat
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update

        from SSM.infer.ssm_update import ssm_update_src_dst

        g = torch.Generator(device="cuda").manual_seed(0)
        blk = self.seq.r.blocks[0]
        H, P, N = blk.nheads, blk.headdim, blk.d_state
        S, b = 40, 7
        state = torch.randn((S, H, P, N), device="cuda", generator=g)
        state[0].zero_()
        x = torch.randn((b, H, P), device="cuda", generator=g)
        dt = repeat(torch.randn((b, H), device="cuda", generator=g), "b h -> b h p", p=P)
        A = repeat(-torch.rand(H, device="cuda", generator=g), "h -> h p n", p=P, n=N)
        B = torch.randn((b, 1, N), device="cuda", generator=g)
        C = torch.randn((b, 1, N), device="cuda", generator=g)
        D = repeat(torch.randn(H, device="cuda", generator=g), "h -> h p", p=P)
        dt_bias = repeat(torch.randn(H, device="cuda", generator=g), "h -> h p", p=P)
        src = torch.tensor([0, 3, 5, 5, 9, 11, 0], dtype=torch.int32, device="cuda")
        dst = torch.tensor([20, 21, 22, 23, 24, 25, 26], dtype=torch.int32, device="cuda")
        ref = state.clone()
        ref[dst.long()] = ref[src.long()]
        y_ref = selective_state_update(ref, x, dt, A, B, C, D, z=None, dt_bias=dt_bias,
                                       dt_softplus=True, state_batch_indices=dst)
        new = state.clone()
        y = ssm_update_src_dst(new, x, dt, A, B, C, D, dt_bias, True, src, dst)
        self.assertTrue(torch.equal(y, y_ref))
        self.assertTrue(torch.equal(new, ref), "目标槽不同或源槽被改写")

    def test_batch_invariant_rows_independent_of_batch(self):
        """批不变模式（固定 FIXED_CHUNK 行块）：同一行单独算、与另外 150 行（跨多块）一起算，结果逐位相同；
        eager 与 graph 也逐位相同。这是共享 GPU 服务结果与时序/进程数无关的前提。"""
        boards = self._line(10)
        outs = []
        for graphs, extra in ((True, 0), (True, 150), (False, 150), (False, 0)):
            fast = self._fast(slots=512, graphs=graphs, invariant=True)
            root = fast.root_state()
            (_, _, parent), = fast.evaluate([fast.child(root, *self._encode(boards[0]))])
            fast.hold(parent)
            filler = [fast.child(root, *self._encode(boards[1 + i % 9])) for i in range(extra)]
            target = fast.child(parent, *self._encode(boards[1]))
            res = fast.evaluate(filler[:100] + [target] + filler[100:])
            lg, wd, node = res[min(100, extra)]
            self.assertIs(node, target)
            t = fast.backend.engine.tensors
            state = [t.conv_views[li][node.slot].clone() for li in range(len(t.conv_views))]
            state += [t.ssm_views[li][node.slot].clone() for li in range(len(t.ssm_views))]
            outs.append((lg.tobytes(), wd.tobytes(), state))
        for o in outs[1:]:
            self.assertEqual(outs[0][0], o[0])
            self.assertEqual(outs[0][1], o[1])
            for a, b in zip(outs[0][2], o[2]):
                self.assertTrue(torch.equal(a, b))

    # ---- 2-4. 搜索 ----

    def _search(self, evaluator, board, sims=48, m0=8, parallel=False, g=0.0):
        from Kit.runtime import Batcher, run_sync
        from Kit.search.gumbel import GumbelConfig

        cfg = GumbelConfig(simulations=sims, m0=m0, g=g, parallel=parallel)
        player = self.ka.SsmPlayer("S", evaluator, cfg)
        from Kit.api import GameStart
        start_fen = None if board.move_stack else board.fen()
        run_sync(player.new_game(GameStart(color=board.turn, fen=start_fen, seed=11)))
        # 追赶到 board（从起始局面按着法步进）
        batcher = Batcher()
        root, _ = run_sync(player._root(board), batcher)
        rng = np.random.default_rng(5)
        res = run_sync(player._search(board, root, rng, sims), batcher)
        out = (res.move, res.root.n.copy(), res.root.q_sum.tobytes(),
               res.pi_prime(cfg)[1].tobytes(), res.stats["n_nodes"])
        player.close()
        return out, batcher.stats

    def _boards(self):
        out = []
        for fen in FENS:
            b = chess.Board()
            if fen is None:
                for uci in ("d2d4", "d7d5", "c2c4"):
                    b.push_uci(uci)
            else:
                b = chess.Board(fen)
            out.append(b)
        return out

    def test_search_bitwise_vs_reference(self):
        for board in self._boards():
            with self.subTest(fen=board.fen()):
                ref, _ = self._search(self.ref, board)
                fast, st = self._search(self._fast(), board)
                self.assertEqual(ref[0], fast[0])
                np.testing.assert_array_equal(ref[1], fast[1])
                self.assertEqual(ref[2:], fast[2:])

    def test_tiny_pool_eviction_same_result(self):
        board = self._boards()[1]
        big, _ = self._search(self._fast(slots=1024), board, sims=64)
        tiny_ev = self._fast(slots=12)
        tiny, _ = self._search(tiny_ev, board, sims=64)
        self.assertGreater(tiny_ev.pool.evictions, 10)
        self.assertGreater(tiny_ev.pool.rematerialized, 0)
        self.assertEqual(big[0], tiny[0])
        np.testing.assert_array_equal(big[1], tiny[1])
        self.assertEqual(big[2:], tiny[2:])

    def test_shared_tiny_pool_two_evaluators_concurrent(self):
        """arena 口径：A/B 两个求值器共用一个极小的池、同一拍交错分配。一方刚重算出的祖先
        不得被另一方的分配淘汰（ensure 全链钉住），结果与各自大池单跑逐位相同。"""
        from Kit.api import GameStart, gather
        from Kit.runtime import Batcher, run_sync
        from Kit.search.gumbel import GumbelConfig

        boards = self._boards()[:2]
        solo = [self._search(self._fast(slots=1024), b, sims=64)[0] for b in boards]
        store = self._store(16)
        pool = store[1]
        evs = [self.fe.SsmFastEvaluator.local(self.seq, "cuda", f"S:shared{i}", store=store)
               for i in range(2)]
        cfg = GumbelConfig(simulations=64, m0=8, g=0.0, parallel=False)
        players = [self.ka.SsmPlayer("S", ev, cfg) for ev in evs]
        batcher = Batcher()
        roots = []
        for player, board in zip(players, boards):
            run_sync(player.new_game(GameStart(color=board.turn,
                                               fen=None if board.move_stack else board.fen(),
                                               seed=11)))
            roots.append(run_sync(player._root(board), batcher)[0])
        res = run_sync(gather([p._search(b, r, np.random.default_rng(5), 64)
                               for p, b, r in zip(players, boards, roots)]), batcher)
        self.assertGreater(pool.evictions, 10)
        self.assertGreater(pool.rematerialized, 0)
        for r, s in zip(res, solo):
            self.assertEqual(r.move, s[0])
            np.testing.assert_array_equal(r.root.n, s[1])
            self.assertEqual(r.root.q_sum.tobytes(), s[2])
        for p in players:
            p.close()

    def test_parallel_close_to_serial(self):
        for board in self._boards():
            with self.subTest(fen=board.fen()):
                ser, s1 = self._search(self._fast(), board, sims=64, m0=16)
                par, s2 = self._search(self._fast(), board, sims=64, m0=16, parallel=True)
                self.assertLess(s2.ticks, s1.ticks)
                pi_s = np.frombuffer(ser[3], np.float32)
                pi_p = np.frombuffer(par[3], np.float32)
                self.assertLess(float(np.abs(pi_s - pi_p).max()), 1e-3)

    def test_slots_returned_after_search(self):
        fast = self._fast(slots=1024)
        self._search(fast, self._boards()[0], sims=64)
        gc.collect()
        fast.pool.collect()
        self.assertEqual(fast.pool.used, 0, "搜索与对局结束后槽应全部归还")

    # ---- 5. 对局 ----

    def test_game_moves_match_reference(self):
        from Kit.api import SearchBudget
        from Kit.pipelines.match import GameTask, play_game
        from Kit.rules.referee import StandardReferee
        from Kit.runtime import run_sync
        from Kit.search.gumbel import GumbelConfig

        def game(evaluator):
            cfg = GumbelConfig(simulations=16, m0=4, g=0.0, parallel=False)
            fa = self.ka.SsmPlayerFactory("A", evaluator, cfg)
            fb = self.ka.SsmPlayerFactory("B", evaluator, cfg)
            task = GameTask(game=0, pair=0, a_is_white=True, opening=("e2e4", "c7c5"),
                            seed_a=3, seed_b=3)
            return run_sync(play_game(task, fa(), fb(), StandardReferee(max_plies=14),
                                      SearchBudget()))["moves"]

        self.assertEqual(game(self.ref), game(self._fast()))


if __name__ == "__main__":
    unittest.main()
