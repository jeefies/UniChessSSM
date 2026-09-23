"""S 自对弈接入 kit（``SsmSelfPlayer`` + ``V3Sink``）单元测试。需要 CUDA 与兄弟仓库 Kit。

核心断言：kit ``run_selfplay`` 与原生成器 ``ssm_gumbel_selfplay.Driver``（随机初始化模型、
并发 1、同 seed / 开局库 / book_plies）写出的 v3 分片 **逐字节相同**（actions / pipol /
pipol.offsets），meta 数组逐项相同；生成统计（book π′ 缓存命中、预算违例、扩展深度直方图、
终局原因）也逐项相同。
"""

from __future__ import annotations

import glob
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
KIT_ROOT = os.environ.get("UNICHESS_KIT_ROOT", os.path.join(os.path.dirname(HERE), "Kit"))
if os.path.isdir(KIT_ROOT) and KIT_ROOT not in sys.path:
    sys.path.insert(0, KIT_ROOT)

try:
    import torch
    import numpy as np

    _HAS_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover - 本机（Windows）无 torch
    _HAS_CUDA = False

try:
    import unichess_kit  # noqa: F401

    _HAS_KIT = True
except ImportError:  # pragma: no cover
    _HAS_KIT = False


def _load_tool():
    path = os.path.join(HERE, "tools", "ssm_gumbel_selfplay.py")
    spec = importlib.util.spec_from_file_location("ssm_gumbel_selfplay_kit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def compare_shard_dirs(dir_a: str, dir_b: str) -> list:
    """两个分片目录逐文件对照 → 差异描述列表（空 = 一致）。meta.npz 比数组而非字节（zip 时间戳）。"""
    diffs = []
    names_a = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dir_a, "shard-*")))
    names_b = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dir_b, "shard-*")))
    if names_a != names_b:
        return [f"文件集不同：{names_a} vs {names_b}"]
    if not names_a:
        return ["没有分片文件"]
    for name in names_a:
        pa, pb = os.path.join(dir_a, name), os.path.join(dir_b, name)
        if name.endswith(".npz"):
            with np.load(pa) as za, np.load(pb) as zb:
                if sorted(za.files) != sorted(zb.files):
                    diffs.append(f"{name}：键不同")
                    continue
                for k in za.files:
                    if za[k].dtype != zb[k].dtype or za[k].tobytes() != zb[k].tobytes():
                        diffs.append(f"{name}[{k}] 不同")
        else:
            with open(pa, "rb") as fa, open(pb, "rb") as fb:
                if fa.read() != fb.read():
                    diffs.append(f"{name} 字节不同")
    return diffs


@unittest.skipUnless(_HAS_CUDA and _HAS_KIT, "需要 CUDA（mamba 步进核）与兄弟仓库 Kit")
class TestKitSelfPlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from stateseq import kit_adapter as ka
        from stateseq.model import SeqModel

        cls.ka = ka
        cls.tool = _load_tool()
        torch.manual_seed(7)
        cls.seq = SeqModel(dropout=0.0).to("cuda").eval()

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="kit_selfplay_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _openings(self, lines):
        path = os.path.join(self.tmp, "openings.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return path

    def _run_s(self, games, seed, n_sims, max_plies, openings, book_plies, g):
        T = self.tool
        model = T.ModelWrapper.__new__(T.ModelWrapper)
        model.device, model.seq, model._tc_cache, model._elo_cache = "cuda", self.seq, {}, {}
        out = os.path.join(self.tmp, "s")
        cfg = T.SelfPlayConfig(ckpt="rand7", out_dir=out, tag="t", num_games=games, concurrency=1,
                               n_sims=n_sims, seed=seed, max_plies=max_plies, gen_id=3,
                               gumbel_g=g, openings_path=openings or "", book_plies=book_plies)
        writer = T.V3ShardWriter(out, "t")
        driver = T.Driver(model, cfg, writer, openings=T.load_openings(openings, book_plies)
                          if openings else [])
        driver.run(progress_every=10 ** 9)
        writer.flush()
        return out, driver

    def _run_kit(self, games, seed, n_sims, max_plies, openings, book_plies, g, concurrency=1):
        from stateseq.data.gshards import V3ShardWriter
        from unichess_kit.pipelines.selfplay import SelfPlayConfig, run_selfplay

        ka = self.ka
        out = os.path.join(self.tmp, f"k{concurrency}")
        ev = ka.SsmEvaluator(self.seq, "cuda", "S:rand7")
        factory = ka.make_selfplay_factory(evaluator=ev, simulations=n_sims, g=g)
        writer = V3ShardWriter(out, "t")
        sink = ka.V3Sink(writer, gen_id=3)
        summary = run_selfplay(SelfPlayConfig(games=games, seed=seed, max_plies=max_plies,
                                              concurrency=concurrency, openings=openings,
                                              book_plies=book_plies), factory, sink)
        writer.flush()
        return out, factory, sink, summary

    def _assert_parity(self, games, seed, n_sims, max_plies, openings, book_plies, g=1.0):
        s_dir, drv = self._run_s(games, seed, n_sims, max_plies, openings, book_plies, g)
        k_dir, fac, sink, summary = self._run_kit(games, seed, n_sims, max_plies, openings,
                                                  book_plies, g)
        self.assertEqual(compare_shard_dirs(s_dir, k_dir), [])
        sh = fac.shared
        self.assertEqual(summary["games"], drv.games_done)
        self.assertEqual(sink.plies, drv.total_plies)
        self.assertEqual(sink.term_reason_counts, drv.term_reason_counts)
        self.assertEqual(sink.truncated_games, drv.truncated_games)
        self.assertEqual((sh.book_memo_hits, sh.book_memo_misses),
                         (drv.book_memo_hits, drv.book_memo_misses))
        self.assertEqual(sh.budget_violations, drv.budget_violations)
        self.assertEqual(sh.expand_hist, drv.expand_hist)
        self.assertEqual((sh.n_nodes, sh.sims, sh.max_depth),
                         (drv.total_nodes, drv.total_sims, drv.total_max_depth))
        return drv, fac

    def test_parity_with_book(self):
        # 3 条线（其一裁切后与第一条重复、各占序号）× 7 局：memo 命中与 book_id 取模都覆盖到
        path = self._openings(["e4 e5 Nf3 Nc6", "d4 Nf6", "e4 e5 Nf3 Nc6 Bb5"])
        drv, fac = self._assert_parity(games=7, seed=11, n_sims=16, max_plies=24,
                                       openings=path, book_plies=3)
        self.assertGreater(drv.book_memo_hits, 0)

    def test_parity_without_book(self):
        self._assert_parity(games=3, seed=5, n_sims=16, max_plies=20, openings=None,
                            book_plies=6)

    def test_concurrent_run_valid(self):
        """并发 > 1 不求逐字节（拼批浮点差），只验结构：局号全、book 着法与 flags、π′ 归一。"""
        from stateseq.actions import move_to_action
        import chess

        path = self._openings(["e4 e5 Nf3", "d4 d5"])
        out, fac, sink, summary = self._run_kit(games=6, seed=3, n_sims=16, max_plies=16,
                                                openings=path, book_plies=2, g=1.0,
                                                concurrency=4)
        self.assertEqual(summary["games"], 6)
        self.assertGreater(summary["batch"]["positions"], summary["batch"]["forwards"])  # 跨局拼批
        (meta_path,) = glob.glob(os.path.join(out, "*.meta.npz"))
        with np.load(meta_path) as z:
            metas = z["metas"]
        from stateseq.data.gshards import make_game_key
        dec = lambda k: k.decode() if isinstance(k, bytes) else str(k)   # noqa: E731
        self.assertEqual(sorted(dec(k) for k in metas["game_key"]),
                         sorted(make_game_key("selfplay_gen3", g) for g in range(6)))
        self.assertTrue(all(int(f) == 2 for f in metas["flags"]))
        book0 = [move_to_action(chess.Move.from_uci(u)) for u in ("e2e4", "d2d4")]
        (act_path,) = glob.glob(os.path.join(out, "*.actions.bin"))
        acts = np.fromfile(act_path, dtype=np.uint16)
        starts = np.concatenate([[0], np.cumsum(metas["n_plies"])[:-1]])
        self.assertEqual(sorted(int(acts[int(s)]) for s in starts), sorted(book0 * 3))
        self.assertEqual(sink.games, 6)


if __name__ == "__main__":
    unittest.main()
