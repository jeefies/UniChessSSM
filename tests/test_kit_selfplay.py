"""S 自对弈接入 kit（``SsmSelfPlayer`` + ``V3Sink``）单元测试。需要 CUDA 与兄弟仓库 Kit。

切换前与原生成器 ``Driver`` 写出的 v3 分片逐字节相同、生成统计逐项相同（随机权重与真实
权重各一组，见 git 历史中的本文件与 ``tools/kit_selfplay_parity.py``）。原生成器已删除，这里锁死：
1. 并发 1 下同配置两次运行分片逐字节相同（book π′ 缓存有命中）；
2. 按全局局号拆成两段（多 worker 的做法）与一次跑完逐局相同——局号、rng、book 分配
   只取决于全局局号，每局的 meta / 着法 / π′ 与在哪个 worker 生成无关；
3. 并发 > 1 结构正确（局号全、book 着法与 flags、跨局拼批）。
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


import glob
import os
import shutil
import sys
import tempfile
import unittest


try:
    import torch
    import numpy as np

    _HAS_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover - 本机（Windows）无 torch
    _HAS_CUDA = False

try:
    import Kit  # noqa: F401

    _HAS_KIT = True
except ImportError:  # pragma: no cover
    _HAS_KIT = False


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


def games_by_key(shard_dir: str) -> dict:
    """分片目录 → {game_key: (meta 行字节, 着法字节, π′ 字节)}，与分片切分方式无关。"""
    out = {}
    for meta_path in glob.glob(os.path.join(shard_dir, "*.meta.npz")):
        base = meta_path[:-len(".meta.npz")]
        with np.load(meta_path) as z:
            metas, offsets = z["metas"], z["offsets"]
        acts = np.fromfile(base + ".actions.bin", dtype=np.uint16)
        poff = np.fromfile(base + ".pipol.offsets.bin", dtype=np.int64)
        with open(base + ".pipol.bin", "rb") as fh:
            blob = fh.read()
        for i, m in enumerate(metas):
            key = m["game_key"]
            key = key.decode() if isinstance(key, bytes) else str(key)
            out[key] = (m.tobytes(), acts[offsets[i]:offsets[i + 1]].tobytes(),
                        blob[poff[i]:poff[i + 1]])
    return out


@unittest.skipUnless(_HAS_CUDA and _HAS_KIT, "需要 CUDA（mamba 步进核）与兄弟仓库 Kit")
class TestKitSelfPlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import SSM.kit as ka
        from SSM.model import SeqModel

        cls.ka = ka
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

    def _run_kit(self, games, seed, n_sims, max_plies, openings, book_plies, g, concurrency=1,
                 first_game=0, name=None):
        from SSM.dataset.gshards import V3ShardWriter
        from Kit.pipelines.selfplay import SelfPlayConfig, run_selfplay

        ka = self.ka
        out = os.path.join(self.tmp, name or f"k{concurrency}")
        ev = ka.SsmEvaluator(self.seq, "cuda", "S:rand7")
        factory = ka.make_selfplay_factory(evaluator=ev, simulations=n_sims, g=g)
        writer = V3ShardWriter(out, "t")
        sink = ka.V3Sink(writer, gen_id=3)
        summary = run_selfplay(SelfPlayConfig(games=games, seed=seed, max_plies=max_plies,
                                              concurrency=concurrency, openings=openings,
                                              book_plies=book_plies, first_game=first_game),
                               factory, sink)
        writer.flush()
        return out, factory, sink, summary

    def test_deterministic_with_book(self):
        # 3 条线（其一裁切后与第一条重复、各占序号）× 7 局：memo 命中与 book_id 取模都覆盖到
        path = self._openings(["e4 e5 Nf3 Nc6", "d4 Nf6", "e4 e5 Nf3 Nc6 Bb5"])
        kw = dict(games=7, seed=11, n_sims=16, max_plies=24, openings=path, book_plies=3, g=1.0)
        d1, f1, s1, _ = self._run_kit(name="r1", **kw)
        d2, f2, s2, _ = self._run_kit(name="r2", **kw)
        self.assertEqual(compare_shard_dirs(d1, d2), [])
        self.assertGreater(f1.shared.book_memo_hits, 0)
        self.assertEqual(f1.shared.expand_hist, f2.shared.expand_hist)
        self.assertEqual(s1.term_reason_counts, s2.term_reason_counts)

    def test_split_by_global_index_equals_whole(self):
        path = self._openings(["e4 e5 Nf3 Nc6", "d4 Nf6", "c4 e5"])
        kw = dict(seed=11, n_sims=16, max_plies=20, openings=path, book_plies=3, g=1.0)
        whole, *_ = self._run_kit(games=7, name="whole", **kw)
        part1, *_ = self._run_kit(games=3, first_game=0, name="p1", **kw)
        part2, *_ = self._run_kit(games=4, first_game=3, name="p2", **kw)
        got = {**games_by_key(part1), **games_by_key(part2)}
        want = games_by_key(whole)
        self.assertEqual(len(want), 7)
        self.assertEqual(sorted(got), sorted(want))
        for key in want:
            self.assertEqual(got[key], want[key], f"{key} 按段生成与整段生成不同")

    def test_concurrent_run_valid(self):
        """并发 > 1 不求逐字节（拼批浮点差），只验结构：局号全、book 着法与 flags、π′ 归一。"""
        from SSM.actions import move_to_action
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
        from SSM.dataset.gshards import make_game_key
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
