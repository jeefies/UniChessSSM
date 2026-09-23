"""arena 开局分配单测：g=0 确定性下开局多样性是对局多样性唯一来源。

背景：arena 用 gumbel_topm(g=0.0)，噪声恒 0，对局（含棋谱与结果）仅是
「网络 + 开局 + 执色」的确定性函数。若多个 pair 共用同一开局，它们的对局逐字节相同、
结果相同，64 局会退化成 16 个有效样本（已实测复现：16 组 × 每组 4 局，组内棋谱完全一致）。

开局解析与分配已移到 kit（``unichess_kit.rules.openings``，其单测覆盖解析/去重/非法行/
CRLF 与排列算法）。这里锁死 S 侧用法的不变量：
1. 内置库经 kit 分配时，库足够则每对开局互不相同；同 seed 可复现、换 seed 换卷；
2. 库不足时如实循环复用（由 duplicate_rate 暴露，不假装多样）；
3. 默认开局文件足够 400 局门禁零重复，且不裁切（按固定 ply 裁切会折叠多样性）；
4. ``_pgn_fingerprint`` 忽略对局元数据，只按棋步判同。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
KIT_ROOT = os.environ.get("UNICHESS_KIT_ROOT", os.path.join(os.path.dirname(HERE), "Kit"))
if os.path.isdir(KIT_ROOT) and KIT_ROOT not in sys.path:
    sys.path.append(KIT_ROOT)  # 追加而非前插：Kit 的 tests 包不得遮蔽本仓库的 tests

_TOOL_PATH = os.path.join(HERE, "tools", "ssm_gumbel_arena.py")
_BOOK_PATH = os.path.join(HERE, "data", "openings_200.txt")

try:
    import chess  # noqa: F401

    from unichess_kit.rules.openings import OpeningBook

    _HAS_KIT = True
except ImportError:  # pragma: no cover - 缺 python-chess 或兄弟仓库 Kit
    _HAS_KIT = False


def _load_tool():
    spec = importlib.util.spec_from_file_location("ssm_gumbel_arena_openings", _TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(_HAS_KIT, "需要 python-chess 与兄弟仓库 Kit")
class TestOpeningPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_tool()
        cls.book = OpeningBook.from_file(cls.mod.resolve_openings("", tempfile.mkdtemp()))

    def test_builtin_library_parses_completely(self):
        self.assertEqual(len(self.mod.OPENINGS), 16)
        self.assertEqual(len(self.book), 16, "内置开局有非法或重复行")

    def test_distinct_openings_when_library_sufficient(self):
        n = len(self.book)
        for n_pairs in (1, 4, n - 1, n):
            ids = [oi for oi, _ in self.book.plan(n_pairs, 20260917)]
            self.assertEqual(len(ids), n_pairs)
            self.assertEqual(len(set(ids)), n_pairs,
                             f"n_pairs={n_pairs} 时开局仍有重复（对局会逐字节重复）")

    def test_plan_reproducible_and_rotates_with_seed(self):
        a = self.book.plan(8, 20260917)
        self.assertEqual(a, self.book.plan(8, 20260917), "同一 seed 的计划必须可复现（固定考卷）")
        self.assertNotEqual(a, self.book.plan(8, 20260918), "换 seed 应换一套开局（新代次换考卷）")

    def test_plan_reuses_library_when_exhausted(self):
        small = OpeningBook(self.book.lines[:3])
        counts: dict = {}
        for oi, _ in small.plan(8, 7):
            counts[oi] = counts.get(oi, 0) + 1
        self.assertEqual(sorted(counts.values()), [2, 3, 3])

    def test_missing_file_falls_back_to_builtin(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.mod.resolve_openings(os.path.join(d, "nope.txt"), d)
            self.assertEqual(len(OpeningBook.from_file(path)), 16)


@unittest.skipUnless(_HAS_KIT, "需要 python-chess 与兄弟仓库 Kit")
class TestRealOpeningsBook(unittest.TestCase):
    def test_openings_file_large_enough_and_uncut(self):
        if not os.path.exists(_BOOK_PATH):
            self.skipTest("data/openings_200.txt 不存在")
        full = OpeningBook.from_file(_BOOK_PATH)
        # 400 局门禁 => 200 对 => 需要至少 200 个开局才能零重复
        self.assertGreaterEqual(len(full), 200,
                                "开局库不足 200：g=0 的 arena 无法给出 400 个不同对局")
        self.assertLess(len(OpeningBook.from_file(_BOOK_PATH, max_plies=6)), len(full),
                        "固定 ply 裁切必然折叠开局多样性（arena 因此不裁切）")


@unittest.skipUnless(_HAS_KIT, "需要 python-chess 与兄弟仓库 Kit")
class TestPgnFingerprint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_tool()

    def test_fingerprint_ignores_metadata(self):
        g1 = {"pgn": '[Event "a"]\n[Site "x"]\n\n1. e4 e5 2. Nf3\n'}
        g2 = {"pgn": '[Event "b"]\n[Site "y"]\n\n1. e4 e5 2. Nf3\n'}
        g3 = {"pgn": '[Event "a"]\n\n1. e4 e5 2. Nf3 Nc6\n'}
        self.assertEqual(self.mod._pgn_fingerprint(g1), self.mod._pgn_fingerprint(g2))
        self.assertNotEqual(self.mod._pgn_fingerprint(g1), self.mod._pgn_fingerprint(g3))


if __name__ == "__main__":
    unittest.main()
