"""arena 开局分配单测：g=0 确定性下开局多样性是对局多样性唯一来源。

背景：arena 用 gumbel_topm(g=0.0)，噪声恒 0 且 ``self.rng`` 不参与任何决策，
seed 是死代码——对局（含棋谱与结果）仅是「网络 + 开局 + 执色」的确定性函数。
若多个 pair 共用同一开局，它们的对局逐字节相同、结果相同，64 局会退化成
16 个有效样本（已实测复现：16 组 × 每组 4 局，组内棋谱完全一致）。

本测试锁死不变量：
1. ``opening_plan`` 在库足够时为每对分配**互不相同**的开局；
2. 同 seed 计划可复现、换 seed 换卷（不同代次用不同固定考卷）；
3. 库不足时如实循环复用（由 duplicate_rate 暴露，不假装多样）；
4. ``load_openings_file`` 跳过非法着法行/去重/按 book_plies 截断；
5. ``_pgn_fingerprint`` 忽略对局元数据，只按棋步判同。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TOOL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "tools", "ssm_gumbel_arena.py")

try:
    import chess  # noqa: F401

    _HAS_CHESS = True
except ImportError:  # pragma: no cover - 本机（Windows）无 python-chess
    _HAS_CHESS = False


def _load_tool():
    spec = importlib.util.spec_from_file_location("ssm_gumbel_arena_openings", _TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(_HAS_CHESS, "需要 python-chess")
class TestOpeningPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_tool()

    def test_distinct_openings_when_library_sufficient(self):
        library = self.mod.OPENINGS
        for n_pairs in (1, 4, len(library) - 1, len(library)):
            plan = self.mod.opening_plan(n_pairs, library, seed=20260917)
            self.assertEqual(len(plan), n_pairs)
            ids = [oi for oi, _ in plan]
            self.assertEqual(len(set(ids)), n_pairs,
                             f"n_pairs={n_pairs} 时开局仍有重复（对局会逐字节重复）")

    def test_plan_reproducible_and_rotates_with_seed(self):
        library = self.mod.OPENINGS
        a = self.mod.opening_plan(8, library, seed=20260917)
        b = self.mod.opening_plan(8, library, seed=20260917)
        c = self.mod.opening_plan(8, library, seed=20260918)
        self.assertEqual(a, b, "同一 seed 的计划必须可复现（固定考卷）")
        self.assertNotEqual(a, c, "换 seed 应换一套开局（新代次换考卷）")
        self.assertEqual(sorted(oi for oi, _ in c), sorted(set(oi for oi, _ in c)))

    def test_plan_reuses_library_when_exhausted(self):
        library = self.mod.OPENINGS[:3]
        plan = self.mod.opening_plan(8, library, seed=7)
        self.assertEqual(len(plan), 8)
        counts = {}
        for oi, _ in plan:
            counts[oi] = counts.get(oi, 0) + 1
        self.assertEqual(sorted(counts.values()), [2, 3, 3])

    def test_empty_library_rejected(self):
        with self.assertRaises(ValueError):
            self.mod.opening_plan(4, [], seed=1)

    def test_builtin_library_supports_400_game_gate(self):
        # 400 局门禁 => 200 对 => 需要至少 200 个开局才能零重复；
        # 内置只有 16 条，靠 --openings-file（默认 data/openings_200.txt）扩容
        self.assertEqual(len(self.mod.OPENINGS), 16)


class TestLoadOpeningsFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_tool()

    def _write(self, lines: list[str]) -> str:
        fh = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        fh.write("\n".join(lines) + "\n")
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_valid_dedup_and_full_line(self):
        # arena 侧不截断：整行都是开局前缀，去重在最终形态上做
        path = self._write([
            "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6",   # 保留整行
            "d4 d5 c4 e6",                     # 短行：原样保留
            "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6",    # 与第 1 行完全相同：去重
            "d4 d5 c4 e6 Nf3 Nf6",             # 以第 2 行为前缀但更长：另一条
            "",
        ])
        got = self.mod.load_openings_file(path)
        self.assertEqual(got, [
            "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6",
            "d4 d5 c4 e6",
            "d4 d5 c4 e6 Nf3 Nf6",
        ])

    def test_truncation_collapses_library(self):
        # 按固定 ply 裁切会折叠多样性（200 → 114），arena 因此默认不截断
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo, "data", "openings_200.txt")
        if not os.path.exists(path):
            self.skipTest("data/openings_200.txt 不存在")
        full = self.mod.load_openings_file(path)
        cut = self.mod.load_openings_file(path, book_plies=6)
        self.assertEqual(len(full), 200)
        self.assertLess(len(cut), len(full), "固定 ply 裁切必然折叠开局多样性")

    def test_illegal_line_skipped(self):
        path = self._write([
            "e4 e5 Nf3 Nc6 Bb5",   # 合法
            "d4 d5 Ke2",           # Ke2 不合法（王被兵挡路）：整条跳过
            "e4 e5",               # 合法
        ])
        got = self.mod.load_openings_file(path)
        self.assertEqual(got, ["e4 e5 Nf3 Nc6 Bb5", "e4 e5"])

    def test_handles_crlf_and_blank_lines(self):
        path = self._write(["e4 e5", "   ", "d4 d5", ""])
        with open(path, "rb") as fh:  # 模拟 Windows CRLF
            data = fh.read().replace(b"\n", b"\r\n")
        with open(path, "wb") as fh:
            fh.write(data)
        self.assertEqual(self.mod.load_openings_file(path), ["e4 e5", "d4 d5"])


@unittest.skipUnless(_HAS_CHESS, "需要 python-chess")
class TestRealOpeningsBook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_tool()

    def test_openings_file_large_enough(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "openings_200.txt")
        if not os.path.exists(path):
            self.skipTest("data/openings_200.txt 不存在")
        lib = self.mod.load_openings_file(path)
        # 400 局门禁 => 200 对 => 需要至少 200 个开局才能零重复
        self.assertGreaterEqual(len(lib), 200,
                                "开局库不足 200：g=0 的 arena 无法给出 400 个不同对局")


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
