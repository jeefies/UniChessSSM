"""换代门禁（promotion_gate）单测：Elo CI 下界 ≥ −5 且局数 ≥ 1000 为主通道，SPRT H1 为快速通道。

导入的是 ``tools/loop_pipeline.py`` 里的**真实函数**（不是本文件的副本）——门禁是系统里
最要命的判断，必须锁在实际执行的那段代码上。``tools`` 不是包，按文件路径导入。

2026-10-12 重设计背景（详见 docs/value-target-experiments.md）：
- 旧门禁三条（SPRT H1 / 得分率≥55% / CI 下界 > −5）在 32 代真实数据上零翻转，
  但 elo1=35 的 SPRT 对 +14 Elo 的真实边际增益**结构上判不出 H1**（LLR 漂移为负），
  得分率 55% 对应 +45 Elo 也够不到——实际只有 CI 那条在工作。
- 新设计把 CI 通道提为主通道并加 1000 局下限，删掉被完全覆盖的得分率条件。
- 开局库同步扩容到 4000 条，Arena 提到 1200 局（600 pairs）零重复。
"""
from __future__ import annotations
import glob
import importlib.util
import json
import os as _os
import sys as _sys
import unittest

_HERE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_TOOLS = _os.path.join(_HERE, "tools")
if _TOOLS not in _sys.path:
    _sys.path.insert(0, _TOOLS)

import loop_pipeline  # noqa: E402


class PromotionGateTest(unittest.TestCase):
    """主通道：CI 下界 ≥ −5 且局数 ≥ 1000；快速通道：SPRT H1。"""

    def test_sprt_h1_alone_promotes(self):
        """H1 是快速通道——大跨步（如 gen 33 的 +45 Elo）不必等满 1200 局。"""
        ok, why = loop_pipeline.promotion_gate("H1", [-30.0, 40.0], games=220)
        self.assertTrue(ok)
        self.assertEqual(why, "SPRT H1")

    def test_sprt_h1_ignores_game_floor(self):
        """H1 不受 1000 局下限约束（序贯检验的判决在越界那一刻就已成立）。"""
        ok, _ = loop_pipeline.promotion_gate("H1", [-30.0, 40.0], games=50)
        self.assertTrue(ok)

    def test_ci_floor_alone_promotes(self):
        """CI 紧且够到 −5（Elo +10、CI [−3,+25]、1200 局）⇒ 晋级。"""
        ok, why = loop_pipeline.promotion_gate("H0", [-3.0, 25.0], games=1200)
        self.assertTrue(ok)
        self.assertIn("Elo CI 下界", why)

    def test_ci_floor_boundary_is_inclusive(self):
        """下界恰好在 −5 算达标（≥ 而非 >）。"""
        ok, _ = loop_pipeline.promotion_gate("H0", [-5.0, 30.0], games=1200)
        self.assertTrue(ok)
        ok, _ = loop_pipeline.promotion_gate("H0", [-5.1, 30.0], games=1200)
        self.assertFalse(ok)

    def test_ci_path_requires_min_games(self):
        """CI 达标但局数不足 1000 ⇒ 不晋级（几百局的 CI 是噪声）。"""
        ok, why = loop_pipeline.promotion_gate("H0", [-2.0, 25.0], games=999)
        self.assertFalse(ok)
        self.assertIn("局数不足", why)
        ok, _ = loop_pipeline.promotion_gate("H0", [-2.0, 25.0], games=1000)
        self.assertTrue(ok)

    def test_games_none_does_not_block_ci_path(self):
        """games 缺失时 CI 通道仍由 CI 本身判定。

        理由：CI 的宽度**本身就是样本量信号**——窄 CI 在数学上不可能来自小样本
        （±20 Elo 的半宽就要上千局）。所以局数下限只是"冗余保险"（防开局库退化导致
        duplicate_rate 虚高、报告 CI 窄于真实 CI），不是主判据；games 未知时无法
        执行这项检查，就交给 CI 判。真实流程里 arena summary 永远带 games 字段，
        这条只在老记录/单测里出现。
        """
        ok, why = loop_pipeline.promotion_gate("H0", [-2.0, 25.0], games=None)
        self.assertTrue(ok)
        self.assertIn("Elo CI 下界", why)
        # CI 宽时即使 games 缺失也过不了（宽度已经在说"样本不够"）
        ok, _ = loop_pipeline.promotion_gate("H0", [-50.0, 45.0], games=None)
        self.assertFalse(ok)

    def test_wide_ci_blocks(self):
        """CI 宽（分不出高下）时守擂。"""
        ok, why = loop_pipeline.promotion_gate("H0", [-50.0, 45.0], games=1200)
        self.assertFalse(ok)
        self.assertEqual(why, "未达门限")

    def test_missing_ci_is_safe(self):
        """CI 缺失（None）不能当成通过。"""
        ok, _ = loop_pipeline.promotion_gate("H0", [None, None], games=1200)
        self.assertFalse(ok)
        ok, _ = loop_pipeline.promotion_gate("H0", None, games=1200)
        self.assertFalse(ok)

    def test_h0_verdict_does_not_block_ci_path(self):
        """SPRT 判 H0 不该堵死 CI 通道——elo1=35 对 +14 Elo 结构上判不出 H1。"""
        ok, why = loop_pipeline.promotion_gate("H0", [-2.0, 20.0], games=1200)
        self.assertTrue(ok)

    def test_none_verdict_does_not_block_ci_path(self):
        """SPRT 无判决（verdict=None，打满上限）时同样走 CI 通道。"""
        ok, _ = loop_pipeline.promotion_gate(None, [-2.0, 20.0], games=1200)
        self.assertTrue(ok)

    def test_backtest_history_no_flip(self):
        """回测历史：新门禁不应让任何一代的判决翻转。

        注意这里必然零翻转的原因有两层，值得写清楚：
        ① SPRT H1 快速通道原样保留 ⇒ 历史上唯一靠它晋级的 gen 33 仍晋级；
        ② CI 通道新增 1000 局下限，而历史 Arena 最多 500 局 ⇒ CI 通道历史上不可达。
        所以"零翻转"不等于"新门禁和历史一样宽"——它恰恰说明新门禁在 CI 通道上
        严格得多，只是把判别力从"永远够不到的 55% 得分率"换成了"1200 局的 CI"。
        """
        rows = []
        for d in sorted(glob.glob(_os.path.join(_HERE, "runs/loop_stage_b/gen_*"))):
            p = _os.path.join(d, "arena.jsonl.summary.json")
            if not _os.path.exists(p):
                continue
            s = json.load(open(p, encoding="utf-8"))
            sprt = (s.get("sprt") or {}).get("verdict")
            score = s.get("score_a")
            games = s.get("games")
            ci = s.get("elo_ci95") or [None, None]
            old = (sprt == "H1") or (score is not None and score >= 0.55)
            new, _ = loop_pipeline.promotion_gate(sprt, ci, games=games)
            rows.append((old, new, games))
        if not rows:
            self.skipTest("本机没有 runs/loop_stage_b 历史产物（远端才有），回测跳过")
        for old, new, g in rows:
            self.assertEqual(old, new, f"新门禁改掉了历史判决（{g} 局那代）")
        # 历史上确有不足 1000 局的代 ⇒ CI 通道的局数下限确实在起作用
        small = [g for _, _, g in rows if g is not None and g < 1000]
        self.assertTrue(small, "历史里没有不足 1000 局的代，这条约束没被验证到")


class ArenaConfigInvariantTest(unittest.TestCase):
    """Arena 配置不变式：开局库规模必须 ≥ arena_pairs。

    2026-10-12 踩过的坑：库只有 200 条而 pairs=250 时，OpeningBook.plan 会循环复用
    开局 ⇒ 20% 的局是逐字节重复棋谱；而 match.py 的 elo_ci95 按原始局数算、
    duplicate_rate 只是诊断字段不参与校正 ⇒ CI 虚窄、门禁被误导。
    把 Arena 从 500 局提到 1200 局时，如果不同时扩容开局库，真实精度一点都不会提升
    （400 个不同开局封顶），报告出来的 CI 却会显著变窄——比不改更危险。
    """

    def test_openings_file_exists(self):
        self.assertTrue(_os.path.exists(loop_pipeline.OPENINGS_FILE),
                        f"开局库不存在：{loop_pipeline.OPENINGS_FILE}")

    def test_book_covers_arena_pairs(self):
        """库规模 ≥ arena_pairs ⇒ plan() 不循环复用 ⇒ 零重复。"""
        from Kit.rules.openings import OpeningBook
        book = OpeningBook.from_file(loop_pipeline.OPENINGS_FILE)
        self.assertGreaterEqual(
            len(book), loop_pipeline.ARENA_PAIRS,
            f"开局库只有 {len(book)} 条，撑不起 {loop_pipeline.ARENA_PAIRS} 对 "
            f"（{loop_pipeline.ARENA_PAIRS * 2} 局）——会循环复用开局产生重复棋谱")

    def test_book_is_deterministic(self):
        """同 seed 的 plan() 可复现（换代判定要能重放）。"""
        from Kit.rules.openings import OpeningBook
        book = OpeningBook.from_file(loop_pipeline.OPENINGS_FILE)
        a = [i for i, _ in book.plan(loop_pipeline.ARENA_PAIRS, 42)]
        b = [i for i, _ in book.plan(loop_pipeline.ARENA_PAIRS, 42)]
        self.assertEqual(a, b)

    def test_orchestrator_rejects_oversized_pairs(self):
        """arena_pairs 超过库规模时必须响亮报错，而不是静默产生重复局。"""
        from Kit.rules.openings import OpeningBook
        book = OpeningBook.from_file(loop_pipeline.OPENINGS_FILE)
        with self.assertRaises(ValueError):
            loop_pipeline.SsmLoopOrchestrator(
                out_dir=_os.path.join(_HERE, "runs", "_unused_invariant_test"),
                total_gens=1, games_per_gen=10,
                initial_champion="x.pt", opp_checkpoint="y.pt",
                arena_pairs=len(book) + 1,
                openings=loop_pipeline.OPENINGS_FILE)

    def test_orchestrator_rejects_missing_openings(self):
        with self.assertRaises(FileNotFoundError):
            loop_pipeline.SsmLoopOrchestrator(
                out_dir=_os.path.join(_HERE, "runs", "_unused_invariant_test"),
                total_gens=1, games_per_gen=10,
                initial_champion="x.pt", opp_checkpoint="y.pt",
                openings=_os.path.join(_HERE, "no_such_book.txt"))


if __name__ == "__main__":
    unittest.main()
