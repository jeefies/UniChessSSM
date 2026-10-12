"""换代门禁（promotion_gate）单测：SPRT H1 / 得分率≥55% / Elo CI 下界 > -5，任一即晋升。

导入的是 ``tools/loop_pipeline.py`` 里的**真实函数**（不是本文件的副本）——门禁是系统里
最要命的判断，必须锁在实际执行的那段代码上。``tools`` 不是包，按文件路径导入。
"""
from __future__ import annotations
import importlib.util
import os as _os
import sys as _sys
import unittest

_HERE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_TOOLS = _os.path.join(_HERE, "tools")
if _TOOLS not in _sys.path:
    _sys.path.insert(0, _TOOLS)

import loop_pipeline  # noqa: E402


class PromotionGateTest(unittest.TestCase):
    """三条任一满足即晋升；都不满足则守擂。"""

    def test_sprt_h1_alone_promotes(self):
        ok, why = loop_pipeline.promotion_gate("H1", 0.50, [-30.0, 40.0])
        self.assertTrue(ok)
        self.assertEqual(why, "SPRT H1")

    def test_score_55_alone_promotes(self):
        ok, why = loop_pipeline.promotion_gate("H0", 0.55, [-40.0, 30.0])
        self.assertTrue(ok)
        self.assertEqual(why, "得分率>=55%")

    def test_ci_floor_alone_promotes(self):
        """CI 很紧但够不到 H1（Elo +10、CI [−3,+25]）——正是新增这条要救济的情形。"""
        ok, why = loop_pipeline.promotion_gate("H0", 0.52, [-3.0, 25.0])
        self.assertTrue(ok)
        self.assertEqual(why, "Elo CI 下界 > -5")

    def test_ci_floor_boundary(self):
        # 下界恰好在阈值上不算（要求严格大于）
        ok, _ = loop_pipeline.promotion_gate("H0", 0.50, [-5.0, 30.0])
        self.assertFalse(ok)
        ok, _ = loop_pipeline.promotion_gate("H0", 0.50, [-4.9, 30.0])
        self.assertTrue(ok)

    def test_wide_ci_blocks(self):
        """CI 宽（分不出高下）时三条都不满足 ⇒ 守擂。"""
        ok, why = loop_pipeline.promotion_gate("H0", 0.49, [-50.0, 45.0])
        self.assertFalse(ok)
        self.assertEqual(why, "未达门限")

    def test_missing_ci_is_safe(self):
        """CI 缺失（None）不能当成通过。"""
        ok, _ = loop_pipeline.promotion_gate("H0", 0.50, [None, None])
        self.assertFalse(ok)
        ok, _ = loop_pipeline.promotion_gate("H0", 0.50, None)
        self.assertFalse(ok)

    def test_h0_verdict_does_not_block_ci_path(self):
        ok, why = loop_pipeline.promotion_gate("H0", 0.51, [-2.0, 20.0])
        self.assertTrue(ok)
        self.assertEqual(why, "Elo CI 下界 > -5")

    def test_backtest_history_no_flip(self):
        """回测 31 代历史：新门限不应让任何一代的判决翻转（CI 下界都远低于 -5）。"""
        import glob
        import json
        rows = []
        for d in sorted(glob.glob(_os.path.join(_HERE, "runs/loop_stage_b/gen_*"))):
            p = _os.path.join(d, "arena.jsonl.summary.json")
            if not _os.path.exists(p):
                continue
            s = json.load(open(p, encoding="utf-8"))
            sprt = (s.get("sprt") or {}).get("verdict")
            score = s.get("score_a")
            ci = s.get("elo_ci95") or [None, None]
            old = (sprt == "H1") or (score is not None and score >= 0.55)
            new, _ = loop_pipeline.promotion_gate(sprt, score, ci)
            rows.append((old, new))
        self.assertTrue(rows, "没读到历史 Arena 记录（远端才有 runs/，本机应跳过）")
        for old, new in rows:
            self.assertEqual(old, new, "新门限改改了历史判决")


if __name__ == "__main__":
    unittest.main()
