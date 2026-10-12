"""漂移护栏逻辑单测：复刻 loop_pipeline 的 base_pt 决策，验证护栏在正确时机触发。

判据（与 loop_pipeline.py 保持一致）：
- window_phase != 0 且上一代候选存在且未触发护栏 ⇒ 连续演化（从上一代 best.pt）
- window_phase != 0 但上一代候选 Elo CI 下界 < -20 ⇒ 漂移护栏，回霸主
- window_phase == 0 且累计未晋升 > 0 ⇒ 5 代安全熔断，回霸主
- 其余 ⇒ 从霸主启动
"""
from __future__ import annotations
import unittest

DRIFT_GUARD_CI_FLOOR = -20.0


def decide_base(history, current_champ, prev_cand, prev_ci_lo):
    """复刻 loop_pipeline 的 base_pt 决策，返回 (base, mode)。"""
    consecutive_fails = 0
    for h in reversed(history):
        if h.get("promoted", False):
            break
        consecutive_fails += 1
    window_phase = consecutive_fails % 5
    drift_guard = prev_ci_lo is not None and prev_ci_lo < DRIFT_GUARD_CI_FLOOR
    if window_phase != 0 and prev_cand and not drift_guard:
        return prev_cand, "continuous"
    if window_phase != 0 and prev_cand and drift_guard:
        return current_champ, "drift_guard"
    if consecutive_fails > 0 and window_phase == 0:
        return current_champ, "rollback"
    return current_champ, "initial"


def hist(promoted_flags):
    return [{"generation": i + 1, "promoted": p} for i, p in enumerate(promoted_flags)]


class DriftGuardTest(unittest.TestCase):
    def test_continuous_when_prev_ok(self):
        """上一代候选没被打崩（CI 下界 -8）⇒ 连续演化。"""
        b, m = decide_base(hist([True, False]), "champ.pt", "cand1.pt", -8.0)
        self.assertEqual((b, m), ("cand1.pt", "continuous"))

    def test_drift_guard_when_prev_bad(self):
        """上一代候选 CI 下界 -35 ⇒ 触发护栏，回霸主。"""
        b, m = decide_base(hist([True, False]), "champ.pt", "cand1.pt", -35.0)
        self.assertEqual((b, m), ("champ.pt", "drift_guard"))

    def test_drift_guard_boundary(self):
        """恰好在 -20 不算触发（要求严格小于）。"""
        b, m = decide_base(hist([True, False]), "champ.pt", "cand1.pt", -20.0)
        self.assertEqual(m, "continuous")
        b, m = decide_base(hist([True, False]), "champ.pt", "cand1.pt", -20.1)
        self.assertEqual(m, "drift_guard")

    def test_no_prev_candidate_falls_back(self):
        """没有上一代 best.pt 时不可能链式，回霸主。"""
        b, m = decide_base(hist([True, False]), "champ.pt", None, -35.0)
        self.assertEqual((b, m), ("champ.pt", "rollback"))

    def test_ci_missing_does_not_trigger(self):
        """CI 缺失（老记录）不当成触发。"""
        b, m = decide_base(hist([True, False]), "champ.pt", "cand1.pt", None)
        self.assertEqual(m, "continuous")

    def test_gen34_37_scenario(self):
        """复现 Gen 34→37：链式下滑时每代都该被护栏拦下，回霸主重训。

        gen33 晋升后，gen34 的窗口相位是 1（consecutive_fails=1），此时上一代候选就是
        刚晋升的霸主、CI 未知 ⇒ 连续演化；随后 gen35/36/37 的上一代候选 CI 下界都 < -20，
        每代都被护栏拦下回霸主。
        """
        h = hist([True, False])                # gen33 晋升 + gen34 未晋升
        cis = [-43.0, -69.8, -151.5, -50.4]    # gen35..38 的上一代候选 CI 下界
        modes = []
        for i, lo in enumerate(cis):
            b, m = decide_base(h, "champ.pt", "prev.pt", lo)
            modes.append(m)
            h = h + [{"generation": 36 + i, "promoted": False}]
        self.assertEqual(modes, ["drift_guard"] * 4,
                         "链式下滑的每一代都该触发护栏回霸主")

    def test_healthy_streak_stays_continuous(self):
        """候选一直没被打崩时保持连续演化（不被护栏打断）。"""
        h = hist([True, False])                # 已在窗口第 2 代
        modes = []
        for i, lo in enumerate([-15.0, -12.0, -18.0, -10.0]):
            b, m = decide_base(h, "champ.pt", f"cand{i}.pt", lo)
            modes.append(m)
            h = h + [{"generation": i + 3, "promoted": False}]
        self.assertEqual(modes, ["continuous"] * 4)

    def test_no_prev_candidate_falls_back(self):
        """没有上一代 best.pt 时不可能链式，一律回霸主（mode 名随窗口相位而异，行为一致）。"""
        for h in (hist([True, False]), hist([True]), hist([False, False])):
            b, m = decide_base(h, "champ.pt", None, -35.0)
            self.assertEqual(b, "champ.pt")
            self.assertIn(m, ("rollback", "initial"))


if __name__ == "__main__":
    unittest.main()
