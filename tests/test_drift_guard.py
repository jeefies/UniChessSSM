"""漂移护栏逻辑单测：复刻 loop_pipeline 的 base_pt 决策，验证护栏在正确时机触发。

判据（与 loop_pipeline.py 保持一致，2026-10-12 重设计）：
- window_phase != 0 且上一代候选存在且 Elo 点估 ≥ 0 ⇒ 连续演化（从上一代 best.pt）
- window_phase != 0 但上一代候选 Elo 点估 < 0 ⇒ 漂移护栏，回霸主
- window_phase == 0 且累计未晋升 > 0 ⇒ 5 代安全熔断，回霸主
- 其余 ⇒ 从霸主启动

为什么从"CI 下界 < -20"改成"Elo 点估 < 0"（2026-10-12 根因取证）：
自对弈 --ckpt 每代都是 champion.pt，实测 gen 31-38 的数据分布全程持平（步数中位
103-108、和棋率 0%、将杀率 98.6-98.8%），所以漂移不是数据问题——是链式 init 与
数据生成模型脱钩。判据因此应该直接问"候选有没有变差"（点估 < 0），而不是问
"CI 下界跌没跌到 -20"。回测 12 代：新判据在 gen 39（点估 +9.10）和 gen 41（+4.20）
两处比旧判据更宽（这两代旧判据回滚、新判据链式），而 gen 34-37 链式坍缩四代点估
全为负，两种判据都能防住。
"""
from __future__ import annotations
import unittest

DRIFT_GUARD_ELO_FLOOR = 0.0


def decide_base(history, current_champ, prev_cand, prev_elo):
    """复刻 loop_pipeline 的 base_pt 决策，返回 (base, mode)。"""
    consecutive_fails = 0
    for h in reversed(history):
        if h.get("promoted", False):
            break
        consecutive_fails += 1
    window_phase = consecutive_fails % 5
    drift_guard = prev_elo is not None and prev_elo < DRIFT_GUARD_ELO_FLOOR
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
    def test_continuous_when_prev_not_worse(self):
        """上一代候选点估 +21（没变差）⇒ 连续演化。"""
        b, m = decide_base(hist([True, False]), "champ.pt", "cand1.pt", 21.0)
        self.assertEqual((b, m), ("cand1.pt", "continuous"))

    def test_drift_guard_when_prev_worse(self):
        """上一代候选点估 -85.65 ⇒ 触发护栏，回霸主。"""
        b, m = decide_base(hist([True, False]), "champ.pt", "cand1.pt", -85.65)
        self.assertEqual((b, m), ("champ.pt", "drift_guard"))

    def test_drift_guard_boundary(self):
        """点估恰好在 0 不算触发（要求严格小于）。"""
        b, m = decide_base(hist([True, False]), "champ.pt", "cand1.pt", 0.0)
        self.assertEqual(m, "continuous")
        b, m = decide_base(hist([True, False]), "champ.pt", "cand1.pt", -0.1)
        self.assertEqual(m, "drift_guard")

    def test_elo_missing_does_not_trigger(self):
        """点估缺失（老记录）不当成触发。"""
        b, m = decide_base(hist([True, False]), "champ.pt", "cand1.pt", None)
        self.assertEqual(m, "continuous")

    def test_gen34_37_scenario(self):
        """复现 Gen 34→37 链式坍缩：四代点估全为负，每代都该被护栏拦下回霸主。

        gen33 晋升后 gen34 窗口相位为 1（consecutive_fails=1），上一代候选是刚晋升的
        霸主、点估未知 ⇒ 连续演化；随后 gen35/36/37/38 的上一代候选点估全为负。
        """
        h = hist([True, False])
        elos = [-4.83, -29.30, -85.65, -11.48]   # gen34..37 的真实点估
        modes = []
        for i, e in enumerate(elos):
            b, m = decide_base(h, "champ.pt", "prev.pt", e)
            modes.append(m)
            h = h + [{"generation": 36 + i, "promoted": False}]
        self.assertEqual(modes, ["drift_guard"] * 4,
                         "链式下滑的每一代都该触发护栏回霸主")

    def test_gen38_42_scenario_real_values(self):
        """本轮 Gen 38-41 真实点估：新判据下 gen39/gen41 不再被误回滚。

        这两代旧判据（CI 下界 < -20）会回滚，但它们的点估分别是 +9.10 / +4.20
        （CI 中点），看着并没变差——这正是用户指出并修正的地方。
        只取前 4 代：任何 5 个连续整数都含 5 的倍数，第 5 次必撞五代熔断（另测）。
        """
        h = hist([True, False])
        elos = [21.02, 9.10, 20.32, 4.20]   # gen38..41 真实点估
        modes = []
        for i, e in enumerate(elos):
            b, m = decide_base(h, "champ.pt", f"cand{i}.pt", e)
            modes.append(m)
            h = h + [{"generation": 39 + i, "promoted": False}]
        self.assertEqual(modes, ["continuous"] * 4,
                         "点估全部为正 ⇒ 保持连续演化，不被护栏误伤")

    def test_healthy_streak_stays_continuous(self):
        """候选一直没变差时保持连续演化（不被护栏打断）。"""
        h = hist([True, False])
        modes = []
        for i, e in enumerate([5.0, 12.0, 3.0, 8.0]):
            b, m = decide_base(h, "champ.pt", f"cand{i}.pt", e)
            modes.append(m)
            h = h + [{"generation": i + 3, "promoted": False}]
        self.assertEqual(modes, ["continuous"] * 4)

    def test_no_prev_candidate_falls_back(self):
        """没有上一代 best.pt 时不可能链式，一律回霸主（mode 名随窗口相位而异）。"""
        for h in (hist([True, False]), hist([True]), hist([False, False])):
            b, m = decide_base(h, "champ.pt", None, -35.0)
            self.assertEqual(b, "champ.pt")
            self.assertIn(m, ("rollback", "initial"))

    def test_five_gen_circuit_breaker(self):
        """连续 5 代未晋升 ⇒ 窗口相位归 0，触发安全熔断回霸主。"""
        h = hist([True] + [False] * 5)   # 1 次晋升 + 5 次失败 ⇒ consecutive_fails=5
        b, m = decide_base(h, "champ.pt", "cand.pt", 15.0)
        self.assertEqual((b, m), ("champ.pt", "rollback"))


if __name__ == "__main__":
    unittest.main()
