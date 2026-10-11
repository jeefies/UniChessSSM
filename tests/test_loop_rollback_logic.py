"""单元测试：验证五代非回滚窗口与安全熔断回滚逻辑"""
import unittest

def determine_train_base(history: list[dict], current_champ: str, prev_cand: str | None) -> tuple[str, str]:
    """模拟 loop_pipeline.py 中的 base_pt 决策逻辑 (支持多轮 5 代非回滚周期)"""
    consecutive_fails = 0
    for h in reversed(history):
        if h.get("promoted", False):
            break
        consecutive_fails += 1

    window_phase = consecutive_fails % 5  # 0: 周期第1代(新王/回滚重置), 1..4: 连续推进代
    if window_phase != 0 and prev_cand:
        return prev_cand, "continuous"
    elif consecutive_fails > 0 and window_phase == 0:
        return current_champ, "rollback"
    else:
        return current_champ, "initial"

class TestLoopRollback(unittest.TestCase):
    def test_initial_run(self):
        # 没有任何历史，从 Champion 启动
        base, mode = determine_train_base([], "champ.pt", None)
        self.assertEqual(base, "champ.pt")
        self.assertEqual(mode, "initial")

    def test_just_promoted(self):
        # 上一代刚晋升成功，从新 Champion 启动
        history = [
            {"generation": 1, "promoted": True}
        ]
        base, mode = determine_train_base(history, "champ_gen1.pt", "cand_gen1.pt")
        self.assertEqual(base, "champ_gen1.pt")
        self.assertEqual(mode, "initial")

    def test_continuous_within_5_gens(self):
        # 连续 1~4 代未晋升，必须不回滚，从上一代 candidate 连续向前推进
        for fail_count in [1, 2, 3, 4]:
            history = [{"generation": 0, "promoted": True}] + [
                {"generation": i + 1, "promoted": False} for i in range(fail_count)
            ]
            base, mode = determine_train_base(history, "champ.pt", f"cand_gen{fail_count}.pt")
            self.assertEqual(base, f"cand_gen{fail_count}.pt", f"Failed at fail_count={fail_count}")
            self.assertEqual(mode, "continuous")

    def test_rollback_at_5_gens(self):
        # 连续 5 代冲击未果，触发安全熔断回滚至 Champion
        history = [{"generation": 0, "promoted": True}] + [
            {"generation": i + 1, "promoted": False} for i in range(5)
        ]
        base, mode = determine_train_base(history, "champ.pt", "cand_gen5.pt")
        self.assertEqual(base, "champ.pt")
        self.assertEqual(mode, "rollback")

    def test_multi_cycle_progression(self):
        # 验证经历过一次熔断回滚后，下一个 5 代周期依然正常进入连续推进
        # fail_count = 6 (新周期的第 2 代，应该 continuous 从 cand_gen6 推进)
        history = [{"generation": 0, "promoted": True}] + [
            {"generation": i + 1, "promoted": False} for i in range(6)
        ]
        base, mode = determine_train_base(history, "champ.pt", "cand_gen6.pt")
        self.assertEqual(base, "cand_gen6.pt")
        self.assertEqual(mode, "continuous")

        # fail_count = 10 (第二个周期的末尾，连续第 10 次失败，再次熔断回滚)
        history_10 = [{"generation": 0, "promoted": True}] + [
            {"generation": i + 1, "promoted": False} for i in range(10)
        ]
        base_10, mode_10 = determine_train_base(history_10, "champ.pt", "cand_gen10.pt")
        self.assertEqual(base_10, "champ.pt")
        self.assertEqual(mode_10, "rollback")

    def test_reset_after_new_promotion(self):
        # 连续 3 代失败后，第 4 代晋升成功，计数重置
        history = [
            {"generation": 1, "promoted": False},
            {"generation": 2, "promoted": False},
            {"generation": 3, "promoted": False},
            {"generation": 4, "promoted": True},
            {"generation": 5, "promoted": False},
        ]
        # Gen 6 应该从 cand_gen5 连续训练（因为距离上一次晋升仅 1 代失败）
        base, mode = determine_train_base(history, "champ_gen4.pt", "cand_gen5.pt")
        self.assertEqual(base, "cand_gen5.pt")
        self.assertEqual(mode, "continuous")

if __name__ == "__main__":
    unittest.main()
