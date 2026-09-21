"""SPRT early stopping unit tests for ssm_gumbel_arena."""

import math
import unittest


def compute_llr(s: float, n: int, p0: float = 0.50, p1: float = 0.55) -> float:
    return s * math.log(p1 / p0) + (n - s) * math.log((1.0 - p1) / (1.0 - p0))


class TestArenaSPRT(unittest.TestCase):
    def test_sprt_llr_boundary(self):
        alpha, beta = 0.05, 0.05
        bound_b = math.log(beta / (1.0 - alpha))  # ~ -2.9444
        p0, p1 = 0.50, 0.55

        # 64 局中，如果候选 A 得分 <= 18（例如 round2 28.1% 得分为 18/64）
        # s = 18.0 时 llr = -3.1310 <= -2.9444，触发早停拒绝 H1
        llr_bad = compute_llr(18.0, 64, p0, p1)
        self.assertLessEqual(llr_bad, bound_b)

        # 64 局中，如果胜率均衡（32 胜 32 负，s = 32.0）
        llr_even = compute_llr(32.0, 64, p0, p1)
        self.assertLess(llr_even, 0.0)

        # 64 局中，如果候选 A 明显领先（40 胜 24 负，s = 40.0）
        llr_good = compute_llr(40.0, 64, p0, p1)
        self.assertGreater(llr_good, 0.0)
        self.assertGreater(llr_good, bound_b)


if __name__ == "__main__":
    unittest.main()
