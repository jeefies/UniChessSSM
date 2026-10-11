#!/usr/bin/env python3
"""
Unit test for Information-Theoretic Surprise-based PER.
Tests mathematical edge cases, asymptotic limits, and numerical stability.
"""
import numpy as np

def compute_surprise(res: np.ndarray, p: float, clip_min=0.05, clip_max=5.0) -> np.ndarray:
    p_clamped = np.clip(p, 0.02, 0.98)
    info = np.zeros_like(res, dtype=np.float64)
    info[res == 0] = -np.log(p_clamped)           # 胜局
    info[res == 2] = -np.log(1.0 - p_clamped)     # 负局
    info[res == 1] = -0.5 * np.log(p_clamped * (1.0 - p_clamped))  # 和局
    return np.clip(info, clip_min, clip_max)

def test_symmetry_at_fair_matchup():
    """测试五五开自博弈时的客观对称性"""
    res = np.array([0, 1, 2]) # 胜, 和, 负
    w = compute_surprise(res, 0.5)
    # 当 p=0.5 时，-log(0.5) = 0.693147, -0.5*log(0.25) = 0.693147
    assert np.allclose(w[0], w[1], atol=1e-5), f"胜和不对称: {w[0]} vs {w[1]}"
    assert np.allclose(w[0], w[2], atol=1e-5), f"胜负不对称: {w[0]} vs {w[2]}"
    print("[PASS] Fair matchup symmetry confirmed")

def test_weak_opponent_limits():
    """测试面对弱对手 (p=0.98) 时的极限表现"""
    res = np.array([0, 1, 2]) # 胜, 和, 负
    w = compute_surprise(res, 0.98)
    # 赢局符合预期，信息量极小
    assert w[0] <= 0.06, f"虐菜赢局权重应趋近于 0，实测: {w[0]}"
    # 翻车输局是爆冷，信息量极大
    assert w[2] >= 3.8, f"虐菜翻车输局权重应打满，实测: {w[2]}"
    # 和棋是被逼平，信息量很高
    assert w[1] >= 1.9, f"虐菜被逼平权重应很高，实测: {w[1]}"
    print("[PASS] Weak opponent limits confirmed")

def test_strong_opponent_limits():
    """测试面对强敌 (p=0.02) 时的极限表现"""
    res = np.array([0, 1, 2]) # 胜, 和, 负
    w = compute_surprise(res, 0.02)
    # 输局符合预期，信息量极小
    assert w[2] <= 0.06, f"强敌输局应被视为预期内，实测: {w[2]}"
    # 逆袭赢局是奇迹，信息量极大
    assert w[0] >= 3.8, f"强敌逆袭赢局应打满，实测: {w[0]}"
    # 和棋是顽强守和，信息量很高
    assert w[1] >= 1.9, f"强敌顽强守和应很高，实测: {w[1]}"
    print("[PASS] Strong opponent limits confirmed")

def test_extreme_inputs():
    """测试极端异常输入下的数值稳定性 (不崩、不产生 NaN/Inf)"""
    res = np.array([0, 1, 2, 0, 2])
    for p_ext in [-1.0, 0.0, 1.0, 2.0]:
        w = compute_surprise(res, p_ext)
        assert not np.isnan(w).any(), f"p={p_ext} 产生了 NaN"
        assert not np.isinf(w).any(), f"p={p_ext} 产生了 Inf"
        assert (w >= 0.05).all() and (w <= 5.0).all(), f"超出门限: {w}"
    print("[PASS] Numerical extreme inputs confirmed")

if __name__ == "__main__":
    test_symmetry_at_fair_matchup()
    test_weak_opponent_limits()
    test_strong_opponent_limits()
    test_extreme_inputs()
    print("ALL MATH UNIT TESTS PASSED!")
