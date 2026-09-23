"""扩展深度直方图（P3 埋点，供 P4 在 SlabCacheStore / PrefillReplayStore 间取舍）。

每次模拟恰好展开一个新节点；深度 d 的节点要从对局根 cache 重放 d-1 步再做 1 次评估前向
（终局节点不评估）。直方图 ``hist[d]`` = 深度为 d 的展开次数（d>=1，下标 0 恒为 0），
纯 list[int]，可直接 JSON 化、跨局/跨进程相加。
"""

from __future__ import annotations


def hist_add(hist: list[int], depth: int) -> None:
    """原地记一次深度为 ``depth`` 的展开。"""
    if depth >= len(hist):
        hist.extend([0] * (depth + 1 - len(hist)))
    hist[depth] += 1


def hist_merge(a: list[int], b: list[int] | None) -> list[int]:
    """返回 a+b（逐深度相加）；b 为 None/空时原样拷贝 a。"""
    out = list(a)
    for d, n in enumerate(b or []):
        if d >= len(out):
            out.extend([0] * (d + 1 - len(out)))
        out[d] += int(n)
    return out


def _percentile(hist: list[int], total: int, q: float) -> int:
    target = q * total
    acc = 0
    for d, n in enumerate(hist):
        acc += n
        if n and acc >= target:
            return d
    return len(hist) - 1


def hist_summary(hist: list[int]) -> dict:
    """直方图 → 摘要：展开数、均值、分位、最大深度、重放前向数（Σ(d-1)·n）。"""
    total = sum(hist)
    if total == 0:
        return {"hist": list(hist), "expansions": 0, "mean": 0.0, "p50": 0, "p90": 0,
                "p99": 0, "max": 0, "replay_forwards": 0}
    return {
        "hist": list(hist),
        "expansions": total,
        "mean": sum(d * n for d, n in enumerate(hist)) / total,
        "p50": _percentile(hist, total, 0.50),
        "p90": _percentile(hist, total, 0.90),
        "p99": _percentile(hist, total, 0.99),
        "max": max(d for d, n in enumerate(hist) if n),
        "replay_forwards": sum((d - 1) * n for d, n in enumerate(hist) if d >= 1),
    }
