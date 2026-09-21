"""条件输入（设计文档 §3.4，D2）：[time_control][elo][color]，各投影到 R^512 后与 x_t 相加。

- time_control：离散桶 embedding（bullet/blitz/rapid/classical/correspondence/other/unknown）；
- elo：双方平均 Elo，标准化到零均值单位方差后线性投影；
- color：行棋方 2 类 one-hot。
推理时置最大 Elo 桶求最强、置目标分段得人类化风格；缺失元数据用 "unknown" 桶。
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    _HAS_TORCH = False

D_MODEL = 512


class TimeControlBucket(IntEnum):
    BULLET = 0
    BLITZ = 1
    RAPID = 2
    CLASSICAL = 3
    CORRESPONDENCE = 4
    OTHER = 5
    UNKNOWN = 6


NUM_TC_BUCKETS = 7


def time_control_bucket(tc: str | None) -> TimeControlBucket:
    """PGN TimeControl 标签 → 桶。口径：有效秒数 = 基础秒 + 40×加秒；通信棋为 */days。"""
    if not tc or tc in ("-", "?"):
        return TimeControlBucket.UNKNOWN
    if "/" in tc:  # 如 1/86400：每天一手 → 通信棋
        return TimeControlBucket.CORRESPONDENCE
    try:
        base, _, inc = tc.partition("+")
        seconds = int(base) + 40 * int(inc or 0)
    except ValueError:
        return TimeControlBucket.UNKNOWN
    if seconds < 180:
        return TimeControlBucket.BULLET
    if seconds < 480:
        return TimeControlBucket.BLITZ
    if seconds < 1500:
        return TimeControlBucket.RAPID
    return TimeControlBucket.CLASSICAL


class EloStandardizer:
    """Elo 标准化器：fit 数据集 P1/P99 截断的均值/方差（§4.3 的 e_min/e_max 同源统计）。"""

    def __init__(self) -> None:
        self.mean = 1500.0
        self.std = 500.0

    def fit(self, elos: np.ndarray) -> "EloStandardizer":
        elos = np.asarray(elos, dtype=np.float64)
        lo, hi = np.percentile(elos, [1.0, 99.0])
        clipped = np.clip(elos, lo, hi)
        self.mean = float(clipped.mean())
        self.std = float(max(clipped.std(), 1.0))
        return self

    def transform(self, elo: float | np.ndarray) -> np.ndarray:
        return (np.asarray(elo, dtype=np.float64) - self.mean) / self.std


_NN_BASE = nn.Module if _HAS_TORCH else object


class ConditionEmbedder(_NN_BASE):
    """三个条件向量各投影到 R^512 相加（tc embedding / elo 线性 / color embedding）。"""

    def __init__(self, d_model: int = D_MODEL):
        if not _HAS_TORCH:
            raise RuntimeError("ConditionEmbedder requires torch to be installed.")
        super().__init__()
        self.tc_emb = nn.Embedding(NUM_TC_BUCKETS, d_model)
        self.elo_proj = nn.Linear(1, d_model)
        self.color_emb = nn.Embedding(2, d_model)

    def forward(
        self,
        tc_bucket: torch.Tensor,   # (...,) int64
        elo_std: torch.Tensor,     # (...) float
        color: torch.Tensor,       # (...,) int64（0=黑走 1=白走）
    ) -> torch.Tensor:
        cond = (
            self.tc_emb(tc_bucket.long())
            + self.elo_proj(elo_std.float().unsqueeze(-1))
            + self.color_emb(color.long())
        )
        return cond  # (..., d_model)
