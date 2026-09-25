"""共享基础层：Pre-RMSNorm、残差 MLP 块（设计文档 §5.1）。

RMSNorm(u) = u / sqrt(mean(u²) + 1e-6) ⊙ γ   （无均值居中、无 β）
所有同形堆叠块统一形式：y = u + F(RMSNorm(u))；禁用 BatchNorm（D7）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """全局 Pre-RMSNorm（参数化增益 γ，元素级）。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # fp32 快路径：u.float() 与 u.to(fp32) 均为 no-op，跳过两次 dtype 调度。
        # 推理（arena/生成器）全程 fp32，profile 显示这两次 no-op .to() 约占每前向
        # CPU 时间的 30%。bf16 autocast 训练走原路径，行为不变。
        if u.dtype is torch.float32:
            return u * torch.rsqrt(u.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight
        dtype = u.dtype
        u = u.float()
        u = u * torch.rsqrt(u.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (u.to(dtype)) * self.weight


class ResidualMLP(nn.Module):
    """残差 MLP 块：y = u + W2·act(W1·RMSNorm(u))（同形堆叠统一形式）。"""

    def __init__(self, dim: int, hidden: int, act: nn.Module | None = None):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = act if act is not None else nn.GELU()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return u + self.fc2(self.act(self.fc1(self.norm(u))))
