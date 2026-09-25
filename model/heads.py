"""f：预测头（设计文档 §5.4）。

u_t       = h_t + MLP₂(RMSNorm(h_t))          # 2 层共享小变换，残差
z_t^p     = W_p·u_t ∈ R^1936                   # policy logits（非法着由 mask 置 -inf）
(W,D,L)_t = softmax(W_v·RMSNorm(u_t))          # 三分类价值
m̂_t       = Mish(w_m·RMSNorm(u_t))             # 剩余 ply 预测（标量，Mish 恒正）
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..actions import NUM_ACTIONS
from .layers import RMSNorm, ResidualMLP

D_MODEL = 512


class PredictionHeads(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_actions: int = NUM_ACTIONS):
        super().__init__()
        self.transform = ResidualMLP(d_model, d_model)  # u = h + MLP₂(RMSNorm(h))
        self.w_p = nn.Linear(d_model, n_actions)
        self.w_v = nn.Linear(d_model, 3)
        self.w_m = nn.Linear(d_model, 1)
        self.head_norm = RMSNorm(d_model)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h (..., d) → (policy_logits (..., 1936), wdl_logits (..., 3), mlh (..., 1))。"""
        u = self.transform(h)
        policy_logits = self.w_p(u)
        n = self.head_norm(u)
        wdl_logits = self.w_v(n)
        mlh = nn.functional.mish(self.w_m(n))
        return policy_logits, wdl_logits, mlh


def apply_legal_mask(policy_logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """非法着置 -inf：p(a|s) = softmax(z + M)，M_t 非法 = -inf。"""
    neg = torch.finfo(policy_logits.dtype).min
    return policy_logits.masked_fill(~legal_mask.bool(), neg)
