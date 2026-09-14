"""g：残差动力学侧枝（仅训练；设计文档 §5.6）。

emb_a: 1936×512 动作嵌入表
u_t = g([RMSNorm(h_{t-1}); emb_a(a_t)])   # MLP 1024→1024→512，残差
Δ̂_t = predictor(u_t)                       # 512→512 两层 MLP
x̂_t = sg(x_{t-1}) + Δ̂_t                   # L_dyn = (1/d)‖Δ̂ − sg(x_t − x_{t-1})‖²

t=0 时 h_{-1} := h_init（可学向量）。定位：表示塑形（原因侧收梯度，答案侧 sg 保护）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actions import NUM_ACTIONS
from .layers import RMSNorm

D_MODEL = 512


class DynamicsG(nn.Module):
    """(h_{t-1}, a_t) → Δ̂；配合 sg 残差基线 x̂_t = sg(x_{t-1}) + Δ̂。"""

    def __init__(self, d_model: int = D_MODEL, n_actions: int = NUM_ACTIONS):
        super().__init__()
        self.emb_a = nn.Embedding(n_actions, d_model)
        self.h_init = nn.Parameter(torch.zeros(d_model))
        self.norm = RMSNorm(d_model)
        self.fc1 = nn.Linear(2 * d_model, 1024)
        self.fc2 = nn.Linear(1024, d_model)
        self.predictor_fc1 = nn.Linear(d_model, d_model)
        self.predictor_fc2 = nn.Linear(d_model, d_model)

    def forward(self, h_prev: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """h_prev (..., d)、action (...,) int64 → Δ̂ (..., d)。"""
        z = torch.cat([self.norm(h_prev), self.emb_a(action.long())], dim=-1)
        u = z + F.gelu(self.fc2(F.gelu(self.fc1(z))))  # MLP 1024→1024→512，残差
        return self.predictor_fc2(F.gelu(self.predictor_fc1(u)))
