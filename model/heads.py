"""f：预测头（设计文档 §5.4）。

u_t       = h_t + MLP₂(RMSNorm(h_t))          # 2 层共享小变换，残差
z_t^p     = W_p·u_t ∈ R^1936                   # policy logits（非法着由 mask 置 -inf）
(W,D,L)_t = softmax(W_v·ValueTower(RMSNorm(u_t)))   # 三分类价值（2026-10-11 价值塔）
m̂_t       = Mish(w_m·RMSNorm(u_t))             # 剩余 ply 预测（标量，Mish 恒正）
mat_t     = w_mat·RMSNorm(u_t)                 # 子力差辅助（仅训练，零初始化）

价值塔（D1）：2 个 ResidualMLP(512, 768) 残差块，**fc2 零初始化** ⇒ 初始时塔是恒等映射，
``W_v·RMSNorm(u_t)`` 与改造前的单层线性头逐位相同，旧 checkpoint 经 ``strict=False``
直接加载且 step 0 行为不变；塔只在此之上学"残差增量"。参数 +1.57M（26.40M → ≈27.97M，
仍在 D4 的 ~28M 预算内）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..actions import NUM_ACTIONS
from .layers import RMSNorm, ResidualMLP

D_MODEL = 512

#: 价值塔残差块数量与隐藏宽度（D1；hidden 768 时总参数 ≈27.97M，超 D4 上限就降 hidden）
VALUE_TOWER_BLOCKS = 2
VALUE_TOWER_HIDDEN = 768


class PredictionHeads(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_actions: int = NUM_ACTIONS):
        super().__init__()
        self.transform = ResidualMLP(d_model, d_model)  # u = h + MLP₂(RMSNorm(h))
        self.w_p = nn.Linear(d_model, n_actions)
        self.value_tower = nn.ModuleList([
            ResidualMLP(d_model, VALUE_TOWER_HIDDEN) for _ in range(VALUE_TOWER_BLOCKS)
        ])
        # 零初始化：塔输出 ≡ 输入，旧 checkpoint 的 w_v 直接可用且前向逐位不变。
        # 必须在这里置零（不能依赖 ckpt 加载器——缺键时会走默认随机初始化）。
        for blk in self.value_tower:
            nn.init.zeros_(blk.fc2.weight)
            nn.init.zeros_(blk.fc2.bias)
        self.w_v = nn.Linear(d_model, 3)
        self.w_m = nn.Linear(d_model, 1)
        # 子力差辅助头（KataGo score head 最小版）：零初始化 ⇒ 预测 0 = "子力相等"，
        # 本身就是一个合理先验；仅训练期使用，推理前向不碰它。
        self.w_mat = nn.Linear(d_model, 1)
        nn.init.zeros_(self.w_mat.weight)
        nn.init.zeros_(self.w_mat.bias)
        self.head_norm = RMSNorm(d_model)

    def value_features(self, n: torch.Tensor) -> torch.Tensor:
        """已归一化的 u → 价值塔输出（零初始化时恒等）。"""
        v = n
        for blk in self.value_tower:
            v = blk(v)
        return v

    def forward(self, h: torch.Tensor, return_material: bool = False):
        """h (..., d) → (policy_logits (..., 1936), wdl_logits (..., 3), mlh (..., 1))。

        ``return_material=True`` 时额外返回子力差辅助预测（..., 1)——只有 ``forward_train``
        用；默认三返回保持推理路径（fast_eval / gpu_server / engine）逐位不变。
        """
        u = self.transform(h)
        policy_logits = self.w_p(u)
        n = self.head_norm(u)
        wdl_logits = self.w_v(self.value_features(n))
        mlh = nn.functional.mish(self.w_m(n))
        if return_material:
            return policy_logits, wdl_logits, mlh, self.w_mat(n)
        return policy_logits, wdl_logits, mlh


def apply_legal_mask(policy_logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """非法着置 -inf：p(a|s) = softmax(z + M)，M_t 非法 = -inf。"""
    neg = torch.finfo(policy_logits.dtype).min
    return policy_logits.masked_fill(~legal_mask.bool(), neg)
