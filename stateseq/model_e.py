"""E：格子级 Transformer（设计文档 §5.2）。

逐格输入 s_i = piece_emb(c_i) + pos_emb(i) + W_g·globals（i=1..64）；
单块 TrmBlock（Pre-RMSNorm 残差：MHA + MLP₄ₓ）权重共享走两遍（K=1）；
可学单查询 q 对 64 格做 cross-attn 聚合 → 256 维，再 W_proj 到 512 并 RMSNorm。

E 只看当前局面（无跨步注意力）；推理时每节点一次。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .features import FEATURE_DIM, GLOBALS_DIM, PIECE_PLANES
from .layers import RMSNorm

D_E = 256
D_MODEL = 512


class TransformerBlock(nn.Module):
    """Pre-RMSNorm 残差块：S ← S + MHA(RMSNorm(S))；S ← S + MLP₄ₓ(RMSNorm(S))。"""

    def __init__(self, dim: int = D_E, heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        n = self.norm1(s)
        a, _ = self.attn(n, n, n, need_weights=False)
        s = s + self.drop(a)
        s = s + self.drop(self.fc2(self.drop(torch.nn.functional.gelu(self.fc1(self.norm2(s))))))
        return s


class BoardEncoderE(nn.Module):
    """局面特征 (..., 785) → x_t (..., 512)。"""

    def __init__(self, d_e: int = D_E, d_model: int = D_MODEL, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_e = d_e
        self.piece_emb = nn.Embedding(PIECE_PLANES + 1, d_e)   # 12 棋子平面 + 空格
        self.pos_emb = nn.Embedding(64, d_e)                    # 可学位置嵌入
        self.w_g = nn.Linear(GLOBALS_DIM, d_e)                  # globals（17 维非棋子字段）
        self.block = TransformerBlock(d_e, heads, dropout=dropout)  # 权重共享，走两遍
        self.q = nn.Parameter(torch.randn(1, 1, d_e) * 0.02)    # 可学单查询
        self.q_proj = nn.Linear(d_e, d_e, bias=False)
        self.k_proj = nn.Linear(d_e, d_e, bias=False)
        self.v_proj = nn.Linear(d_e, d_e, bias=False)
        self.attn_norm = RMSNorm(d_e)
        self.out_norm = RMSNorm(d_e)
        self.w_proj = nn.Linear(d_e, d_model, bias=False)
        self.final_norm = RMSNorm(d_model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (..., 785) → x_t: (..., 512)。"""
        *lead, feat_dim = features.shape
        assert feat_dim == FEATURE_DIM
        n = int(np.prod(lead)) if lead else 1
        flat = features.reshape(n, FEATURE_DIM)
        planes = flat[:, : PIECE_PLANES * 64].reshape(n, PIECE_PLANES, 64)
        occupied = planes.sum(dim=-2) > 0.5                      # (n, 64)
        piece_idx = planes.argmax(dim=-2)                        # (n, 64) 0..11
        piece_idx = torch.where(occupied, piece_idx, torch.full_like(piece_idx, PIECE_PLANES))

        s = (
            self.piece_emb(piece_idx)                            # (n, 64, d_e)
            + self.pos_emb.weight                                # (64, d_e)
            + self.w_g(flat[:, PIECE_PLANES * 64:]).unsqueeze(-2)
        )
        s = self.block(self.block(s))                            # 同一权重走两遍（K=1）
        # 单查询 cross-attn：x = RMSNorm(q + CrossAttn(RMSNorm(q), S, S))
        q = self.q.expand(n, 1, self.d_e)
        nq = self.attn_norm(q)
        k = self.k_proj(s)
        v = self.v_proj(s)
        attn = torch.softmax((nq @ k.transpose(-1, -2)) / (self.d_e ** 0.5), dim=-1)
        x = self.out_norm(q + attn @ v).squeeze(-2)              # (n, d_e)
        out = self.final_norm(self.w_proj(x))                    # (n, 512)
        return out.reshape(*lead, out.shape[-1]) if lead else out
