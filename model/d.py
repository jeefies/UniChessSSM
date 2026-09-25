"""D：MLP 重建解码器（仅训练；设计文档 §5.5）。

B̂_t = reshape(W₂·GELU(W₁·RMSNorm(x_t)), 64, 13)；辅助位头：走子方(2 类) / 易位权(4 路独立 sigmoid) / 半回合计数分桶(16 类)。
定位：辅助监督与诊断工具，不是可逆性证书，不作为进入 RL 的门槛。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features import FEATURE_DIM
from .layers import RMSNorm

D_MODEL = 512
HALFMOVE_BUCKETS = 16


class ReconDecoderD(nn.Module):
    """x_t (512) → 棋盘重建 B̂ (64, 13：12 棋子平面+空格) + 辅助位头。"""

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.fc1 = nn.Linear(d_model, 1024)
        self.fc2 = nn.Linear(1024, 64 * 13)
        self.side_head = nn.Linear(d_model, 2)                    # 走子方
        self.castling_head = nn.Linear(d_model, 4)                # 易位权（独立 sigmoid）
        self.halfmove_head = nn.Linear(d_model, HALFMOVE_BUCKETS)  # 半回合计数分桶

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        n = self.norm(x)
        hidden = F.gelu(self.fc1(n))
        board_logits = self.fc2(hidden).reshape(*x.shape[:-1], 64, 13)
        return {
            "board_logits": board_logits,             # (..., 64, 13)
            "side_logits": self.side_head(n),         # (..., 2)
            "castling_logits": self.castling_head(n),  # (..., 4)
            "halfmove_logits": self.halfmove_head(n),  # (..., 16)
        }


def board_classes(features: torch.Tensor) -> torch.Tensor:
    """785 维特征 → 每格 13 类标签（12 棋子平面 + 空格=12），供 L_recon 使用。"""
    planes = features[..., : FEATURE_DIM - 17].reshape(*features.shape[:-1], 12, 64)
    occupied = planes.sum(dim=-2) > 0.5
    idx = planes.argmax(dim=-2)
    return torch.where(occupied, idx, torch.full_like(idx, 12))  # (..., 64)


def halfmove_bucket(halfmove_frac: torch.Tensor) -> torch.Tensor:
    """特征中的半回合计数（/100）→ 16 桶标签。"""
    return (halfmove_frac.clamp(0, 1) * HALFMOVE_BUCKETS).long().clamp(max=HALFMOVE_BUCKETS - 1)
