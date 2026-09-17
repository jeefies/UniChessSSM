"""损失函数（设计文档 §6；归一化口径统一，强制）。

每步 t：π_t = policy target（Stage A=人类走子 one-hot），y_t=对局结果（对行棋方归一），e=双方平均 Elo。

L_policy = Σ_t w(e)·CE(π_t, p(·|s_t)) / Σ_t w(e)     # 按有效权重和归一
L_value  = mean_t CE((W,D,L)_t, y_t)
L_mlh    = mean_t Huber(m̂_t − m_t)
L_recon  = mean_{t,sq} CE(B̂_t[sq], B_t[sq]) + 0.3·辅助位损失     # 按格平均（不是求和）
L_dyn    = mean_t (1/d)·‖Δ̂_t − sg(x_t − x_{t-1})‖²

初值：w_p=1.0, w_v=0.8, w_m=0.1, w_d=0.5；w_r 从 1.0 线性退火到 0.1（前 30% 训练步）。
上线前按首 1000 步各分量对共享主干的梯度范数校准（§6）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model_d import HALFMOVE_BUCKETS, board_classes, halfmove_bucket

# 结果标签（对行棋方归一）：胜=0 和=1 负=2
RESULT_WIN, RESULT_DRAW, RESULT_LOSS = 0, 1, 2


def elo_weights(elo_mean: torch.Tensor, e_min: float, e_max: float, r: float = 20.0) -> torch.Tensor:
    """线性 Elo 加权（D9，r≈20；e_min/e_max 取数据分布 P1/P99）：w = (e−e_min)/(e_max−e_min)·(r−1)+1。"""
    span = max(e_max - e_min, 1.0)
    w = (elo_mean - e_min) / span * (r - 1.0) + 1.0
    return w.clamp(min=1.0, max=r)


def _mask(weights: torch.Tensor, pos_mask: torch.Tensor | None) -> torch.Tensor:
    """可选填充掩码（True=有效步）按位并入权重。"""
    if pos_mask is None:
        return weights
    return weights * pos_mask.float()


def policy_loss(logits: torch.Tensor, target_action: torch.Tensor, weights: torch.Tensor,
                pos_mask: torch.Tensor | None = None) -> torch.Tensor:
    """加权 CE，按有效权重和归一。logits/target/weights 任意同形批量维。"""
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_action.reshape(-1).long(), reduction="none")
    w = _mask(weights.reshape(-1).float(), pos_mask.reshape(-1) if pos_mask is not None else None)
    return (ce * w).sum() / w.sum().clamp(min=1e-8)


def policy_soft_loss(logits: torch.Tensor, target_probs: torch.Tensor, weights: torch.Tensor,
                     pos_mask: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    """软目标 CE：−Σ π′(a)·log p(a)；非法位置目标概率为 0，此处只做稳定求和。"""
    log_p = F.log_softmax(logits, dim=-1)
    ce = -(target_probs * log_p).sum(dim=-1).reshape(-1)  # (B,T) -> (B*T,)，对齐 policy_loss 的展平口径
    w = _mask(weights.reshape(-1).float(), pos_mask.reshape(-1) if pos_mask is not None else None)
    return (ce * w).sum() / w.sum().clamp(min=1e-8)


def value_loss(wdl_logits: torch.Tensor, result: torch.Tensor,
               pos_mask: torch.Tensor | None = None) -> torch.Tensor:
    """result: 0 胜 / 1 和 / 2 负（对行棋方归一）。"""
    ce = F.cross_entropy(wdl_logits.reshape(-1, 3), result.reshape(-1).long(), reduction="none")
    w = torch.ones_like(ce) if pos_mask is None else pos_mask.reshape(-1).float()
    return (ce * w).sum() / w.sum().clamp(min=1e-8)


def mlh_loss(mlh_pred: torch.Tensor, moves_left: torch.Tensor,
             pos_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Huber(δ=1) 于剩余 ply 预测；moves_left 以 ply 计、截断到 T_max=200。"""
    h = F.huber_loss(mlh_pred.reshape(-1).float(), moves_left.reshape(-1).float(), delta=1.0, reduction="none")
    w = torch.ones_like(h) if pos_mask is None else pos_mask.reshape(-1).float()
    return (h * w).sum() / w.sum().clamp(min=1e-8)


def recon_loss(d_out: dict[str, torch.Tensor], features: torch.Tensor,
               pos_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """L_recon：每格 CE（按格平均）+ 0.3·辅助位损失；pos_mask 屏蔽填充步。返回 (loss, 诊断指标)。"""
    target_cls = board_classes(features)                                   # (..., 64)
    per_sq = F.cross_entropy(
        d_out["board_logits"].reshape(-1, 13), target_cls.reshape(-1).long(), reduction="none"
    ).reshape(target_cls.shape)
    if pos_mask is not None:
        m = pos_mask.float().unsqueeze(-1)
        board_ce = (per_sq * m).sum() / m.sum().clamp(min=1e-8) / per_sq.shape[-1]
    else:
        board_ce = per_sq.mean()

    def _masked_mean(loss_vec: torch.Tensor) -> torch.Tensor:
        w = torch.ones_like(loss_vec) if pos_mask is None else pos_mask.reshape(-1).float()
        return (loss_vec * w).sum() / w.sum().clamp(min=1e-8)

    aux = (
        _masked_mean(F.cross_entropy(d_out["side_logits"].reshape(-1, 2), (features[..., 768] > 0.5).long().reshape(-1), reduction="none"))
        + _masked_mean(F.binary_cross_entropy_with_logits(
            d_out["castling_logits"].reshape(-1, 4), features[..., 769:773].reshape(-1, 4), reduction="none"
        ).mean(-1))
        + _masked_mean(F.cross_entropy(
            d_out["halfmove_logits"].reshape(-1, HALFMOVE_BUCKETS),
            halfmove_bucket(features[..., 781]).reshape(-1), reduction="none",
        ))
    )
    loss = board_ce + 0.3 * aux
    with torch.no_grad():
        pred_cls = d_out["board_logits"].argmax(dim=-1)  # (..., 64)
        if pos_mask is not None:
            whole = ((pred_cls == target_cls).all(dim=-1) & pos_mask.bool()).float().sum() / pos_mask.sum().clamp(min=1)
        else:
            whole = (pred_cls == target_cls).all(dim=-1).float().mean()
        diag = {
            "recon_board_ce": float(board_ce),
            "recon_whole_board_acc": float(whole),
        }
    return loss, diag


def dyn_loss(delta_hat: torch.Tensor, x_t: torch.Tensor, x_prev: torch.Tensor,
             pos_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """L_dyn = mean_t (1/d)·‖Δ̂_t − sg(x_t − x_{t-1})‖²；target 与 base 双侧 sg（§7.1 答案侧保护）。"""
    d = x_t.shape[-1]
    delta = (x_t - x_prev).detach()
    diff = delta_hat - delta
    per_t = diff.pow(2).sum(-1) / d
    if pos_mask is None:
        loss = per_t.mean()
    else:
        w = pos_mask.float()
        loss = (per_t * w).sum() / w.sum().clamp(min=1e-8)
    with torch.no_grad():
        num = diff.pow(2).sum(-1).mean()
        den = delta.pow(2).sum(-1).mean()
        diag = {"dyn_rel_err": float(num / den.clamp(min=1e-12))}
    return loss, diag


@dataclass
class LossWeights:
    """总损失权重；w_r 线性退火 1.0→0.1 于前 30% 训练步。"""

    w_p: float = 1.0
    w_v: float = 0.8
    w_m: float = 0.1
    w_d: float = 0.5
    w_r_start: float = 1.0
    w_r_end: float = 0.1
    anneal_frac: float = 0.3  # 前 30% 步退火

    def w_r(self, step: int, total_steps: int) -> float:
        t = min(step / max(int(total_steps * self.anneal_frac), 1), 1.0)
        return self.w_r_start + (self.w_r_end - self.w_r_start) * t
