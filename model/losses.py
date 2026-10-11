"""损失函数（设计文档 §6；归一化口径统一，强制）。

每步 t：π_t = policy target（Stage A=人类走子 one-hot），y_t=对局结果（对行棋方归一），e=双方平均 Elo。

L_policy = Σ_t w(e)·CE(π_t, p(·|s_t)) / Σ_t w(e)     # 按有效权重和归一
L_value  = mean_t CE((W,D,L)_t, y_t)                  # value_target="mc"（硬标签）
L_value  = mean_t w_t·CE((W,D,L)_t, proj(G^λ_t))      # value_target="lambda"（软目标，D2/D4）
L_mlh    = mean_t Huber(m̂_t − m_t)
L_recon  = mean_{t,sq} CE(B̂_t[sq], B_t[sq]) + 0.3·辅助位损失     # 按格平均（不是求和）
L_dyn    = mean_t (1/d)·‖Δ̂_t − sg(x_t − x_{t-1})‖²
L_mat    = mean_t Huber(mat_t − (白方子力 − 黑方子力)/10)        # 子力差辅助（D9，仅训练）

初值：w_p=1.0, w_v=0.8, w_m=0.1, w_d=0.5；w_r 从 1.0 线性退火到 0.1（前 30% 训练步）。
上线前按首 1000 步各分量对共享主干的梯度范数校准（§6）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .d import HALFMOVE_BUCKETS, board_classes, halfmove_bucket

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
                     pos_mask: torch.Tensor | None = None, eps: float = 1e-8,
                     kl_reweight: bool = False) -> torch.Tensor:
    """软目标 CE：−Σ π′(a)·log p(a)；非法位置目标概率为 0，此处只做稳定求和。

    kl_reweight: 是否启用高 KL 难样本批内自适应加权（相对均值开方压缩，双向 clamp 0.3~3.0）。
    """
    log_p = F.log_softmax(logits, dim=-1)
    ce = -(target_probs * log_p).sum(dim=-1).reshape(-1)  # (B,T) -> (B*T,)，对齐 policy_loss 的展平口径
    w = _mask(weights.reshape(-1).float(), pos_mask.reshape(-1) if pos_mask is not None else None)
    if kl_reweight:
        mask_valid = (w > 0)
        if mask_valid.any():
            mean_ce = ce[mask_valid].mean().clamp(min=1e-5)
            kl_w = (ce / mean_ce).pow(0.5).clamp(min=0.3, max=3.0).detach()
            w = w * kl_w
    return (ce * w).sum() / w.sum().clamp(min=1e-8)


def value_loss(wdl_logits: torch.Tensor, result: torch.Tensor,
               pos_mask: torch.Tensor | None = None) -> torch.Tensor:
    """result: 0 胜 / 1 和 / 2 负（对行棋方归一）。"""
    ce = F.cross_entropy(wdl_logits.reshape(-1, 3), result.reshape(-1).long(), reduction="none")
    w = torch.ones_like(ce) if pos_mask is None else pos_mask.reshape(-1).float()
    return (ce * w).sum() / w.sum().clamp(min=1e-8)


# ------------------------- λ-return 价值目标（D2 / D4） -------------------------

#: 12 个占用平面的标准子力（白 P,N,B,R,Q,K / 黑 P,N,B,R,Q,K，王 0）
_PIECE_MAT_VALUES = (1.0, 3.0, 3.0, 5.0, 9.0, 0.0)


def to_mover_perspective(value_white: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
    """白方绝对视角 → 行棋方视角（color: 1=白走 / 0=黑走，同 TrainBatch.color 口径）。"""
    c = color.reshape(value_white.shape).to(value_white.dtype)
    return value_white * torch.where(c > 0.5, torch.ones_like(value_white),
                                     -torch.ones_like(value_white))


def z_white_from_results(results: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
    """results (B,T)（0 胜/1 和/2 负，行棋方归一）→ 白方绝对视角 z (B,T)，取值 +1/0/−1。"""
    z_mover = 1.0 - results.to(torch.float32)
    return to_mover_perspective(z_mover, color)


def _lambda_return_scan(q_white: torch.Tensor, q_valid: torch.Tensor, z_white: torch.Tensor,
                        lam: float, valid: torch.Tensor,
                        anchor_last: torch.Tensor, last: torch.Tensor,
                        anchor_val: torch.Tensor) -> torch.Tensor:
    """通用兜底：按定义逐位递推（部分 ply 有 q̂ 时才走这里）。"""
    g = torch.zeros_like(q_white)
    g.scatter_(1, last.clamp(min=0).unsqueeze(1), anchor_val.unsqueeze(1))
    for t in range(q_white.shape[1] - 2, -1, -1):
        q_eff = torch.where(q_valid[:, t + 1] & valid[:, t + 1], q_white[:, t + 1], g[:, t + 1])
        if lam >= 1.0:                      # 纯 MC：避开 0·NaN
            cand = g[:, t + 1]
        else:
            cand = (1.0 - lam) * q_eff + lam * g[:, t + 1]
        g[:, t] = torch.where(last > t, cand, g[:, t])
    return g


def _lambda_return_closed(q_white: torch.Tensor, anchor_val: torch.Tensor,
                          last: torch.Tensor, lam: float) -> torch.Tensor:
    """闭式：全部有效位都有 q̂ 时一次 bmm 算完。

    g[t] = Σ_{s=t+1}^{last} (1−λ)·λ^{s−t−1}·q[s] + λ^{last−t}·anchor

    B=4,T=300 上递推要 ~9.6 ms、闭式 ~0.12 ms；训练每步调 32 次，差 14% 吞吐。
    λ≥1 时闭式退化为"g ≡ anchor"，直接返回以免 0·NaN。
    """
    bsz, seqlen = q_white.shape
    dev = q_white.device
    idx = torch.arange(seqlen, device=dev)
    keep = last.unsqueeze(1) >= idx.unsqueeze(0)                 # (B,T)：t ≤ last
    if lam >= 1.0:
        g = anchor_val.unsqueeze(1).expand(bsz, seqlen).clone()
        return torch.where(keep, g, torch.zeros_like(g))
    expo = (idx[None, :] - idx[:, None] - 1).clamp(min=0).to(q_white.dtype)      # (T,T)
    base = (1.0 - lam) * torch.pow(torch.full_like(expo, lam), expo)             # (T,T)
    upper = (idx[None, :] > idx[:, None]).unsqueeze(0)                           # (1,T,T) s>t
    within = (idx[None, :].expand(bsz, seqlen) <= last[:, None]).unsqueeze(1)    # (B,1,T) s≤last
    mat = base.unsqueeze(0) * (upper & within).to(q_white.dtype)                 # (B,T,T)
    decay = torch.pow(torch.full((bsz, seqlen), lam, device=dev, dtype=q_white.dtype),
                      (last[:, None] - idx[None, :]).clamp(min=0).to(q_white.dtype))
    g = torch.bmm(mat, q_white.unsqueeze(-1)).squeeze(-1) + decay * anchor_val.unsqueeze(1)
    return torch.where(keep, g, torch.zeros_like(g))


def lambda_return(q_white: torch.Tensor, q_valid: torch.Tensor, z_white: torch.Tensor,
                  lam: float, valid: torch.Tensor | None = None,
                  anchor_last: torch.Tensor | None = None) -> torch.Tensor:
    """TD(λ) 价值目标（白方绝对视角；无中间奖励 ⇒ 向后递推，D2）。

    q_white (B,T)：每 ply 搜索价值 Σ_a π′(a)·completedQ(a)（白方绝对视角）
    q_valid (B,T)：该 ply 的 q̂ 是否可用。不可用时用后续的 g 代替 ⇒ 该步不引入自举、
                   权重自动归一到 g[t+1]；因此"整局无 q̂"的老分片自动退化为纯 MC。
    z_white (B,)：终局结果（白方绝对视角，+1/0/−1）
    anchor_last (B,)：True ⇒ 真实终局不在加载窗口内（截断局 / 被 t_max 截断），
                   锚点改用 q̂[last]（截断局的 z=0 是假标签）

    递推：g[last] = anchor；g[t] = (1−λ)·q_eff[t+1] + λ·g[t+1]。
    λ=1 ⇒ g ≡ z（= 现 MC 基线，可逐位对拍）；λ=0 ⇒ 纯一步自举（明确要避免的端点）。

    性能：全部有效位都有 q̂ 时走闭式（一次 bmm），只有"部分 ply 有 q̂"的混合批次才
    逐位递推——前者覆盖新分片与人类谱自举，是训练的热路径。
    """
    bsz, seqlen = q_white.shape
    dev = q_white.device
    lam = float(lam)
    if valid is None:
        valid = torch.ones(bsz, seqlen, dtype=torch.bool, device=dev)
    if anchor_last is None:
        anchor_last = torch.zeros(bsz, dtype=torch.bool, device=dev)
    idx = torch.arange(seqlen, device=dev).expand(bsz, seqlen)
    last = torch.where(valid, idx, torch.full_like(idx, -1)).max(dim=1).values   # (B,)
    safe_last = last.clamp(min=0)
    z_row = z_white.reshape(-1, 1).to(q_white.dtype).expand(bsz, seqlen)
    z_anchor = z_row.gather(1, safe_last.unsqueeze(1)).squeeze(1)
    q_anchor = q_white.gather(1, safe_last.unsqueeze(1)).squeeze(1)
    anchor_ok = anchor_last.reshape(-1) & q_valid.gather(1, safe_last.unsqueeze(1)).squeeze(1)
    anchor_val = torch.where(anchor_ok, q_anchor, z_anchor)

    if not bool((valid & ~q_valid).any()):
        # 热路径：有效位全有 q̂（新分片 / 人类谱自举）
        return _lambda_return_closed(q_white, anchor_val, last, lam)
    if not bool(q_valid.any()):
        # 老分片：整局无 q̂ ⇒ 纯 MC
        g = z_anchor.unsqueeze(1).expand(bsz, seqlen).clone()
        return torch.where(valid, g, torch.zeros_like(g))
    return _lambda_return_scan(q_white, q_valid, z_white, lam, valid, anchor_last,
                               last, anchor_val)


def value_soft_loss(wdl_logits: torch.Tensor, target_g: torch.Tensor,
                    pos_mask: torch.Tensor | None = None,
                    value_weights: torch.Tensor | None = None) -> torch.Tensor:
    """λ-return 标量目标 → WDL 三分类软目标 CE（D4）。

    target_g (B,T)：行棋方视角的 λ-return（∈ [−1,1]，期望得分）。
    目标分布取"均值 = g 的最大熵分布"：和棋质量 = 1−|g|，剩余质量按符号全给胜/负方——

        pW = max(g, 0),  pD = 1 − |g|,  pL = max(−g, 0)

    |g|=1 ⇒ 精确 one-hot 胜/负；g=0 ⇒ 精确 one-hot 和。于是 λ=1 时与硬标签 MC **逐位一致**，
    架构改动（价值塔 / 子力差辅助头）因此可以单独消融。

    Round 1 的教训：早期版本把和棋质量取成 clamp(pD.detach(), 0.05, 0.5)，g=0 的目标被稀释成
    ((1−d)/2, d, (1−d)/2)——和棋信号被冲淡，且 d 取自模型自身预测形成自举固定点。
    实测后果是优势局转换能力退化（Arena 和棋归因 41:5），故改为确定性投影。
    """
    logits = wdl_logits.reshape(-1, 3).float()
    g = target_g.reshape(-1).float().clamp(-1.0, 1.0)
    target = torch.stack([g.clamp(min=0.0), 1.0 - g.abs(), (-g).clamp(min=0.0)], dim=-1)
    log_p = torch.log_softmax(logits, dim=-1)
    ce = -(target * log_p).sum(dim=-1)
    w = torch.ones_like(ce)
    if pos_mask is not None:
        w = w * pos_mask.reshape(-1).float()
    if value_weights is not None:
        w = w * value_weights.reshape(-1).float()
    return (ce * w).sum() / w.sum().clamp(min=1e-8)


def material_loss(pred: torch.Tensor, features: torch.Tensor,
                  pos_mask: torch.Tensor | None = None, scale: float = 10.0) -> torch.Tensor:
    """子力差辅助（D9）：目标 = (白方标准子力 − 黑方标准子力)/scale。

    特征 [0,768) 即 12×64 占用平面（白 P,N,B,R,Q,K / 黑 P,N,B,R,Q,K），目标由重放棋盘
    精确算出 ⇒ 零存储、零标注成本，只受规则引擎约束。
    """
    planes = features[..., :768].reshape(*features.shape[:-1], 12, 64).float()
    counts = planes.sum(dim=-1)
    vals = torch.tensor(_PIECE_MAT_VALUES + _PIECE_MAT_VALUES,
                        device=features.device, dtype=torch.float32)
    white = (counts[..., :6] * vals[:6]).sum(dim=-1)
    black = (counts[..., 6:] * vals[6:]).sum(dim=-1)
    target = ((white - black) / float(scale)).reshape(-1)
    p = pred.reshape(-1).float()
    h = F.huber_loss(p, target, delta=1.0, reduction="none")
    w = torch.ones_like(h) if pos_mask is None else pos_mask.reshape(-1).float()
    return (h * w).sum() / w.sum().clamp(min=1e-8)


def mlh_loss(mlh_pred: torch.Tensor, moves_left: torch.Tensor,
             pos_mask: torch.Tensor | None = None,
             log_target: bool = False) -> torch.Tensor:
    """Huber(δ=1，或 log_target 时 δ=0.5) 于剩余 ply 预测；moves_left 以 ply 计、截断到 T_max=200/300。
    当 log_target 为 True 时，对预测值与目标值施加 torch.log1p(F.relu(...)) 变换，采用 delta=0.5。
    """
    pred = mlh_pred.reshape(-1).float()
    target = moves_left.reshape(-1).float()
    if log_target:
        pred = torch.log1p(F.relu(pred))
        target = torch.log1p(F.relu(target))
        delta = 0.5
    else:
        delta = 1.0
    h = F.huber_loss(pred, target, delta=delta, reduction="none")
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
