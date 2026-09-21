"""SeqModel 总装：E → (+条件) → R → f 主路径；训练期 D / g 侧枝（设计文档 §3.1）。

主路径（推理 = 这条）：局面序列 B₀..B_T → E → x₀..x_T → R → h₀..h_T → f(policy/WDL/moves-left)。
训练期：D(x_t) 重建监督；g(h_{t-1}, a_t) → Δ̂ 对齐 sg(x_t − x_{t-1})（t≥1；t=0 无 x_{-1}，不入 L_dyn；
h_init 为推理起点隐状态，即 t=0 的 h_{-1}）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from . import losses
from .actions import NUM_ACTIONS
from .conditions import ConditionEmbedder
from .features import FEATURE_DIM
from .heads import PredictionHeads
from .layers import RMSNorm
from .model_d import ReconDecoderD
from .model_e import BoardEncoderE
from .model_g import DynamicsG
from .model_r import Cache, MambaTower


@dataclass
class TrainBatch:
    """整序列训练批（Stage A 全序列，T≤200 不用 TBPTT，§7.3）。"""

    features: torch.Tensor      # (B, T, 785)
    actions: torch.Tensor       # (B, T) int64：B_t 实际走的动作 a_t
    legal_mask: torch.Tensor    # (B, T, 1936) bool
    results: torch.Tensor       # (B, T) int64：0 胜/1 和/2 负（对行棋方归一）
    moves_left: torch.Tensor    # (B, T) float：剩余 ply，截断 200
    elo_weight: torch.Tensor    # (B,) float：对局级 Elo 权重（w(e) 已算好）
    tc_bucket: torch.Tensor     # (B,) int64（对局级；逐步同局一致）
    elo_std: torch.Tensor       # (B,) float
    color: torch.Tensor         # (B, T) int64：0 黑走 / 1 白走


class SeqModel(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        self.in_norm = RMSNorm(d_model)          # x_t 进 R 前的最后一层 RMSNorm（§5.1）
        self.cond = ConditionEmbedder(d_model)
        self.e = BoardEncoderE(d_model=d_model, dropout=dropout)
        self.r = MambaTower(d_model=d_model, dropout=dropout)
        self.f = PredictionHeads(d_model=d_model, n_actions=NUM_ACTIONS)
        self.d = ReconDecoderD(d_model=d_model)  # 仅训练
        self.g = DynamicsG(d_model=d_model)      # 仅训练

    # ---- 主路径 ----

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """E：(..., 785) → (..., 512)。"""
        return self.e(features)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """R：x (B, T, 512) → h (B, T, 512)（整序列并行扫描）。"""
        return self.r(self.in_norm(x))

    def _cond_expand(self, batch: "TrainBatch", shape: tuple[int, ...]) -> torch.Tensor:
        """条件向量按步展开：(B,) 对局级 tc/elo 广播到 (B,T)，color 逐步取值（D2）。"""
        bsz, seqlen = batch.features.shape[:2]
        tc = batch.tc_bucket.unsqueeze(1).expand(bsz, seqlen)
        elo = batch.elo_std.unsqueeze(1).expand(bsz, seqlen)
        cond = self.cond(tc, elo, batch.color)  # (B, T, d)
        return cond

    def forward_train(self, batch: TrainBatch, weights: losses.LossWeights, step: int, total_steps: int,
                      valid_mask: torch.Tensor | None = None,
                      policy_soft_target: torch.Tensor | None = None,
                      mlh_valid_mask: torch.Tensor | None = None,
                      log_target: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
        """整序列前向 + 五损失；valid_mask (B,T) 屏蔽填充步。返回 (总损失, 指标 dict)。

        policy_soft_target (B,T,NUM_ACTIONS)：Stage B 自对弈 π′ 软目标（§2.2 修正②，
        支持集=全部合法着）；提供时 policy 损失改用软 CE（losses.policy_soft_loss），
        其余四项损失（value/mlh/recon/dyn）计算方式不变——value 目标同为该局真实结果，
        dyn 仍用实战 batch.actions（§2.6：value/recon/dyn/mlh 代码零改动，只换 policy 监督）。
        mlh_valid_mask：默认等于 valid_mask；封顶截断局无真实终局，"剩余 ply"无定义，
        §2.5 要求整局剔除 mlh——调用方对截断局传入全 False 的行即可，其余损失不受影响。
        log_target：是否对 mlh 启用 Log-Huber 变换（torch.log1p(F.relu(...)), delta=0.5）。
        """
        bsz, seqlen, _ = batch.features.shape
        x = self.encode(batch.features)                                   # (B, T, 512)
        cond = self._cond_expand(batch, x.shape[:2] + (x.shape[-1],))
        h = self.trunk(x + cond)                                          # (B, T, 512)
        policy_logits, wdl_logits, mlh = self.f(h)

        # Apply legal mask to policy logits before loss
        if hasattr(batch, 'legal_mask') and batch.legal_mask is not None:
            from .heads import apply_legal_mask
            policy_logits_masked = apply_legal_mask(policy_logits, batch.legal_mask)
        else:
            policy_logits_masked = policy_logits

        if policy_soft_target is not None:
            l_pol = losses.policy_soft_loss(policy_logits_masked, policy_soft_target,
                                            batch.elo_weight.unsqueeze(1).expand(bsz, seqlen), valid_mask)
        else:
            l_pol = losses.policy_loss(policy_logits_masked, batch.actions,
                                       batch.elo_weight.unsqueeze(1).expand(bsz, seqlen), valid_mask)
        l_val = losses.value_loss(wdl_logits, batch.results, valid_mask)
        l_mlh = losses.mlh_loss(mlh, batch.moves_left,
                                mlh_valid_mask if mlh_valid_mask is not None else valid_mask,
                                log_target=log_target)

        d_out = self.d(x)
        l_rec, diag_rec = losses.recon_loss(d_out, batch.features, valid_mask)

        # g 动力学：t≥1（t=0 无 x_{-1}，不入 L_dyn）；valid 取 t≥1 段
        # actions[t-1] 从 B_t 推进到 B_{t+1}，配 h_{t-1} 预测 x_t - x_{t-1}
        # 注意：Stage A 训练使用 actions[:, 1:]，此处改为 actions[:, :-1] 对齐设计规格 §5.6 / §6。
        # 从 Stage A checkpoint 续训时，dyn 损失定义发生语义变化，前若干步需观察 dyn_rel_err 跳变。
        delta_hat = self.g(h[:, :-1], batch.actions[:, :-1])  # (h_{t-1}, a_{t-1})
        dyn_mask = valid_mask[:, 1:] if valid_mask is not None else None
        l_dyn, diag_dyn = losses.dyn_loss(delta_hat, x[:, 1:], x[:, :-1], dyn_mask)

        total = (
            weights.w_p * l_pol
            + weights.w_v * l_val
            + weights.w_m * l_mlh
            + weights.w_r(step, total_steps) * l_rec
            + weights.w_d * l_dyn
        )
        metrics: dict[str, float] = {
            "loss_total": float(total.detach()),
            "loss_policy": float(l_pol.detach()),
            "loss_value": float(l_val.detach()),
            "loss_mlh": float(l_mlh.detach()),
            "loss_recon": float(l_rec.detach()),
            "loss_dyn": float(l_dyn.detach()),
            **diag_rec,
            **diag_dyn,
        }
        return total, metrics

    # ---- 递推口径（推理 / 未来 MCTS 节点扩展） ----

    def initial_cache(self, batch_size: int, device, dtype=torch.float32) -> Cache:
        return self.r.initial_cache(batch_size, device=device, dtype=dtype)

    def step(
        self,
        features_t: torch.Tensor,      # (B, 785)
        tc_bucket: torch.Tensor,       # (B,)
        elo_std: torch.Tensor,         # (B,)
        color: torch.Tensor,           # (B,)
        cache: Cache,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Cache]:
        """单节点递推：返回 (policy_logits, wdl_logits, mlh, x_t, 新 cache)。"""
        x = self.encode(features_t).unsqueeze(1)                          # (B, 1, 512)
        cond = self.cond(tc_bucket, elo_std, color).unsqueeze(1)          # (B, 1, 512)
        h, cache_new = self.r.step(self.in_norm(x + cond), cache)
        policy_logits, wdl_logits, mlh = self.f(h)
        return policy_logits.squeeze(1), wdl_logits.squeeze(1), mlh.squeeze(1), x.squeeze(1), cache_new


def count_parameters(model: nn.Module) -> dict[str, int]:
    """分模块参数统计（对照 §5.6 预算表 ~27M）。"""
    names = {"cond": "cond", "e.": "E", "r.": "R", "f.": "f", "d.": "D", "g.": "g+predictor"}
    counts: dict[str, int] = {v: 0 for v in names.values()}
    other = 0
    for name, p in model.named_parameters():
        for prefix, label in names.items():
            if name.startswith(prefix):
                counts[label] += p.numel()
                break
        else:
            other += p.numel()
    counts["其他(含 in_norm)"] = other
    counts["合计"] = sum(counts.values())
    return counts
