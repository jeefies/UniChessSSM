"""R：基础 Mamba 主干（设计文档 §5.3），12 层 Mamba-2。

每块配置：d_model=512, expand=2（d_inner=1024）, d_state=16, d_conv=4, headdim=64；
块级：h ← h + MambaBlock(RMSNorm(h))；scan 部分保持 fp32（官方 kernel 默认）。

序列口径：整序列并行扫描前向（Stage A 全序列训练，不用 TBPTT）。
递推口径（推理/MCTS）：逐节点单步，节点状态 = 完整逐层 cache（conv_state + ssm_state）。
注意：官方 Mamba2.step 会原地改写传入 cache —— 分支隔离必须由调用方克隆（copy-on-write，§9）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None  # type: ignore

from .layers import RMSNorm

D_MODEL = 512
N_LAYERS = 12
D_STATE = 16
D_CONV = 4
EXPAND = 2
HEADDIM = 64

Cache = list[tuple[torch.Tensor, torch.Tensor]]  # 每层 (conv_state, ssm_state)


class MambaTower(nn.Module):
    def __init__(
        self,
        d_model: int = D_MODEL,
        n_layers: int = N_LAYERS,
        d_state: int = D_STATE,
        d_conv: int = D_CONV,
        expand: int = EXPAND,
        headdim: int = HEADDIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        if Mamba2 is None:
            raise RuntimeError("mamba_ssm 未安装或在当前环境不可用（需要 CUDA 及 mamba_ssm）")
        self.d_model = d_model
        self.n_layers = n_layers
        self.blocks = nn.ModuleList(
            Mamba2(
                d_model=d_model,
                d_state=d_state,
                headdim=headdim,
                expand=expand,
                d_conv=d_conv,
            )
            for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(RMSNorm(d_model) for _ in range(n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """整序列前向：x (B, T, d) → h (B, T, d)（Mamba 并行扫描，训练用）。"""
        for block, norm in zip(self.blocks, self.norms):
            x = x + block(norm(x))
        return x

    # ---- 递推口径（推理 / MCTS 节点扩展）----

    def initial_cache(self, batch_size: int, device, dtype=torch.float32) -> Cache:
        """初始隐状态 cache：每层 (conv_state, ssm_state)，张量独立于后续 step。"""
        cache: Cache = []
        for block in self.blocks:
            conv, ssm = block.allocate_inference_cache(batch_size, max_seqlen=1, dtype=dtype)
            cache.append((conv.to(device), ssm.to(device)))
        return cache

    def step(self, x_t: torch.Tensor, cache: Cache) -> tuple[torch.Tensor, Cache]:
        """单步递推：x_t (B, 1, d) + cache → h_t (B, 1, d), 新 cache。

        官方 step 原地改写传入 cache；本方法内先克隆再调用，返回的新 cache 不与入参共享存储，
        天然满足 copy-on-write（子节点复制父 cache 后互不污染，验收 #4 / §9 分支隔离）。
        """
        new_cache: Cache = []
        h = x_t
        for block, norm, (conv, ssm) in zip(self.blocks, self.norms, cache):
            y, conv_new, ssm_new = block.step(norm(h), conv.clone(), ssm.clone())
            h = h + y
            new_cache.append((conv_new, ssm_new))
        return h, new_cache


def clone_cache(cache: Cache) -> Cache:
    """显式深克隆一份 cache（MCTS 子节点分支时调用）。"""
    return [(conv.clone(), ssm.clone()) for conv, ssm in cache]
