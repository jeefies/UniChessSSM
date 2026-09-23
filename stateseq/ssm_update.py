# 改编自 mamba_ssm 2.3.2 的 mamba_ssm/ops/triton/selective_state_update.py
# （Copyright (c) 2024, Tri Dao, Albert Gu；Apache-2.0）。
"""带「源槽 / 目标槽」的 selective_state_update（P5）。

原内核按 ``state_batch_indices`` 就地读写同一个槽；快速前向为了「父状态 → 子状态」须先把父槽
整块复制到子槽再就地更新，每行多搬 2×12 层 × 64 KiB（实测占 64 行块 GPU 时间的 ~20%）。
这里只把**读地址**换成源槽、**写地址**换成目标槽，其余逐行照搬原内核：同样的装载、同样的算式
与次序、同样的块大小和 warp 数——所以结果与「先复制再就地更新」逐位相同
（``tests/test_fast_eval.py::test_ssm_update_src_dst_bitwise`` 对照）。

只保留快速前向用到的形态：state (S, nheads, dim, dstate)、x/dt (b, nheads, dim)、B/C
(b, ngroups, dstate)、D/dt_bias (nheads, dim)，无 z。源、目标槽不得与同批其他行的槽重叠
（子槽都是新分配的，父槽已算完）。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from mamba_ssm.ops.triton.softplus import softplus


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _ssm_update_src_dst_kernel(
    # Pointers to matrices
    state_ptr, x_ptr, dt_ptr, dt_bias_ptr, A_ptr, B_ptr, C_ptr, D_ptr, out_ptr,
    src_idx_ptr, dst_idx_ptr,
    # Matrix dimensions
    batch, nheads, dim, dstate, nheads_ngroups_ratio,
    # Strides
    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
    stride_x_batch, stride_x_head, stride_x_dim,
    stride_dt_batch, stride_dt_head, stride_dt_dim,
    stride_dt_bias_head, stride_dt_bias_dim,
    stride_A_head, stride_A_dim, stride_A_dstate,
    stride_B_batch, stride_B_group, stride_B_dstate,
    stride_C_batch, stride_C_group, stride_C_dstate,
    stride_D_head, stride_D_dim,
    stride_out_batch, stride_out_head, stride_out_dim,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    out_ptrs = out_ptr + offs_m * stride_out_dim

    src_idx = tl.load(src_idx_ptr + pid_b)
    dst_idx = tl.load(dst_idx_ptr + pid_b)
    src_ptr = state_ptr + src_idx * stride_state_batch + pid_h * stride_state_head
    dst_ptr = state_ptr + dst_idx * stride_state_batch + pid_h * stride_state_head

    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group

    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    tile = offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    src_ptrs = src_ptr + tile
    dst_ptrs = dst_ptr + tile
    x_ptrs = x_ptr + offs_m * stride_x_dim
    dt_ptrs = dt_ptr + offs_m * stride_dt_dim
    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
    A_ptrs = A_ptr + (offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate)
    B_ptrs = B_ptr + offs_n * stride_B_dstate
    C_ptrs = C_ptr + offs_n * stride_C_dstate
    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim

    state = tl.load(src_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)
    x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if not TIE_HDIM:
        dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if DT_SOFTPLUS:
            dt = tl.where(dt <= 20.0, softplus(dt), dt)
        A = tl.load(A_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        dA = tl.exp(A * dt[:, None])
    else:
        dt = tl.load(dt_ptr).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptr).to(tl.float32)
        if DT_SOFTPLUS:
            dt = tl.where(dt <= 20.0, softplus(dt), dt)
        A = tl.load(A_ptr).to(tl.float32)
        dA = tl.exp(A * dt)  # scalar, not a matrix

    B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    if HAS_D:
        D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

    if not TIE_HDIM:
        dB = B[None, :] * dt[:, None]
    else:
        dB = B * dt  # vector of size (dstate,)
    state = state * dA + dB * x[:, None]
    tl.store(dst_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))
    out = tl.sum(state * C[None, :], axis=1)
    if HAS_D:
        out += x * D
    tl.store(out_ptrs, out, mask=offs_m < dim)


def ssm_update_src_dst(state, x, dt, A, B, C, D, dt_bias, dt_softplus: bool,
                       src_idx: torch.Tensor, dst_idx: torch.Tensor) -> torch.Tensor:
    """state[dst] = 更新(state[src])，返回 out (b, nheads, dim)。参数形态与原
    ``selective_state_update(..., z=None, state_batch_indices=...)`` 相同（heads 形态）。"""
    _, nheads, dim, dstate = state.shape
    batch = x.shape[0]
    assert x.shape == (batch, nheads, dim) and dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0 and B.shape == (batch, ngroups, dstate) and C.shape == B.shape
    assert D.shape == (nheads, dim) and dt_bias.shape == (nheads, dim)
    assert src_idx.shape == (batch,) and dst_idx.shape == (batch,)
    assert src_idx.dtype == dst_idx.dtype == torch.int32
    out = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE_M']), batch, nheads)  # noqa: E731
    # 与原内核相同的手调块大小 / warp 数（不能 autotune：会改写状态）
    BLOCK_SIZE_M, num_warps = ((32, 4) if dstate <= 16
                               else ((16, 4) if dstate <= 32 else
                                     ((8, 4) if dstate <= 64 else
                                      ((4, 4) if dstate <= 128 else
                                       ((4, 8))))))
    tie_hdim = A.stride(-1) == 0 and A.stride(-2) == 0 and dt.stride(-1) == 0 and dt_bias.stride(-1) == 0
    with torch.cuda.device(x.device.index):
        _ssm_update_src_dst_kernel[grid](
            state, x, dt, dt_bias, A, B, C, D, out, src_idx, dst_idx,
            batch, nheads, dim, dstate, nheads // ngroups,
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            x.stride(0), x.stride(1), x.stride(2),
            dt.stride(0), dt.stride(1), dt.stride(2),
            dt_bias.stride(0), dt_bias.stride(1),
            A.stride(0), A.stride(1), A.stride(2),
            B.stride(0), B.stride(1), B.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            D.stride(0), D.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            dt_softplus,
            tie_hdim,
            BLOCK_SIZE_M,
            num_warps=num_warps,
        )
    return out
