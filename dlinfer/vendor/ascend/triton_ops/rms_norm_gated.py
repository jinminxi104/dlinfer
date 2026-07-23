# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Tri Dao.
#
# RMSNorm with gated SiLU activation (Triton kernel for Ascend NPU).
# Phase 1 replacement: 2D-tiled _layer_norm_fwd_1pass_kernel_npu replacing the
# loop-based _rms_norm_fwd_kernel, ported from
#   https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/layernorm_gated.py
# mypy: ignore-errors

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit(do_not_specialize=["stride_x_row", "stride_y_row", "stride_z_row", "M", "N", "eps"])
def _layer_norm_fwd_1pass_kernel_npu(
    X,           # pointer to input  (M, N)
    Y,           # pointer to output (M, N)
    W,           # pointer to weight (N,)
    B,           # pointer to bias   (N,) or None
    Z,           # pointer to gate   (M, N) or None
    Mean,        # pointer to mean   (ngroups*M,) or None when IS_RMS_NORM
    Rstd,        # pointer to rstd   (ngroups*M,)
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    group = tl.program_id(1)

    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N

    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=col_mask).to(tl.float32)

    x_ptrs = X + rows[:, None] * stride_x_row + cols[None, :] + group * N
    x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)

    if HAS_Z:
        z_ptrs = Z + rows[:, None] * stride_z_row + cols[None, :] + group * N
        z = tl.load(z_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
        if not NORM_BEFORE_GATE:
            x *= z * tl.sigmoid(z)

    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N
        xbar = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
        tl.store(Mean + rows, mean, mask=row_mask)
    else:
        xbar = tl.where(col_mask[None, :], x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N

    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd + rows, rstd, mask=row_mask)

    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]

    y = x_hat * w[None, :]
    if HAS_BIAS:
        y += b[None, :]
    if HAS_Z and NORM_BEFORE_GATE:
        y *= z * tl.sigmoid(z)

    y_ptrs = Y + rows[:, None] * stride_y_row + cols[None, :] + group * N
    tl.store(y_ptrs, y, mask=row_mask[:, None] & col_mask[None, :])


def layer_norm_fwd_npu(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    """RMSNorm (or LayerNorm) with optional SiLU gate, 2D-tiled Triton kernel.

    Returns (out, mean, rstd). mean is None when is_rms_norm=True.
    """
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z is not None:
        z = z.reshape(-1, z.shape[-1])
        if z.stride(-1) != 1:
            z = z.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size

    if out is None:
        out = torch.empty_like(x)
    mean = (
        None
        if is_rms_norm
        else torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    BLOCK_M = 64
    grid = (triton.cdiv(M, BLOCK_M), ngroups)

    with torch.npu.device(x.device.index):
        _layer_norm_fwd_1pass_kernel_npu[grid](
            x,
            out,
            weight,
            bias,
            z,
            mean,
            rstd,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            num_warps=num_warps,
        )
    return out.reshape(x_shape_og), mean, rstd


class RMSNormGated(nn.Module):

    def __init__(
        self,
        hidden_size,
        eps: float = 1e-5,
        group_size: Optional[int] = None,
        norm_before_gate: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.empty(hidden_size, device=device, dtype=torch.bfloat16)
        )
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, z=None):
        out, _, _ = layer_norm_fwd_npu(
            x,
            self.weight,
            self.bias,
            self.eps,
            z=z,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            is_rms_norm=True,
        )
        return out
