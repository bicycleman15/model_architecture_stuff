# **************************************************
# Copyright (c) 2025, Mayank Mishra
#
# Vendored from open-lm-engine/accelerated-model-architectures:
#   xma/layers/m2rnn/triton_implementation/forward.py
#
# Edits for state_tracking baseline:
#   - Removed the `@xma_op(mutates_args=...)` decorator. The kernel is
#     called directly from our own `torch.autograd.Function` wrapper in
#     `state_tracking/models/rnn/op.py`, so no torch-operator registration
#     is needed.
#   - Rewrote upstream relative imports (`....custom_op`, `....math`,
#     `....triton_utils`) to point at the local vendored copies under
#     `state_tracking/models/rnn/triton/`.
#   - Dropped the variable-length (`cu_seqlens`) batching path: the
#     state-tracking task trains on fixed-length sequences only. This
#     removes the `IS_VARLEN` branch and collapses `S_DIM/N_DIM/K_DIM`
#     to 1/2/3 throughout.
#   - Replaced tuple `*_stride` kernel arguments with individual `int`
#     scalars (e.g. `q_stride_b, q_stride_s, q_stride_n, q_stride_k`).
#     Triton >= 3.2 accepts tuple args natively, but we target 3.1.
# **************************************************

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .kernels import matmul, tanh
from .math_utils import ceil_divide, get_next_power_of_2, get_powers_of_2


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for num_warps in get_powers_of_2(1, 32):
        for num_stages in range(1, 5):
            configs.append(triton.Config({}, num_stages=num_stages, num_warps=num_warps))

    return configs


@triton.jit
def _forward_single_step(h_prev, W, k, v, f):
    x = matmul(A=k[:, None], B=v[None, :], C=None, output_dtype=k.dtype)
    z = matmul(A=h_prev, B=W, C=x, output_dtype=tl.float32)
    z = tanh(z, output_dtype=x.dtype)

    # NOTE: under triton 3.1, `(1 - f) * z` with an `int` literal can
    # promote the result to fp32 on the first iteration, breaking the
    # loop-carried type of `h`. Cast explicitly back to `h_prev.dtype`
    # to keep the kernel compiling (upstream triton >= 3.2 infers this).
    h = f * h_prev + (1 - f) * z
    h = h.to(h_prev.dtype)

    return z, h


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_K", "BLOCK_SIZE_V"])
@triton.jit
def _m2rnn_forward_triton_kernel(
    q_ptr,
    q_stride_b,
    q_stride_s,
    q_stride_n,
    q_stride_k,
    k_ptr,
    k_stride_b,
    k_stride_s,
    k_stride_n,
    k_stride_k,
    v_ptr,
    v_stride_b,
    v_stride_s,
    v_stride_n,
    v_stride_v,
    W_ptr,
    W_stride_n,
    W_stride_r,
    W_stride_c,
    xf_ptr,
    xf_stride_b,
    xf_stride_s,
    xf_stride_n,
    h0_ptr,
    h0_stride_b,
    h0_stride_n,
    h0_stride_k,
    h0_stride_v,
    h_ptr,
    h_stride_b,
    h_stride_s,
    h_stride_n,
    h_stride_k,
    h_stride_v,
    ht_ptr,
    ht_stride_b,
    ht_stride_n,
    ht_stride_k,
    ht_stride_v,
    y_ptr,
    y_stride_b,
    y_stride_s,
    y_stride_n,
    y_stride_v,
    S,
    K: tl.constexpr,
    V: tl.constexpr,
    Gq: tl.constexpr,
    Gk: tl.constexpr,
    Gv: tl.constexpr,
    Gw: tl.constexpr,
    Gxf: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(0)
    BLOCK_ID_N = tl.program_id(1)
    BLOCK_ID_K = tl.program_id(2)

    BLOCK_ID_Nq = BLOCK_ID_N // Gq
    BLOCK_ID_Nk = BLOCK_ID_N // Gk
    BLOCK_ID_Nv = BLOCK_ID_N // Gv

    BLOCK_ID_Nw = BLOCK_ID_N // Gw
    BLOCK_ID_Nxf = BLOCK_ID_N // Gxf

    BLOCK_K = BLOCK_ID_K * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    BLOCK_V = tl.arange(0, BLOCK_SIZE_V)

    MASK_K = BLOCK_K < K
    MASK_V = BLOCK_V < V

    MASK_KV = MASK_K[:, None] & MASK_V[None, :]
    MASK_VV = MASK_V[:, None] & MASK_V[None, :]

    W = tl.load(
        W_ptr + BLOCK_ID_Nw * W_stride_n + BLOCK_V[:, None] * W_stride_r + BLOCK_V[None, :] * W_stride_c,
        mask=MASK_VV,
    )

    if h0_ptr is None:
        h = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_V), dtype=k_ptr.dtype.element_ty)
    else:
        h = tl.load(
            h0_ptr
            + BLOCK_ID_B * h0_stride_b
            + BLOCK_ID_N * h0_stride_n
            + BLOCK_K[:, None] * h0_stride_k
            + BLOCK_V[None, :] * h0_stride_v,
            mask=MASK_KV,
        )

    k_ptrs = k_ptr + BLOCK_ID_B * k_stride_b + BLOCK_ID_Nk * k_stride_n + BLOCK_K * k_stride_k
    v_ptrs = v_ptr + BLOCK_ID_B * v_stride_b + BLOCK_ID_Nv * v_stride_n + BLOCK_V * v_stride_v
    xf_ptrs = xf_ptr + BLOCK_ID_B * xf_stride_b + BLOCK_ID_Nxf * xf_stride_n

    if h_ptr is not None:
        # NOTE: h is a [B, S, N, K, V] state tape. For eval (S=512, B=512,
        # K=V=32, N=12) the batch stride is 6.29M elements and
        # BLOCK_ID_B * h_stride_b reaches ~3.2B, which overflows signed
        # i32. Cast one operand to int64 so the full offset is computed
        # in 64 bits. (Upstream's `tl.cast(..., tl.uint32)` after the
        # product doesn't prevent the overflow that happens *inside*
        # the product.)
        h_ptrs = (
            h_ptr
            + tl.cast(BLOCK_ID_B, tl.int64) * h_stride_b
            + BLOCK_ID_N * h_stride_n
            + BLOCK_K[:, None] * h_stride_k
            + BLOCK_V[None, :] * h_stride_v
        )

    if q_ptr is not None:
        q_ptrs = q_ptr + BLOCK_ID_B * q_stride_b + BLOCK_ID_Nq * q_stride_n + BLOCK_K * q_stride_k

    if y_ptr is not None:
        y_ptrs = y_ptr + BLOCK_ID_B * y_stride_b + BLOCK_ID_N * y_stride_n + BLOCK_V * y_stride_v

    for s in range(1, S + 1):
        k = tl.load(k_ptrs, mask=MASK_K)
        k_ptrs += k_stride_s

        v = tl.load(v_ptrs, mask=MASK_V)
        v_ptrs += v_stride_s

        f = tl.load(xf_ptrs)
        xf_ptrs += xf_stride_s

        _, h = _forward_single_step(h_prev=h, W=W, k=k, v=v, f=f)

        if h_ptr is not None:
            tl.store(h_ptrs, h, mask=MASK_KV)
            h_ptrs += h_stride_s

        if y_ptr is not None:
            q = tl.load(q_ptrs, mask=MASK_K)
            q_ptrs += q_stride_s

            y = matmul(A=q[None, :], B=h, C=None, output_dtype=q.dtype)

            tl.store(y_ptrs[None, :], y, mask=MASK_V[None, :])
            y_ptrs += y_stride_s

    if ht_ptr is not None:
        tl.store(
            ht_ptr
            + BLOCK_ID_B * ht_stride_b
            + BLOCK_ID_N * ht_stride_n
            + BLOCK_K[:, None] * ht_stride_k
            + BLOCK_V[None, :] * ht_stride_v,
            h,
            mask=MASK_KV,
        )


def _zero_strides(ndim: int) -> tuple[int, ...]:
    return (0,) * ndim


def _m2rnn_forward_triton(
    q: torch.Tensor | None,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    h0: torch.Tensor | None,
    h: torch.Tensor | None,
    ht: torch.Tensor | None,
    y: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,  # kept for signature parity; must be None
    Nq: int,
    Nk: int,
    Nv: int,
    Nw: int,
    Nxf: int,
    N: int,
) -> None:
    assert cu_seqlens is None, "varlen (cu_seqlens) path is not vendored"

    B, S, _, K = k.size()
    V = v.size(-1)

    BLOCK_SIZE_K = get_next_power_of_2(K)
    BLOCK_SIZE_K = max(16, BLOCK_SIZE_K)
    BLOCK_SIZE_K = min(64, BLOCK_SIZE_K)

    BLOCK_SIZE_V = get_next_power_of_2(V)
    BLOCK_SIZE_V = max(16, BLOCK_SIZE_V)

    q_b, q_s, q_n, q_k = (q.stride() if q is not None else _zero_strides(4))
    k_b, k_s, k_n, k_k = k.stride()
    v_b, v_s, v_n, v_v = v.stride()
    W_n, W_r, W_c = W.stride()
    xf_b, xf_s, xf_n = xf.stride()
    h0_b, h0_n, h0_k, h0_v = (h0.stride() if h0 is not None else _zero_strides(4))
    h_b, h_s, h_n, h_k, h_v = (h.stride() if h is not None else _zero_strides(5))
    ht_b, ht_n, ht_k, ht_v = (ht.stride() if ht is not None else _zero_strides(4))
    y_b, y_s, y_n, y_v = (y.stride() if y is not None else _zero_strides(4))

    _m2rnn_forward_triton_kernel[B, N, ceil_divide(K, BLOCK_SIZE_K)](
        q_ptr=q,
        q_stride_b=q_b, q_stride_s=q_s, q_stride_n=q_n, q_stride_k=q_k,
        k_ptr=k,
        k_stride_b=k_b, k_stride_s=k_s, k_stride_n=k_n, k_stride_k=k_k,
        v_ptr=v,
        v_stride_b=v_b, v_stride_s=v_s, v_stride_n=v_n, v_stride_v=v_v,
        W_ptr=W,
        W_stride_n=W_n, W_stride_r=W_r, W_stride_c=W_c,
        xf_ptr=xf,
        xf_stride_b=xf_b, xf_stride_s=xf_s, xf_stride_n=xf_n,
        h0_ptr=h0,
        h0_stride_b=h0_b, h0_stride_n=h0_n, h0_stride_k=h0_k, h0_stride_v=h0_v,
        h_ptr=h,
        h_stride_b=h_b, h_stride_s=h_s, h_stride_n=h_n, h_stride_k=h_k, h_stride_v=h_v,
        ht_ptr=ht,
        ht_stride_b=ht_b, ht_stride_n=ht_n, ht_stride_k=ht_k, ht_stride_v=ht_v,
        y_ptr=y,
        y_stride_b=y_b, y_stride_s=y_s, y_stride_n=y_n, y_stride_v=y_v,
        S=S,
        K=K,
        V=V,
        Gq=N // Nq,
        Gk=N // Nk,
        Gv=N // Nv,
        Gw=N // Nw,
        Gxf=N // Nxf,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )
