# **************************************************
# Copyright (c) 2025, Mayank Mishra
#
# Vendored from open-lm-engine/accelerated-model-architectures:
#   xma/layers/m2rnn/triton_implementation/backward.py
#
# Edits for state_tracking baseline:
#   - Removed the `@xma_op(mutates_args=...)` decorator. The kernel is
#     called directly from our own `torch.autograd.Function` wrapper in
#     `state_tracking/models/rnn/op.py`.
#   - Rewrote upstream relative imports (`....custom_op`, `....math`,
#     `....triton_utils`, `..utils`) to point at the local vendored
#     copies / the parent `op.py`.
#   - Dropped the variable-length (`cu_seqlens`) path: the state-tracking
#     task uses fixed-length sequences, so `IS_VARLEN` is removed and
#     `S_DIM/N_DIM/K_DIM` collapse to 1/2/3.
#   - Replaced tuple `*_stride` kernel arguments with individual `int`
#     scalars (Triton >= 3.2 accepts tuples natively; we target 3.1).
# **************************************************

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..op import _get_num_heads
from .forward import _forward_single_step, _get_autotune_configs
from .kernels import clamp, matmul, sigmoid_backward, tanh_backward  # noqa: F401
from .math_utils import ceil_divide, get_next_power_of_2


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["BLOCK_SIZE_K", "BLOCK_SIZE_V"],
    reset_to_zero=["dxf_ptr", "dq_ptr", "dk_ptr", "dv_ptr", "dW_ptr"],
)
@triton.jit
def _m2rnn_backward_triton_kernel(
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
    h_ptr,
    h_stride_b,
    h_stride_s,
    h_stride_n,
    h_stride_k,
    h_stride_v,
    xf_ptr,
    xf_stride_b,
    xf_stride_s,
    xf_stride_n,
    dxf_ptr,
    dxf_stride_b,
    dxf_stride_s,
    dxf_stride_n,
    h0_ptr,
    h0_stride_b,
    h0_stride_n,
    h0_stride_k,
    h0_stride_v,
    dy_ptr,
    dy_stride_b,
    dy_stride_s,
    dy_stride_n,
    dy_stride_v,
    dq_ptr,
    dq_stride_b,
    dq_stride_s,
    dq_stride_n,
    dq_stride_k,
    dk_ptr,
    dk_stride_b,
    dk_stride_s,
    dk_stride_n,
    dk_stride_k,
    dv_ptr,
    dv_stride_b,
    dv_stride_s,
    dv_stride_n,
    dv_stride_v,
    dW_ptr,
    dW_stride_n,
    dW_stride_r,
    dW_stride_c,
    dh0_ptr,
    dh0_stride_b,
    dh0_stride_n,
    dh0_stride_k,
    dh0_stride_v,
    gradient_clipping,
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

    dh = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_V), dtype=W_ptr.dtype.element_ty)
    dW = tl.zeros((BLOCK_SIZE_V, BLOCK_SIZE_V), dtype=tl.float32)

    _S = S - 1

    q_ptrs = q_ptr + BLOCK_ID_B * q_stride_b + _S * q_stride_s + BLOCK_ID_Nq * q_stride_n + BLOCK_K * q_stride_k
    k_ptrs = k_ptr + BLOCK_ID_B * k_stride_b + _S * k_stride_s + BLOCK_ID_Nk * k_stride_n + BLOCK_K * k_stride_k
    v_ptrs = v_ptr + BLOCK_ID_B * v_stride_b + _S * v_stride_s + BLOCK_ID_Nv * v_stride_n + BLOCK_V * v_stride_v

    dy_ptrs = (
        dy_ptr + BLOCK_ID_B * dy_stride_b + _S * dy_stride_s + BLOCK_ID_N * dy_stride_n + BLOCK_V * dy_stride_v
    )

    dq_ptrs = (
        dq_ptr + BLOCK_ID_B * dq_stride_b + _S * dq_stride_s + BLOCK_ID_Nq * dq_stride_n + BLOCK_K * dq_stride_k
    )
    dk_ptrs = (
        dk_ptr + BLOCK_ID_B * dk_stride_b + _S * dk_stride_s + BLOCK_ID_Nk * dk_stride_n + BLOCK_K * dk_stride_k
    )
    dv_ptrs = (
        dv_ptr + BLOCK_ID_B * dv_stride_b + _S * dv_stride_s + BLOCK_ID_Nv * dv_stride_n + BLOCK_V * dv_stride_v
    )

    xf_ptrs = xf_ptr + BLOCK_ID_B * xf_stride_b + _S * xf_stride_s + BLOCK_ID_Nxf * xf_stride_n

    # NOTE: h is [B, S, N, K, V] - at large (B, S) the batch offset
    # `BLOCK_ID_B * h_stride_b` can exceed signed i32 range (see the
    # forward kernel comment). Cast to int64 before multiplying so the
    # whole sum is evaluated in 64 bits.
    h_ptrs = (
        h_ptr
        + tl.cast(BLOCK_ID_B, tl.int64) * h_stride_b
        + _S * h_stride_s
        + BLOCK_ID_N * h_stride_n
        + BLOCK_K[:, None] * h_stride_k
        + BLOCK_V[None, :] * h_stride_v
    )

    dxf_ptrs = dxf_ptr + BLOCK_ID_B * dxf_stride_b + _S * dxf_stride_s + BLOCK_ID_Nxf * dxf_stride_n

    for s in range(S - 1, -1, -1):
        if s == 0:
            if h0_ptr is None:
                h_prev = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_V), dtype=W.dtype)
            else:
                h_prev = tl.load(
                    h0_ptr
                    + BLOCK_ID_B * h0_stride_b
                    + BLOCK_ID_N * h0_stride_n
                    + BLOCK_K[:, None] * h0_stride_k
                    + BLOCK_V[None, :] * h0_stride_v,
                    mask=MASK_KV,
                )
        else:
            h_ptrs -= h_stride_s
            h_prev = tl.load(h_ptrs, mask=MASK_KV)

        q = tl.load(q_ptrs, mask=MASK_K)
        q_ptrs -= q_stride_s

        k = tl.load(k_ptrs, mask=MASK_K)
        k_ptrs -= k_stride_s

        v = tl.load(v_ptrs, mask=MASK_V)
        v_ptrs -= v_stride_s

        f = tl.load(xf_ptrs)
        xf_ptrs -= xf_stride_s

        z, h = _forward_single_step(h_prev=h_prev, W=W, k=k, v=v, f=f)

        dy = tl.load(dy_ptrs, mask=MASK_V)
        dy_ptrs -= dy_stride_s

        dq = matmul(A=h, B=dy[:, None], C=None, output_dtype=q.dtype)

        if Gq == 1:
            tl.store(dq_ptrs[:, None], dq, mask=MASK_K[:, None])
        else:
            tl.atomic_add(dq_ptrs[:, None], dq, mask=MASK_K[:, None], sem="relaxed")

        dq_ptrs -= dq_stride_s

        if gradient_clipping is not None:
            dh = clamp(dh, min_value=-gradient_clipping, max_value=gradient_clipping)

        dyh = matmul(A=q[:, None], B=dy[None, :], C=dh, output_dtype=q.dtype)

        df = dyh * (h_prev - z)
        df = tl.sum(df)

        if Gxf == 1:
            tl.store(dxf_ptrs, df)
        else:
            tl.atomic_add(dxf_ptrs, df, sem="relaxed")

        dxf_ptrs -= dxf_stride_s

        # NOTE: `(1 - f)` with an `int` literal promotes to fp32 under
        # triton 3.1; cast the result back so matmul dtypes agree below.
        dh = f * dyh
        dz = (dyh * (1 - f)).to(dyh.dtype)

        dx = (dz * tanh_backward(z)).to(k.dtype)
        dh = matmul(A=dx, B=W.T, C=dh, output_dtype=dx.dtype)
        dW = matmul(A=h_prev.T, B=dx, C=dW, output_dtype=dW.dtype)

        dv = matmul(A=dx.T, B=k[:, None], C=None, output_dtype=k.dtype)

        if Gv == 1:
            tl.store(dv_ptrs[:, None], dv, mask=MASK_V[:, None])
        else:
            tl.atomic_add(dv_ptrs[:, None], dv, mask=MASK_V[:, None], sem="relaxed")

        dv_ptrs -= dv_stride_s
        dk = matmul(A=dx, B=v[:, None], C=None, output_dtype=k.dtype)

        if Gk == 1:
            tl.store(dk_ptrs[:, None], dk, mask=MASK_K[:, None])
        else:
            tl.atomic_add(dk_ptrs[:, None], dk, mask=MASK_K[:, None], sem="relaxed")

        dk_ptrs -= dk_stride_s

    if dh0_ptr is not None:
        tl.store(
            dh0_ptr
            + BLOCK_ID_B * dh0_stride_b
            + BLOCK_ID_N * dh0_stride_n
            + BLOCK_K[:, None] * dh0_stride_k
            + BLOCK_V[None, :] * dh0_stride_v,
            dh,
            mask=MASK_KV,
        )

    tl.atomic_add(
        dW_ptr + BLOCK_ID_Nw * dW_stride_n + BLOCK_V[:, None] * dW_stride_r + BLOCK_V[None, :] * dW_stride_c,
        dW,
        mask=MASK_VV,
        sem="relaxed",
    )


def _zero_strides(ndim: int) -> tuple[int, ...]:
    return (0,) * ndim


def _m2rnn_backward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    h0: torch.Tensor | None,
    dy: torch.Tensor,
    h: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dW: torch.Tensor,
    dxf: torch.Tensor,
    dh0: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,  # kept for signature parity; must be None
    gradient_clipping: float | None,
) -> None:
    assert cu_seqlens is None, "varlen (cu_seqlens) path is not vendored"

    Nq, Nk, Nv, Nw, Nxf, N = _get_num_heads(q=q, k=k, v=v, W=W, xf=xf, run_check=False)

    B, S, _, K, V = h.size()

    BLOCK_SIZE_K = get_next_power_of_2(K)
    BLOCK_SIZE_K = max(16, BLOCK_SIZE_K)
    BLOCK_SIZE_K = min(64, BLOCK_SIZE_K)

    BLOCK_SIZE_V = get_next_power_of_2(V)
    BLOCK_SIZE_V = max(16, BLOCK_SIZE_V)

    q_b, q_s, q_n, q_k = q.stride()
    k_b, k_s, k_n, k_k = k.stride()
    v_b, v_s, v_n, v_v = v.stride()
    W_n, W_r, W_c = W.stride()
    h_b, h_s, h_n, h_k_, h_v = h.stride()
    xf_b, xf_s, xf_n = xf.stride()
    dxf_b, dxf_s, dxf_n = dxf.stride()
    h0_b, h0_n, h0_k, h0_v = (h0.stride() if h0 is not None else _zero_strides(4))
    dy_b, dy_s, dy_n, dy_v = dy.stride()
    dq_b, dq_s, dq_n, dq_k = dq.stride()
    dk_b, dk_s, dk_n, dk_k = dk.stride()
    dv_b, dv_s, dv_n, dv_v = dv.stride()
    dW_n, dW_r, dW_c = dW.stride()
    dh0_b, dh0_n, dh0_k, dh0_v = (dh0.stride() if dh0 is not None else _zero_strides(4))

    # NOTE: passed positionally so that the autotuner's `reset_to_zero`
    # hook (which indexes `args` by position) can locate dq/dk/dv/dxf/dW
    # during tuning. Must match the kernel parameter order.
    _m2rnn_backward_triton_kernel[B, N, ceil_divide(K, BLOCK_SIZE_K)](
        q, q_b, q_s, q_n, q_k,
        k, k_b, k_s, k_n, k_k,
        v, v_b, v_s, v_n, v_v,
        W, W_n, W_r, W_c,
        h, h_b, h_s, h_n, h_k_, h_v,
        xf, xf_b, xf_s, xf_n,
        dxf, dxf_b, dxf_s, dxf_n,
        h0, h0_b, h0_n, h0_k, h0_v,
        dy, dy_b, dy_s, dy_n, dy_v,
        dq, dq_b, dq_s, dq_n, dq_k,
        dk, dk_b, dk_s, dk_n, dk_k,
        dv, dv_b, dv_s, dv_n, dv_v,
        dW, dW_n, dW_r, dW_c,
        dh0, dh0_b, dh0_n, dh0_k, dh0_v,
        gradient_clipping,
        S,
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
