# **************************************************
# Copyright (c) 2025, Mayank Mishra
#
# Vendored from open-lm-engine/accelerated-model-architectures
#   xma/layers/m2rnn/op.py   (pure-torch path of `_M2RNN.forward_backward_torch`)
#   xma/layers/m2rnn/utils.py (`_get_num_heads`)
#
# Edits (for state_tracking baseline):
#   - triton / CUDA / CustomOp backends removed; only the pure-PyTorch
#     reference recurrence is kept. PyTorch autograd handles the backward
#     natively through the Python time loop.
#   - cu_seqlens / variable-length path removed (we train on fixed lengths).
#   - `tanh` is just torch.tanh and `clip_gradients` is a plain clamp
#     (upstream uses a straight-through-estimator variant; for fixed-length
#     S_3 training the difference is negligible).
# **************************************************

from __future__ import annotations

import torch


def divide_if_divisible(a: int, b: int) -> int:
    assert a % b == 0, f"{a} is not divisible by {b}"
    return a // b


def _get_num_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    run_check: bool,
) -> tuple[int, int, int, int, int, int]:
    Nq = q.size(-2)
    Nk = k.size(-2)
    Nv = v.size(-2)

    Nw = W.size(0)
    Nxf = xf.size(-1)

    N = max(Nq, Nk, Nv, Nw, Nxf)

    if run_check:
        assert N % Nq == 0
        assert N % Nk == 0
        assert N % Nv == 0
        assert N % Nw == 0
        assert N % Nxf == 0

    return Nq, Nk, Nv, Nw, Nxf, N


def _clip_gradients(x: torch.Tensor, c: float | None) -> torch.Tensor:
    if c is None:
        return x
    return x.clamp(-c, c)


def _m2rnn_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    forget_input: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch M2RNN recurrence.

    Shapes (with B=batch, S=seq_len, K=key_head_dim, V=value_head_dim,
    N=max(num_heads_over_q/k/v/W/xf))::

        query         : [B, S, Nq, K]
        key           : [B, S, Nk, K]
        value         : [B, S, Nv, V]
        weight        : [Nw, V, V]
        forget_input  : [B, S, Nxf]
        input_state   : [B, N, K, V] or None (-> zeros)

    Returns
    -------
    output : [B, S, N, V]
    h_final: [B, N, K, V]
    """
    Nq, Nk, Nv, Nw, Nxf, N = _get_num_heads(
        q=query, k=key, v=value, W=weight, xf=forget_input, run_check=True
    )

    B, S, _, K = query.size()
    V = value.size(-1)

    assert query.size() == (B, S, Nq, K)
    assert key.size() == (B, S, Nk, K)
    assert value.size() == (B, S, Nv, V)
    assert forget_input.size() == (B, S, Nxf)
    assert weight.size() == (Nw, V, V)
    if input_state is not None:
        assert input_state.size() == (B, N, K, V)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv
    Gw = N // Nw
    Gxf = N // Nxf

    # Broadcast grouped heads so each tensor has N heads (no-op when all
    # head counts already equal N).
    q = query.repeat_interleave(Gq, dim=-2)
    k = key.repeat_interleave(Gk, dim=-2)
    v = value.repeat_interleave(Gv, dim=-2)
    W = weight.repeat_interleave(Gw, dim=0)
    xf = forget_input.repeat_interleave(Gxf, dim=-1)

    if gradient_clipping is not None and gradient_clipping < 0:
        gradient_clipping = -gradient_clipping

    # Outer product per timestep: x[b,s,n,i,j] = k[b,s,n,i] * v[b,s,n,j]
    x = k.unsqueeze(-1) * v.unsqueeze(-2)  # [B, S, N, K, V]
    Wb = W.unsqueeze(0)                    # [1, N, V, V]

    if input_state is None:
        h = torch.zeros(B, N, K, V, device=k.device, dtype=k.dtype)
    else:
        h = input_state

    # Pre-allocate state tape.
    states = torch.empty(B, S, N, K, V, device=q.device, dtype=q.dtype)

    for s in range(S):
        f = xf[:, s, :, None, None]        # [B, N, 1, 1]
        h_new = torch.tanh(h @ Wb + x[:, s])
        h = f * h + (1.0 - f) * h_new
        h = _clip_gradients(h, gradient_clipping)
        states[:, s] = h

    # Project back to value-space with the query: q @ state[b,s,n]  gives
    # a single V-dim vector per (b,s,n).
    #   q: [B,S,N,K] -> [B,S,N,1,K]
    #   states: [B,S,N,K,V]
    #   out: [B,S,N,1,V] -> [B,S,N,V]
    y = (q.unsqueeze(-2) @ states).squeeze(-2)

    return y, h


class _M2RNNTritonFunction(torch.autograd.Function):
    """torch.autograd.Function wrapper around the vendored triton kernels.

    Lives in this module (not in `triton/`) so that we don't import triton
    at top level — the kernels are only loaded when the user picks the
    triton backend (and therefore must be on CUDA).
    """

    @staticmethod
    def forward(ctx, q, k, v, W, xf, h0, gradient_clipping):
        from state_tracking.models.rnn.triton.forward import _m2rnn_forward_triton

        Nq, Nk, Nv, Nw, Nxf, N = _get_num_heads(q=q, k=k, v=v, W=W, xf=xf, run_check=True)

        B, S, _, K = q.size()
        V = v.size(-1)

        ht = torch.empty(B, N, K, V, device=k.device, dtype=k.dtype)
        h = torch.empty(B, S, N, K, V, device=k.device, dtype=k.dtype)
        y = torch.empty(B, S, N, V, device=q.device, dtype=q.dtype)

        _m2rnn_forward_triton(
            q=q,
            k=k,
            v=v,
            W=W,
            xf=xf,
            h0=h0,
            h=h,
            ht=ht,
            y=y,
            cu_seqlens=None,
            Nq=Nq,
            Nk=Nk,
            Nv=Nv,
            Nw=Nw,
            Nxf=Nxf,
            N=N,
        )

        ctx.save_for_backward(q, k, v, W, xf, h0 if h0 is not None else torch.empty(0), h)
        ctx.has_h0 = h0 is not None
        ctx.gradient_clipping = gradient_clipping
        return y, ht

    @staticmethod
    def backward(ctx, dy, dht):  # dht is unused: upstream triton kernel doesn't feed it
        from state_tracking.models.rnn.triton.backward import _m2rnn_backward_triton

        q, k, v, W, xf, h0_saved, h = ctx.saved_tensors
        h0 = h0_saved if ctx.has_h0 else None

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        # dW is accumulated atomically inside the kernel in fp32
        dW = torch.zeros_like(W, dtype=torch.float32)
        dxf = torch.empty_like(xf)
        dh0 = torch.empty_like(h0) if (h0 is not None and h0.requires_grad) else None

        _m2rnn_backward_triton(
            q=q,
            k=k,
            v=v,
            W=W,
            xf=xf,
            h0=h0,
            dy=dy.contiguous(),
            h=h,
            dq=dq,
            dk=dk,
            dv=dv,
            dW=dW,
            dxf=dxf,
            dh0=dh0,
            cu_seqlens=None,
            gradient_clipping=ctx.gradient_clipping,
        )

        return dq, dk, dv, dW.to(W.dtype), dxf, dh0, None


def m2rnn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    forget_input: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
    backend: str = "torch",
) -> tuple[torch.Tensor, torch.Tensor]:
    """M2RNN recurrence with a backend switch.

    Parameters
    ----------
    backend : {"torch", "triton"}
        "torch"  - pure-PyTorch reference (works on CPU and CUDA).
        "triton" - vendored triton kernels from upstream. CUDA only.
    """
    if backend == "triton":
        if not torch.cuda.is_available():
            raise RuntimeError("m2rnn: triton backend requires CUDA")
        return _M2RNNTritonFunction.apply(
            query, key, value, weight, forget_input, input_state, gradient_clipping
        )
    if backend != "torch":
        raise ValueError(f"m2rnn: unknown backend {backend!r} (expected 'torch' or 'triton')")
    return _m2rnn_torch(
        query, key, value, weight, forget_input, input_state, gradient_clipping
    )
