# **************************************************
# Copyright (c) 2025, Mayank Mishra
#
# Vendored from open-lm-engine/accelerated-model-architectures
#   xma/layers/m2rnn/layer.py
#
# Edits (for state_tracking baseline):
#   - cu_seqlens / max_seqlen arguments removed (fixed-length S_3 training).
#   - Exposes a `backend` argument ("torch" or "triton") that is forwarded
#     to the local `m2rnn` op.
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from .op import divide_if_divisible, m2rnn


class M2RNNLayer(nn.Module):
    """Memory-matrix RNN mixer layer.

    Projects the input into grouped Q/K/V + forget signals, runs the
    pure-torch recurrence in `m2rnn`, and projects the per-head outputs
    back to `output_size` via a shared linear.
    """

    def __init__(
        self,
        input_size: int,
        key_head_dim: int,
        value_head_dim: int,
        output_size: int,
        num_query_heads: int,
        num_key_heads: int,
        num_value_heads: int,
        num_forget_input_heads: int,
        num_weight_heads: int,
        add_bias: bool = False,
        gradient_clipping: float | None = None,
        backend: str = "torch",
    ) -> None:
        super().__init__()

        self.gradient_clipping = gradient_clipping
        self.backend = backend
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim

        self.num_query_heads = num_query_heads
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.num_forget_input_heads = num_forget_input_heads
        self.num_weight_heads = num_weight_heads

        self.num_heads = max(
            num_query_heads,
            num_key_heads,
            num_value_heads,
            num_forget_input_heads,
            num_weight_heads,
        )
        self.state_size = self.num_heads * self.value_head_dim

        divide_if_divisible(self.num_heads, self.num_query_heads)
        divide_if_divisible(self.num_heads, self.num_key_heads)
        divide_if_divisible(self.num_heads, self.num_value_heads)
        divide_if_divisible(self.num_heads, self.num_forget_input_heads)
        divide_if_divisible(self.num_heads, self.num_weight_heads)

        self.input_projection = nn.Linear(
            input_size,
            self.num_query_heads * self.key_head_dim
            + self.num_key_heads * self.key_head_dim
            + self.num_value_heads * self.value_head_dim
            + self.num_forget_input_heads,
            bias=add_bias,
        )

        self.state_weight = nn.Parameter(
            torch.empty(self.num_weight_heads, self.value_head_dim, self.value_head_dim)
        )
        self.output_projection = nn.Linear(self.state_size, output_size, bias=False)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight)

    def forward(
        self,
        input: torch.Tensor,
        input_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proj = self.input_projection(input)
        q, k, v, f = proj.split(
            (
                self.num_query_heads * self.key_head_dim,
                self.num_key_heads * self.key_head_dim,
                self.num_value_heads * self.value_head_dim,
                self.num_forget_input_heads,
            ),
            dim=-1,
        )

        q = q.view(*q.size()[:-1], -1, self.key_head_dim)
        k = k.view(*k.size()[:-1], -1, self.key_head_dim)
        v = v.view(*v.size()[:-1], -1, self.value_head_dim)

        if input_state is not None:
            input_state = input_state.view(
                -1, self.num_heads, self.key_head_dim, self.value_head_dim
            )

        # The triton kernel's inner `tl.dot(h_prev, W)` requires matching
        # dtypes for A and B. Under `accelerator.autocast`, activations
        # (q/k/v/f) get cast to bf16 while `state_weight` is still an
        # fp32 Parameter - so cast it to the active activation dtype
        # before dispatching. (The pure-torch backend tolerates the mix
        # via PyTorch's type promotion, but we do the same cast there
        # for behavioural parity.)
        weight = self.state_weight.to(q.dtype)

        out, new_state = m2rnn(
            query=q,
            key=k,
            value=v,
            weight=weight,
            forget_input=f,
            input_state=input_state,
            gradient_clipping=self.gradient_clipping,
            backend=self.backend,
        )

        # out: [B, S, N, V] -> [B, S, N*V]
        out = out.flatten(-2, -1)
        new_state = new_state.flatten(-2, -1)

        return self.output_projection(out), new_state
