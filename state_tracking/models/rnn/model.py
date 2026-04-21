"""Transformer-style LM wrapper around M2RNN (for state-tracking baselines).

Structure per block:

    x -> RMSNorm -> M2RNN -> residual -> RMSNorm -> SwiGLU MLP -> residual

and the full model:

    [B, T] --(embed)--> [B, T, H] --(N x RNNBlock)--> [B, T, H]
           --(RMSNorm)--> --(lm_head)--> [B, T, V]

M2RNN is an autoregressive recurrence, so no positional embeddings are added.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layer import M2RNNLayer


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RNNBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size)
        self.mixer = M2RNNLayer(
            input_size=hidden_size,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            output_size=hidden_size,
            num_query_heads=num_heads,
            num_key_heads=num_heads,
            num_value_heads=num_heads,
            num_forget_input_heads=num_heads,
            num_weight_heads=num_heads,
            add_bias=False,
            gradient_clipping=None,
            backend=backend,
        )
        self.norm2 = nn.RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

    def forward(self, x: Tensor) -> Tensor:
        mixer_out, _ = self.mixer(self.norm1(x))
        x = x + mixer_out
        x = x + self.mlp(self.norm2(x))
        return x


class RNNModel(nn.Module):
    """LM head built on stacked M2RNN blocks. ``forward(x: [B, T]) -> [B, T, V]``."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        n_layers: int,
        num_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList(
            [
                RNNBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    key_head_dim=key_head_dim,
                    value_head_dim=value_head_dim,
                    backend=backend,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_f = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm_f(h)
        return self.lm_head(h)


def build_rnn(vocab_size: int, mcfg) -> RNNModel:
    backend = getattr(mcfg, "backend", "torch") or "torch"
    return RNNModel(
        vocab_size=vocab_size,
        hidden_size=mcfg.hidden_size,
        intermediate_size=mcfg.intermediate_size,
        n_layers=mcfg.n_layers,
        num_heads=mcfg.num_heads,
        key_head_dim=mcfg.key_head_dim,
        value_head_dim=mcfg.value_head_dim,
        backend=backend,
    )
