import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


# ---- RoPE utilities ----

def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: float = 10000,
) -> Tuple[Tensor, Tensor]:
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    seq_idx = torch.arange(seq_len, device=device).float()
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)
    if idx_theta.shape[-1] > n_elem > 1:
        idx_theta = idx_theta[..., :n_elem]
    return torch.cos(idx_theta).unsqueeze(0), torch.sin(idx_theta).unsqueeze(0)


@torch.amp.autocast("cuda", enabled=False)
def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    head_size_half = x.size(-1) // 2
    x1 = x[..., :head_size_half]
    x2 = x[..., head_size_half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    dims_diff = x.dim() - cos.dim()
    if dims_diff > 0:
        new_shape = cos.shape[0:1] + (1,) * dims_diff + cos.shape[1:]
        cos = cos.view(*new_shape)
        sin = sin.view(*new_shape)
    return ((x * cos) + (rotated * sin)).to(dtype=x.dtype)


@torch.amp.autocast("cuda", enabled=False)
def apply_rope_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_n_elem: int) -> Tensor:
    x_roped = apply_rope(x[..., :rope_n_elem], cos, sin)
    return torch.cat((x_roped, x[..., rope_n_elem:]), dim=-1)


# ---- Core building blocks ----

class RMSNorm(nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (x_normed * self.weight.float()).to(dtype=dtype)


class SwiGLUMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.fc_2 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.proj = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.silu(self.fc_1(x)) * self.fc_2(x))


class Attention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_local_heads = config.n_local_heads
        self.head_dim = config.head_dim
        self.dim = config.dim
        self.rope_n_elem = config.rope_n_elem

        self.wqkv = nn.Linear(
            config.dim,
            (config.n_head + 2 * config.n_local_heads) * config.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(config.n_head * config.head_dim, config.dim, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, is_causal: bool = True) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)

        q = apply_rope_emb(q, cos, sin, self.rope_n_elem)
        k = apply_rope_emb(k, cos, sin, self.rope_n_elem)

        scale = 1.0 / math.sqrt(self.head_dim)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0,
            scale=scale, is_causal=is_causal,
            enable_gqa=(self.n_head != self.n_local_heads),
        )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.wo(y)


class TransformerBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = SwiGLUMLP(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, is_causal: bool = True) -> Tensor:
        h = x + self.attention(self.attention_norm(x), cos, sin, is_causal)
        return h + self.feed_forward(self.ffn_norm(h))
