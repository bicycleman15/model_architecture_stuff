import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask

from models.transformer import (
    TransformerConfig,
    Transformer,
    TransformerBlock,
    RMSNorm,
)
from liger_kernel.transformers import LigerRMSNorm


@dataclass
class MeanResidualTransformerConfig(TransformerConfig):
    alpha: float = 1.0
    mean_power: float = 1.0


class MeanResidualTransformerBlock(TransformerBlock):
    """TransformerBlock with mean-recurrence connections instead of additive residuals."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        if config.use_fused_ops:
            self.attn_out_norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
            self.ffn_out_norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.attn_out_norm = RMSNorm(config.dim, eps=config.norm_eps)
            self.ffn_out_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        mu: Tensor,
        t: int,
        sqrt_d: float,
        alpha: float,
        p: float,
        is_causal: Optional[bool] = True,
        mask: Optional[BlockMask] = None,
        input_pos: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, int]:

        # --- attention sublayer ---
        attn_out = self.attention(
            self.attention_norm(x), cos, sin, is_causal, mask=mask, input_pos=input_pos
        )

        # we assume we get the correct mu_t upto this position
        # x_{t+1} = RMSNorm(sqrt(d) * tanh(f_t(x_t) / sqrt(d))) - alpha * mu_t
        x_new = self.attn_out_norm(sqrt_d * torch.tanh(attn_out / sqrt_d)) - alpha * mu

        # mu_{t+1} = (t^p / (t+1)^p) * mu_t + (1 / (t+1)^p) * x_t
        mu = (t ** p / (t + 1) ** p) * mu + (1 / (t + 1) ** p) * x

        t += 1
        x = x_new

        # --- FFN sublayer ---
        # do the same as above
        ffn_out = self.feed_forward(self.ffn_norm(x))
        x_new = self.ffn_out_norm(sqrt_d * torch.tanh(ffn_out / sqrt_d)) - alpha * mu
        mu = (t ** p / (t + 1) ** p) * mu + (1 / (t + 1) ** p) * x
        t += 1
        x = x_new

        return x, mu, t


class MeanResidualTransformer(Transformer):

    def __init__(self, config: MeanResidualTransformerConfig) -> None:
        super().__init__(config)

        self.layers = nn.ModuleList(
            MeanResidualTransformerBlock(config, layer_idx=i)
            for i in range(config.n_layer)
        )
        self._init_weights(config.initializer_range)

        # if config.use_fused_ops:
        #     self.emb_rms_norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        # else:
        #     self.emb_rms_norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.alpha = config.alpha
        self.mean_power = config.mean_power
        self.sqrt_d = math.sqrt(config.dim)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        input_pos: Optional[Tensor] = None,
        mask: Optional[BlockMask] = None,
        **kwargs,
    ) -> Tensor:
        bsz, seqlen = input_ids.shape
        stats = {}
        log_norms = kwargs.get("log_norms", False)

        if (mask is not None) and (input_pos is not None):
            mask.mask_mod = self.get_mask_mod(mask.mask_mod, input_pos[0])

        if input_pos is not None:
            cos = self.cos[:, input_pos]
            sin = self.sin[:, input_pos]
        else:
            cos = self.cos[:, :seqlen]
            sin = self.sin[:, :seqlen]

        x = self.wte(input_ids)
        # # lets put a norm here
        # x = self.emb_rms_norm(x)

        if log_norms:
            stats["norm/embed"] = x.float().norm(dim=-1).mean().item()

        mu = torch.zeros_like(x)
        t = 0

        for i, layer in enumerate(self.layers):
            x, mu, t = layer(
                x, cos, sin, mu, t, self.sqrt_d, self.alpha, self.mean_power,
                mask=mask, input_pos=input_pos,
            )
            if log_norms and i % 4 == 3:
                stats[f"norm/layer_{i}"] = x.float().norm(dim=-1).mean().item()

        x = self.norm(x)

        if log_norms:
            stats["norm/final"] = x.float().norm(dim=-1).mean().item()

        if labels is not None:
            if self.config.use_fused_ops:
                loss = self.fused_linear_cross_entropy(
                    self.output.weight, x.view(-1, x.size(-1)), labels.view(-1)
                )
                return loss, stats
            else:
                logits = self.output(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
                )
                return loss, stats

        logits = self.output(x)
        return logits, stats


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = MeanResidualTransformerConfig(
        block_size=2048,
        n_layer=12,
        n_head=12,
        dim=768,
        alpha=1.0,
    )
    model = MeanResidualTransformer(config)
    model.to(device)
    model.setup_cache(device=device)

    input_ids = torch.randint(0, config.vocab_size, (1, config.block_size), device=device)
    print("input_ids.shape:", input_ids.shape, input_ids.dtype)

    logits, _ = model(input_ids)
    print("logits.shape:", logits.shape, logits.dtype)

    labels = torch.randint(0, config.vocab_size, (1, config.block_size), device=device)
    loss, _ = model(input_ids, labels=labels)
    print("loss:", loss.item())
