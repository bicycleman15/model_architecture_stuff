"""PathTransformer: Transformer with a custom autograd function applied to the
final logits so the output-layer gradient is rescaled in a natural-gradient-like
way before cross-entropy.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from models.transformer import Transformer as _BaseTransformer

import einops


class PathPreservingAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, damping):
        
        theta[..., -1] = 0 # this is the change!

        ctx.save_for_backward(theta)
        ctx.damping = damping
        return theta.clone()

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        grad_input = grad_output.clone()

        p = F.softmax(theta, dim=-1)

        # max_p = torch.max(p, dim=-1, keepdim=True).values
        # min_p = torch.min(p, dim=-1, keepdim=True).values

        # factor = 1 / (p + ctx.damping) # the max value is ~100
        # factor = 1 / (ctx.damping + (p ** (grad_input ** 2)))
        # factor = 1 / (p + ctx.damping * (1- p))
        # factor = (1 - max_p) * min_p / (p + 0.1)
        # factor = (1 - max_p) / (p + 0.1)
        # factor = 1
        factor = 1 / (p + (1-p)*torch.exp(-25*p))
        
        # factor = 1.0 / (p * (1.0 - p) + ctx.damping)

        # factor = 1
        modified_grad = grad_input * factor

        # breakpoint()

        modified_grad -= einops.repeat(modified_grad[..., -1:], "... 1 -> ... v", v=modified_grad.shape[-1]) # this is the change!

        # modified_grad is shape [B, L, V]
        # we just subtract the last dim of modified_grad to say to optimizer we don't want to update the last logit
        # basically modified_grad[..., -1] is zero!

        #modified_grad[..., -1] = 0

        # modified_grad = modified_grad - modified_grad.mean(dim=-1, keepdim=True)
        # [B, L, V]

        return modified_grad, None


class PathTransformer(_BaseTransformer):
    def __init__(self, config, damping: float = 1e-2) -> None:
        super().__init__(config)
        self.damping = damping

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tensor:
        # if self.config.use_fused_ops:
        #     raise ValueError(
        #         "PathTransformer does not support use_fused_ops=True: "
        #         "LigerFusedLinearCrossEntropyLoss fuses the output projection "
        #         "with softmax+CE and does not expose logits for "
        #         "PathPreservingAutograd to wrap."
        #     )

        logits, stats = super().forward(input_ids=input_ids, labels=None, **kwargs)
        logits = PathPreservingAutograd.apply(logits, self.damping)

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return loss, stats

        return logits, stats
