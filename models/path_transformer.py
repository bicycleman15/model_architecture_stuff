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


class PathPreservingAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, damping):
        ctx.save_for_backward(theta)
        ctx.damping = damping
        return theta.clone()

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        grad_input = grad_output.clone()

        probs = F.softmax(theta, dim=-1)

        modified_grad = grad_input / (probs + ctx.damping)
        modified_grad = modified_grad - modified_grad.mean(dim=-1, keepdim=True)

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
