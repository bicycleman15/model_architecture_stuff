"""
REINFORCE-based learned tokenization for the Hourglass architecture.

Implements the method from "You Can Learn Tokenization End-to-End with Reinforcement Learning" (arXiv:2602.13940). 

Instead of straight-through estimation for token boundaries, uses stochastic sampling with REINFORCE policy gradient. 

A baseline vocab head on compressor output provides variance reduction, combined with time-discounted per-position rewards.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List

from models.hourglass import (
    Config,
    Compressor,
    HierarchicalModel,
    HierarchicalLM,
    Processor,
    apply_optimization_params,
)


@dataclass
class ReinforceConfig(Config):
    # new hps
    reinforce_gamma: float = 0.99
    target_downsample_rate: float = 0.2

    # handle router logits/probs
    router_logit_scale: float = 16.0
    router_softcap: float = 50.0


class ReinforceCompressor(Compressor):
    """Compressor that samples token boundaries stochastically via Bernoulli
    during training (deterministic threshold at eval).

    Returns (x, compressed_x, boundaries, counts, avg_chunk_size, probs, log_probs)
    where probs/log_probs are None for non-router chunk methods.
    """

    def chunk(self, x, input_ids=None):
        if self.config.chunk_method != "router":
            select, boundary_positions, probs, counts = super().chunk(x, input_ids)
            return select, boundary_positions, probs, counts, None

        B, L, D = x.shape

        raw_logits = self.router(x).squeeze(-1)  # [B, L]

        # Eq 18: scale raw logits and bias toward target downsample rate
        logit_bias = math.log(self.config.target_downsample_rate
                              / (1.0 - self.config.target_downsample_rate))
        logits = raw_logits / self.config.router_logit_scale + logit_bias

        # Eq 19: softcap to prevent exploding logits (training only)
        if self.training and self.config.router_softcap > 0:
            cap = self.config.router_softcap
            logits = cap * torch.tanh(logits / cap)

        probs = torch.sigmoid(logits)  # [B, L]

        boundaries_mask = torch.bernoulli(probs)

        boundaries_mask[:, 0] = 1.0

        # log pi(a_i) = a_i * log(p_i) + (1 - a_i) * log(1 - p_i)
        log_probs = (
            boundaries_mask * torch.log(probs.clamp(min=1e-8))
            + (1 - boundaries_mask) * torch.log((1 - probs).clamp(min=1e-8))
        )

        select, boundary_positions, counts = self._boundaries_mask_to_select(
            boundaries_mask, x.device
        )
        return select, boundary_positions, probs, counts, log_probs

    def compress(self, x, input_ids=None):
        select, boundary_positions, probs, counts, log_probs = self.chunk(x, input_ids=input_ids)
        compressed_x = select @ x  # [B, S, D]
        return compressed_x, boundary_positions, probs, counts, log_probs

    def forward(self, x, cos, sin, input_ids=None):
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)

        compressed_x, boundaries, probs, counts, log_probs = self.compress(x, input_ids=input_ids)
        avg_chunk_size = x.shape[1] / counts.float().mean().item()

        return x, compressed_x, boundaries, counts, avg_chunk_size, probs, log_probs


class ReinforceHierarchicalModel(HierarchicalModel):

    def __init__(self, config, depth=0):
        super().__init__(config, depth)
        self.compressor = ReinforceCompressor(config)

    def forward(self, x, input_ids=None):
        B, L, D = x.shape

        cos = self.cos[:, :L]
        sin = self.sin[:, :L]

        x, x_compressed, boundaries, counts, avg_chunk_size, probs, log_probs = \
            self.compressor(x, cos, sin, input_ids=input_ids)

        stats = {f"level_{self.depth}/avg_chunk_size": avg_chunk_size}

        x_compressed = self.upsample(x_compressed)

        processor_out = self.processor(x_compressed)
        if isinstance(processor_out, tuple):
            x_processed, inner_stats = processor_out
            stats.update(inner_stats)
        else:
            x_processed = processor_out

        x_processed = self.downsample(x_processed, D)

        out = self.decoder(x_processed, boundaries, counts, x, cos, sin, L)

        return out, stats, x, probs, log_probs


class ReinforceHierarchicalLM(HierarchicalLM):

    def __init__(self, config):
        super().__init__(config)
        # Replace model with the REINFORCE variant
        self.model = ReinforceHierarchicalModel(config)
        self.baseline_vocab = nn.Linear(config.dim, config.padded_vocab_size, bias=False)

    def _init_weights(self, initializer_range: float = 0.02):
        super()._init_weights(initializer_range)
        print(f"[Init] ReinforceHierarchicalLM: baseline_vocab std={initializer_range}")
        nn.init.normal_(self.baseline_vocab.weight, mean=0.0, std=initializer_range)

    def apply_lr_multiplier(self, lr_multiplier: List[float]):
        super().apply_lr_multiplier(lr_multiplier)
        for param in self.baseline_vocab.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[0])

    def _compute_discounted_returns(self, advantage, gamma):
        """Reverse cumulative sum with exponential discount.

        R[:, t] = advantage[:, t] + gamma * R[:, t+1]
        """
        B, L = advantage.shape
        R = torch.zeros_like(advantage)
        R[:, -1] = advantage[:, -1]
        for t in range(L - 2, -1, -1):
            R[:, t] = advantage[:, t] + gamma * R[:, t + 1]
        return R

    def forward(self, input_ids, labels=None):
        x = self.emb(input_ids)  # [B, L, D]
        out, stats, compressor_hidden, probs, log_probs = self.model(x, input_ids=input_ids)

        if labels is not None:
            B, L = labels.shape

            # --- main CE loss (per-position) ---
            logits = self.vocab(out)
            per_pos_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(B, L)
            ce_loss = per_pos_loss.mean()

            # Non-router chunk methods don't produce REINFORCE signals.
            if log_probs is None:
                return ce_loss, stats

            # --- baseline CE loss (compressor hidden -> baseline vocab head) ---
            baseline_logits = self.baseline_vocab(compressor_hidden)
            baseline_per_pos = F.cross_entropy(
                baseline_logits.view(-1, baseline_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(B, L)
            baseline_ce = baseline_per_pos.mean()

            # --- REINFORCE loss ---
            # advantage: positive means the processor didn't help enough
            # (actual > baseline), so penalise the boundary decisions that led here.
            advantage = (per_pos_loss - baseline_per_pos).detach()

            gamma = self.config.reinforce_gamma
            R = self._compute_discounted_returns(advantage, gamma)

            reinforce_loss = (R * log_probs).mean()

            # --- target downsample-rate regulariser ---
            mean_prob = probs.mean()
            target_rate_loss = (mean_prob - self.config.target_downsample_rate) ** 2

            total_loss = (
                ce_loss # log p_theta maximize
                + (0.01 * reinforce_loss) # reward * log a_theta -- papers says it uses a small values here since it wants to "encourage exploration"

                # below are rest of the stuff we want to do besides maximize the expectation
                # all hps are according to the paper
                + (0.1 * baseline_ce)
                + (0.01 * target_rate_loss)
            )

            stats["reinforce/ce_loss"] = ce_loss.item()
            stats["reinforce/baseline_loss"] = baseline_ce.item()
            stats["reinforce/reinforce_loss"] = reinforce_loss.item()
            stats["reinforce/target_rate_loss"] = target_rate_loss.item()
            stats["reinforce/mean_boundary_prob"] = mean_prob.item()

            return total_loss, stats

        logits = self.vocab(out)  # [B, L, V]
        return logits, stats


if __name__ == "__main__":

    config = ReinforceConfig(
        block_size=128,
        vocab_size=256,
        dim=128,
        n_head=4,
        n_compressor_layers=2,
        n_processor_layers=4,
        n_decoder_layers=2,
        chunk_method="router",
        target_downsample_rate=0.2,
        reinforce_gamma=0.99,
    )

    model = ReinforceHierarchicalLM(config)
    model._init_weights()

    print(model)
    print(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.setup_cache(device=device)

    input_ids = torch.randint(0, config.vocab_size, (2, 128), device=device)
    labels = torch.randint(0, config.vocab_size, (2, 128), device=device)
    print("input_ids.shape:", input_ids.shape)

    # training mode (stochastic sampling)
    model.train()
    loss, stats = model(input_ids, labels=labels)
    print(f"loss={loss.item():.4f}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    # inference mode (deterministic threshold)
    model.eval()
    with torch.no_grad():
        logits, stats = model(input_ids)
    print("logits.shape:", logits.shape)
