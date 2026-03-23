"""
H-Net: Hierarchical Network with Dynamic Chunking (arXiv:2507.07955)

Implements the H-Net architecture on top of the existing Hourglass model.
Key novel components over the base hourglass:
  1. Cosine-similarity-based routing module (replaces MLP router)
  2. EMA-based dechunking / smoothing module (replaces hard-copy expansion)
  3. Target rate auxiliary loss for controlling compression ratio
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from dataclasses import dataclass
from typing import Optional, List

from models.hourglass import (
    Config,
    Compressor,
    Decoder,
    Processor,
    HierarchicalModel,
    HierarchicalLM,
    apply_optimization_params,
)
from models.transformer import TransformerBlock, RMSNorm, build_rope_cache, find_multiple
from liger_kernel.transformers import LigerRMSNorm, LigerFusedLinearCrossEntropyLoss


@dataclass
class HNetConfig(Config):
    target_downsample_rate: float = 0.2
    target_rate_weight: float = 0.01


class HNetCompressor(Compressor):
    """Compressor with cosine-similarity routing (H-Net paper Section 2.2).

    Boundary probability between adjacent tokens x_t and x_{t+1}:
        cos_sim = cosine_sim(q_proj(x_t), k_proj(x_{t+1}))
        boundary_prob_t = (1 - cos_sim) / 2

    Q/K projections are initialized to identity so the router starts as
    raw cosine similarity and learns to specialize during training.
    """

    def __init__(self, config):
        super().__init__(config)
        del self.router

        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.dim, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(config.dim))
            self.k_proj.weight.copy_(torch.eye(config.dim))
        self.q_proj.weight._no_reinit = True
        self.k_proj.weight._no_reinit = True

    def chunk(self, x, input_ids=None):
        B, L, D = x.shape

        cos_sim = F.cosine_similarity(
            self.q_proj(x[:, :-1]),
            self.k_proj(x[:, 1:]),
            dim=-1,
        )  # [B, L-1]

        boundary_prob = torch.clamp((1 - cos_sim) / 2, min=0.0, max=1.0)  # [B, L-1]
        boundary_prob = F.pad(boundary_prob, (1, 0), value=1.0)  # [B, L]

        # don't use STE here
        # we just gather here for compression
        boundaries_mask = (boundary_prob > 0.5).float()
        boundaries_mask[:, 0] = 1.0

        select, boundary_positions, counts = self._boundaries_mask_to_select(
            boundaries_mask, x.device
        )
        return select, boundary_positions, boundary_prob, counts

    def forward(self, x, cos, sin, input_ids=None):
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)

        compressed_x, boundaries, probs, counts = self.compress(x, input_ids=input_ids)
        avg_chunk_size = x.shape[1] / counts.float().mean().item()

        return x, compressed_x, boundaries, counts, avg_chunk_size, probs


class HNetHierarchicalModel(HierarchicalModel):

    def __init__(self, config, depth=0):
        super().__init__(config, depth)
        self.compressor = HNetCompressor(config)
        if config.processor_config is not None:
            self.processor = HNetHierarchicalModel(config.processor_config, depth=depth + 1)

    def _expand_chunks(self, x_processed, boundaries, counts, seq_len):
        """Expand chunk-level representations back to full sequence length."""
        B, K, D = x_processed.shape
        L = seq_len
        device = x_processed.device

        positions = einops.rearrange(torch.arange(L, device=device), 'L -> 1 1 L')
        starts = boundaries.unsqueeze(-1)  # [B, K, 1]
        valid_mask = counts[:, None] > torch.arange(K, device=device)[None, :]  # [B, K]
        starts = torch.where(valid_mask.unsqueeze(-1), starts, L)
        chunk_ids = (positions >= starts).sum(dim=1) - 1  # [B, L]
        chunk_ids = chunk_ids.clamp(min=0)

        batch_idx = einops.repeat(torch.arange(B, device=device), 'b -> b L', L=L)
        return x_processed[batch_idx, chunk_ids]  # [B, L, D]

    def ema_dechunk(self, chunk_reps, boundaries, counts, boundary_probs, seq_len):
        """EMA-based dechunking: smooth chunk representations then expand.

        Runs an exponential moving average at the chunk level:
            h[0] = chunk_reps[0]
            h[i] = p[i] * chunk_reps[i] + (1 - p[i]) * h[i-1]

        p[i] is the boundary probability at chunk i's start position.
        Confident boundaries (p~1) reset the EMA; uncertain ones (p~0.5)
        blend with the previous chunk for smoother transitions and better
        gradient flow through the discrete routing decisions.
        """
        B, S, D = chunk_reps.shape
        device = chunk_reps.device

        boundary_p = torch.gather(
            boundary_probs, 1, boundaries.clamp(max=boundary_probs.shape[1] - 1)
        )  # [B, S]
        boundary_p = boundary_p.clamp(min=1e-4, max=1 - 1e-4)

        valid_mask = counts[:, None] > torch.arange(S, device=device)[None, :]  # [B, S]

        # Build smoothed chunks via list + cat to avoid in-place ops
        # and maintain gradient flow back to chunk_reps and boundary_p.
        chunks = [chunk_reps[:, 0:1]]  # [B, 1, D], keeps dim
        for i in range(1, S):
            p = boundary_p[:, i:i+1].unsqueeze(-1)  # [B, 1, 1]
            prev = chunks[-1]
            curr = chunk_reps[:, i:i+1]

            # h[0] = chunk_reps[0]
            # h[i] = p[i] * chunk_reps[i] + (1 - p[i]) * h[i-1]
            new_h = torch.where(
                valid_mask[:, i:i+1].unsqueeze(-1),
                p * curr + (1 - p) * prev,
                torch.zeros(1, device=device, dtype=chunk_reps.dtype),
            )
            chunks.append(new_h)
        h = torch.cat(chunks, dim=1)  # [B, S, D]

        # now expand like you'd normally do in hourglass transformer
        return self._expand_chunks(h, boundaries, counts, seq_len)

    def forward(self, x, input_ids=None):
        B, L, D = x.shape

        cos = self.cos[:, :L]
        sin = self.sin[:, :L]

        x_encoder, x_compressed, boundaries, counts, avg_chunk_size, boundary_probs = \
            self.compressor(x, cos, sin, input_ids=input_ids)

        stats = {f"level_{self.depth}/avg_chunk_size": avg_chunk_size}
        stats[f"level_{self.depth}/mean_boundary_prob"] = boundary_probs.mean().item()

        # pass through processor
        x_compressed = self.upsample(x_compressed)
        processor_out = self.processor(x_compressed)
        if isinstance(processor_out, tuple):
            x_processed = processor_out[0]
            stats.update(processor_out[1])
        else:
            x_processed = processor_out
        x_processed = self.downsample(x_processed, D)

        # decoder
        # we don't pass it to decoder for now
        # we do that logic here
        x_smoothed = self.ema_dechunk(x_processed, boundaries, counts, boundary_probs, L)
        x = x_smoothed + x_encoder
        for layer in self.decoder.layers:
            x = layer(x, cos, sin)
        x = self.decoder.norm(x)

        return x, stats, boundary_probs

    def _init_weights(self, initializer_range: float = 0.02, parent_residuals: int = 0):
        """Depth-scaled weight init, preserving identity-initialized Q/K projections."""
        n_residuals = parent_residuals + self.config.n_compressor_layers * 2 + self.config.n_decoder_layers * 2
        out_std = initializer_range / math.sqrt(n_residuals)
        print(f"[Init] depth={self.depth}: n_residuals={n_residuals} "
              f"(parent={parent_residuals} + comp={self.config.n_compressor_layers * 2} "
              f"+ dec={self.config.n_decoder_layers * 2})")
        print(f"[Init]   compressor/decoder output proj std={out_std:.6f}, other linear std={initializer_range}")
        print(f"[Init]   q_proj/k_proj: identity-initialized (skipped)")

        for block_list in [self.compressor.layers, self.decoder.layers]:
            for name, m in block_list.named_modules():
                if isinstance(m, nn.Linear):
                    if name.endswith(".wo") or name.endswith(".proj"):
                        nn.init.normal_(m.weight, mean=0.0, std=out_std)
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        if isinstance(self.processor, HierarchicalModel):
            self.processor._init_weights(initializer_range, n_residuals)
        else:
            proc_n_residuals = n_residuals + self.processor.config.n_processor_layers * 2
            proc_out_std = initializer_range / math.sqrt(proc_n_residuals)
            print(f"[Init] depth={self.depth + 1} (processor): n_residuals={proc_n_residuals} "
                  f"(outer={n_residuals} + proc={self.processor.config.n_processor_layers * 2})")
            print(f"[Init]   processor output proj std={proc_out_std:.6f}, other linear std={initializer_range}")
            for name, m in self.processor.named_modules():
                if isinstance(m, nn.Linear):
                    if name.endswith(".wo") or name.endswith(".proj"):
                        nn.init.normal_(m.weight, mean=0.0, std=proc_out_std)
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


class HNetLM(HierarchicalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = HNetHierarchicalModel(config)

    def forward(self, input_ids, labels=None):
        x = self.emb(input_ids)  # [B, L, D]
        out, stats, boundary_probs = self.model(x, input_ids=input_ids)

        if labels is not None:
            # Compute target rate loss BEFORE fused CE, since LigerFusedLinearCrossEntropyLoss
            # may modify `out` in-place and corrupt the shared computation graph.
            target_rate_loss = None
            if boundary_probs is not None and self.config.target_rate_weight > 0:
                # H-Net load-balancing loss (hnet/utils/train.py).
                # N = downsampling factor (e.g. target_rate=0.2 → N=5).
                N = 1.0 / self.config.target_downsample_rate
                boundary_mask = (boundary_probs > 0.5)
                true_ratio = boundary_mask.float().mean()       # hard — no gradient
                average_prob = boundary_probs.mean()             # soft — has gradient
                target_rate_loss = (
                    (1 - true_ratio) * (1 - average_prob)
                    + true_ratio * average_prob * (N - 1)
                ) * N / (N - 1)

            # if self.config.use_fused_ops:
            #     ce_loss = self.fused_linear_cross_entropy(
            #         self.vocab.weight, out.view(-1, out.size(-1)), labels.view(-1)
            #     )
            # else:

            # lets not use fused ops here for now
            # it can interfere with total losses and gradient of that 
            # due to in-place updates in fused losses
            logits = self.vocab(out)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )

            total_loss = ce_loss
            if target_rate_loss is not None:
                total_loss = total_loss + self.config.target_rate_weight * target_rate_loss
                stats["hnet/ce_loss"] = ce_loss.item()
                stats["hnet/target_rate_loss"] = target_rate_loss.item()

            return total_loss, stats

        logits = self.vocab(out)  # [B, L, V]
        return logits, stats


if __name__ == "__main__":

    config = HNetConfig(
        block_size=128,
        vocab_size=256,
        dim=128,
        n_head=4,
        n_compressor_layers=2,
        n_processor_layers=4,
        n_decoder_layers=2,
        chunk_method="router",
        target_downsample_rate=0.2,
        target_rate_weight=0.01,
    )

    model = HNetLM(config)
    model._init_weights()

    print(model)
    print(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.setup_cache(device=device)

    input_ids = torch.randint(0, config.vocab_size, (2, 128), device=device)
    labels = torch.randint(0, config.vocab_size, (2, 128), device=device)
    print("input_ids.shape:", input_ids.shape)

    # training mode
    model.train()
    loss, stats = model(input_ids, labels=labels)
    print(f"loss={loss.item():.4f}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    # backward pass
    loss.backward()
    grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
    print(f"\nBackward OK — {len(grad_norms)} params have gradients")
    print(f"  q_proj grad norm: {grad_norms.get('model.compressor.q_proj.weight', 0):.6f}")
    print(f"  k_proj grad norm: {grad_norms.get('model.compressor.k_proj.weight', 0):.6f}")

    processor_grads = [v for k, v in grad_norms.items() if "processor" in k]
    if processor_grads:
        print(f"  processor grad norms: min={min(processor_grads):.6f}, max={max(processor_grads):.6f}")
    else:
        print("  WARNING: processor has NO gradients!")

    # inference mode
    model.eval()
    with torch.no_grad():
        logits, stats = model(input_ids)
    print("logits.shape:", logits.shape)
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
