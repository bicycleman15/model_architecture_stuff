import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from dataclasses import dataclass
from typing import Optional, List

from models.transformer import TransformerBlock, RMSNorm, build_rope_cache, find_multiple
from liger_kernel.transformers import LigerRMSNorm, LigerFusedLinearCrossEntropyLoss


def apply_optimization_params(param: torch.Tensor, **kwargs) -> None:
    """Annotate a parameter with per-param optimizer settings (e.g. lr_multiplier, weight_decay)."""
    if hasattr(param, "_optim"):
        param._optim.update(kwargs)
    else:
        param._optim = kwargs


@dataclass
class Config:
    block_size: int = 128
    vocab_size: int = 32000
    padded_vocab_size: Optional[int] = None

    # shared transformer dimensions
    dim: int = 768
    n_head: int = 12
    n_local_heads: int = -1
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None

    # per-stage layer counts
    n_compressor_layers: int = 3
    n_processor_layers: int = 6
    n_decoder_layers: int = 3

    # compressor
    chunk_method: str = "router"  # "uniform", "router", or "spacebyte"
    chunk_size: int = 4           # fixed chunk size for uniform chunking
    bos_idx: int = 254  # BOS token index (always treated as a boundary for spacebyte)
    pad_zero_idx: int = 0  # token ID of the \x00 padding byte

    # processor
    processor_dim: Optional[int] = None  # dim the processor operates at (default: 3/2 * dim)
    processor_config: Optional['Config'] = None  # None = flat transformer blocks, or a Config for recursive nesting

    # rope / norm
    rope_base: float = 10000
    norm_eps: float = 1e-5
    rope_n_elem: Optional[int] = None

    # weight init / lr
    initializer_range: float = 0.02
    lr_multiplier: Optional[List[float]] = None

    # optional
    use_fused_ops: bool = False
    use_qk_norm: bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)

        assert self.dim % self.n_head == 0
        self.head_dim = self.dim // self.n_head

        if self.processor_dim is None:
            self.processor_dim = (self.dim * 3) // 2

        self.padded_vocab_size = find_multiple(self.vocab_size, 256)
        self.rope_n_elem = self.head_dim


class Compressor(nn.Module):
    # compressor "implicitly" decides how to chunk up the sequence
    # it acts as a router
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            TransformerBlock(config, layer_idx=i) for i in range(config.n_compressor_layers)
        )
        if config.use_fused_ops:
            self.norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.router = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, 1),
        )

    def _boundaries_mask_to_select(self, boundaries_mask, device):
        """Convert a [B, L] binary boundary mask to the select matrix and positions."""
        counts = boundaries_mask.sum(dim=-1).long()  # [B]
        S = counts.max().item()

        cumsum = torch.cumsum(boundaries_mask.detach(), dim=-1)  # [B, L]
        segment_indices = torch.arange(1, S + 1, device=device).view(1, S, 1)

        assignment = (cumsum.unsqueeze(1) == segment_indices).float()  # [B, S, L]
        select = assignment * boundaries_mask.unsqueeze(1)  # [B, S, L]

        boundary_positions = boundaries_mask.detach().argsort(
            dim=-1, descending=True, stable=True
        )[:, :S]

        return select, boundary_positions, counts

    def chunk(self, x, input_ids=None):
        # decides how to chunk up x
        # (a) use uniform chunking
        # (b) use router based chunking
        # (c) spacebyte — deterministic boundaries from UTF-8 byte structure

        # returns (select, boundary_positions, probs, counts)
        # select: [B, S, L] — selector matrix, one-hot at chunk start positions
        # boundary_positions: [B, S] — start position of each chunk
        # probs: [B, L] or None — router probabilities (only for "router")
        # counts: [B] — number of chunks per batch element

        B, L, D = x.shape

        if self.config.chunk_method == "uniform":
            starts = torch.arange(0, L, self.config.chunk_size, device=x.device)  # [S]
            S = starts.shape[0]

            select = torch.zeros(B, S, L, device=x.device)
            select[:, torch.arange(S, device=x.device), starts] = 1.0

            boundary_positions = starts.unsqueeze(0).expand(B, S)
            counts = torch.full((B,), S, device=x.device, dtype=torch.long)

            return select, boundary_positions, None, counts

        elif self.config.chunk_method == "router":

            logits = self.router(x).squeeze(-1)  # [B, L]
            probs = torch.sigmoid(logits)  # [B, L]

            boundaries_mask = probs + ((probs > 0.5).float() - probs).detach()  # [B, L]
            boundaries_mask[:, 0] = 1.0 # force first token to be a boundary start

            ### helpers to get the select matrix
            select, boundary_positions, counts = self._boundaries_mask_to_select(
                boundaries_mask, x.device
            )
            return select, boundary_positions, probs, counts

        elif self.config.chunk_method == "spacebyte":
            assert input_ids is not None, "spacebyte chunking requires input_ids"

            # # UTF-8 byte boundary detection (https://github.com/kjslag/spacebyte):
            # # non-alphanumeric ASCII bytes or UTF-8 multi-byte start bytes (>= 0xC0)
            # is_boundary = (
            #     (input_ids < ord('0')) |
            #     ((ord('9') < input_ids) & (input_ids < ord('A'))) |
            #     ((ord('Z') < input_ids) & (input_ids < ord('a'))) |
            #     ((ord('z') < input_ids) & (input_ids < 0b1000_0000)) |
            #     (0b1100_0000 <= input_ids)
            # )
            #
            # is_boundary[:, 1:] &= is_boundary[:, :-1].bitwise_not()
            # is_boundary |= (input_ids == self.config.bos_idx)

            # boundary at every non-padding token: chunks become [char, 0, ..., 0]
            is_boundary = (input_ids != self.config.pad_zero_idx)

            boundaries_mask = is_boundary.float()
            select, boundary_positions, counts = self._boundaries_mask_to_select(
                boundaries_mask, x.device
            )
            return select, boundary_positions, None, counts

        else:
            raise NotImplementedError(f"Unknown chunk_method: {self.config.chunk_method}")


    def compress(self, x, input_ids=None):
        select, boundary_positions, probs, counts = self.chunk(x, input_ids=input_ids)
        compressed_x = select @ x  # [B, S, D]
        return compressed_x, boundary_positions, probs, counts


    def forward(self, x, cos, sin, input_ids=None):
        # x: [B, L, D]

        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)

        compressed_x, boundaries, probs, counts = self.compress(x, input_ids=input_ids)

        avg_chunk_size = x.shape[1] / counts.float().mean().item()

        return x, compressed_x, boundaries, counts, avg_chunk_size
        

class Decoder(nn.Module):
    # decoder expands chunk-level processor outputs back to full sequence length
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            TransformerBlock(config, layer_idx=i) for i in range(config.n_decoder_layers)
        )
        if config.use_fused_ops:
            self.norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x_processed, boundaries, counts, x_residual, cos, sin, seq_len):
        # x_processed: [B, K, D]
        # boundaries: [B, K] — start position of each chunk, sorted ascending
        # counts: [B] — number of valid chunks per batch element
        # x_residual: [B, L, D] — full-length output from compressor
        B, K, D = x_processed.shape
        L = x_residual.shape[1]

        # Expand chunk[i] to all positions in chunk[i] via indexing.
        # No causal shift: the compressed rep at start[i] only depends on
        # positions <= start[i], so broadcasting it to positions >= start[i] is causal.
        positions = einops.rearrange(torch.arange(L, device=x_processed.device), 'L -> 1 1 L')  # [1, 1, L]
        starts = boundaries.unsqueeze(-1)  # [B, K, 1]
        valid_mask = counts[:, None] > torch.arange(K, device=boundaries.device)[None, :]  # [B, K]
        starts = torch.where(valid_mask.unsqueeze(-1), starts, L)  # junk boundaries are set as L (never triggered)
        chunk_ids = (positions >= starts).sum(dim=1) - 1  # [B, L]

        batch_idx = einops.repeat(torch.arange(B, device=x_processed.device), 'b -> b L', L=L)  # [B, L]
        x = x_processed[batch_idx, chunk_ids]  # [B, L, D]

        # The expanded chunk representations are identical within each chunk.
        # Adding the compressor's full-length residual provides per-position
        # context so the decoder can distinguish individual token positions.
        x = x + x_residual

        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)

        return x  # [B, L, D]


class Processor(nn.Module):
    # processor processes compressed chunks
    # just transformer blocks with their own RoPE cache, forward(x) -> x

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            TransformerBlock(config, layer_idx=i) for i in range(config.n_processor_layers)
        )
        if config.use_fused_ops:
            self.norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.norm = RMSNorm(config.dim, eps=config.norm_eps)

    def setup_cache(self, device=None):
        cos, sin = build_rope_cache(
            self.config.block_size, self.config.rope_n_elem,
            device=device, base=self.config.rope_base,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        # x: [B, K, D]
        B, K, D = x.shape

        cos = self.cos[:, :K]
        sin = self.sin[:, :K]

        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)

        return x  # [B, K, D]


class HierarchicalModel(nn.Module):
    # pure embedding-to-embedding model with its own RoPE cache
    # forward(x) -> (x, stats_dict), making it usable as a processor in a parent HierarchicalModel

    def __init__(self, config, depth=0):
        super().__init__()
        self.config = config
        self.depth = depth

        self.compressor = Compressor(config)

        proc_dim = config.processor_dim
        if proc_dim > config.dim:
            self.pad_dimension = nn.Parameter(torch.zeros(proc_dim - config.dim))
        else:
            self.pad_dimension = None

        if config.processor_config is not None:
            self.processor = HierarchicalModel(config.processor_config, depth=depth + 1)
        else:
            proc_config = Config(
                block_size=config.block_size,
                vocab_size=config.vocab_size,
                dim=proc_dim,
                n_head=config.n_head,
                n_processor_layers=config.n_processor_layers,
                rope_base=config.rope_base,
                norm_eps=config.norm_eps,
                use_fused_ops=config.use_fused_ops,
                use_qk_norm=config.use_qk_norm,
            )
            self.processor = Processor(proc_config)

        self.decoder = Decoder(config)

    def _init_weights(self, initializer_range: float = 0.02, parent_residuals: int = 0):
        """Depth-scaled weight initialization approach (taken from H-Net arxiv).

        Output projections (wo, proj) that add to the residual stream are scaled
        by 1/sqrt(n_residuals) where n_residuals counts total residual-stream
        additions across all stages.
        """
        n_residuals = parent_residuals + self.config.n_compressor_layers * 2 + self.config.n_decoder_layers * 2
        out_std = initializer_range / math.sqrt(n_residuals)
        print(f"[Init] depth={self.depth}: n_residuals={n_residuals} (parent={parent_residuals} + comp={self.config.n_compressor_layers * 2} + dec={self.config.n_decoder_layers * 2})")
        print(f"[Init]   compressor/decoder output proj std={out_std:.6f}, other linear std={initializer_range}")
        print(f"[Init]   pad_dimension (zero-init), router std={initializer_range}")

        for block_list in [self.compressor.layers, self.decoder.layers]:
            for name, m in block_list.named_modules():
                if isinstance(m, nn.Linear):
                    if name.endswith(".wo") or name.endswith(".proj"):
                        nn.init.normal_(m.weight, mean=0.0, std=out_std)
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        for m in self.compressor.router.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if isinstance(self.processor, HierarchicalModel):
            self.processor._init_weights(initializer_range, n_residuals)
        else:
            proc_n_residuals = n_residuals + self.processor.config.n_processor_layers * 2
            proc_out_std = initializer_range / math.sqrt(proc_n_residuals)
            print(f"[Init] depth={self.depth + 1} (processor): n_residuals={proc_n_residuals} (outer={n_residuals} + proc={self.processor.config.n_processor_layers * 2})")
            print(f"[Init]   processor output proj std={proc_out_std:.6f}, other linear std={initializer_range}")
            for name, m in self.processor.named_modules():
                if isinstance(m, nn.Linear):
                    if name.endswith(".wo") or name.endswith(".proj"):
                        nn.init.normal_(m.weight, mean=0.0, std=proc_out_std)
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _apply_lr_multiplier(self, lr_multiplier: List[float]):
        """Tag all parameters at this depth with the corresponding LR multiplier.

        Parameters belonging to deeper stages are first tagged with the current
        stage's multiplier, then overwritten by the recursive call.
        """
        n_outer = sum(p.numel() for p in self.parameters())
        print(f"[LR mult] depth={self.depth}: tagging {n_outer:,} params with lr_multiplier={lr_multiplier[self.depth]}")
        for param in self.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[self.depth])

        if isinstance(self.processor, HierarchicalModel):
            self.processor._apply_lr_multiplier(lr_multiplier)
        else:
            n_proc = sum(p.numel() for p in self.processor.parameters())
            print(f"[LR mult] depth={self.depth + 1} (processor): overwriting {n_proc:,} params with lr_multiplier={lr_multiplier[self.depth + 1]}")
            for param in self.processor.parameters():
                apply_optimization_params(param, lr_multiplier=lr_multiplier[self.depth + 1])

    def setup_cache(self, device=None):
        cos, sin = build_rope_cache(
            self.config.block_size, self.config.rope_n_elem,
            device=device, base=self.config.rope_base,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.processor.setup_cache(device=device)

    def upsample(self, x):
        if self.pad_dimension is not None:
            x = torch.cat(
                (x, self.pad_dimension.expand(x.shape[:-1] + (-1,))),
                dim=-1,
            )
        return x

    def downsample(self, x, target_dim):
        return x[..., :target_dim]

    def forward(self, x, input_ids=None):
        # x: [B, L, D]
        B, L, D = x.shape

        cos = self.cos[:, :L]
        sin = self.sin[:, :L]

        x, x_compressed, boundaries, counts, avg_chunk_size = self.compressor(x, cos, sin, input_ids=input_ids)

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

        return out, stats


class HierarchicalLM(nn.Module):
    # language model wrapper: handles tokenization and vocab projection

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.emb = nn.Embedding(config.padded_vocab_size, config.dim)
        self.model = HierarchicalModel(config)
        self.vocab = nn.Linear(config.dim, config.padded_vocab_size, bias=False)

        if config.use_fused_ops:
            self.fused_linear_cross_entropy = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)

    def _init_weights(self, initializer_range: float = 0.02):
        print(f"[Init] HierarchicalLM: emb std={initializer_range}, vocab std={initializer_range}")
        nn.init.normal_(self.emb.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.vocab.weight, mean=0.0, std=initializer_range)
        self.model._init_weights(initializer_range)

    def apply_lr_multiplier(self, lr_multiplier: List[float]):
        """Apply per-stage LR multipliers. Must be called before creating optimizer param groups."""
        print(f"[LR mult] HierarchicalLM: emb/vocab lr_multiplier={lr_multiplier[0]}")
        for param in self.emb.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[0])
        for param in self.vocab.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[0])
        self.model._apply_lr_multiplier(lr_multiplier)

    def setup_cache(self, device=None):
        self.model.setup_cache(device=device)

    def forward(self, input_ids, labels=None):
        x = self.emb(input_ids)  # [B, L, D]
        out, stats = self.model(x, input_ids=input_ids)  # [B, L, D], dict

        if labels is not None:
            if self.config.use_fused_ops:
                loss = self.fused_linear_cross_entropy(
                    self.vocab.weight, out.view(-1, out.size(-1)), labels.view(-1)
                )
                return loss, stats
            else:
                logits = self.vocab(out)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                return loss, stats

        logits = self.vocab(out)  # [B, L, V]
        return logits, stats


if __name__ == "__main__":

    config = Config()

    model = HierarchicalLM(config)

    print(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.setup_cache(device=device)

    input_ids = torch.randint(0, config.vocab_size, (2, 128), device=device)
    print("input_ids.shape:", input_ids.shape)

    logits, stats = model(input_ids)
    print("logits.shape:", logits.shape)
    print("stats:", stats)