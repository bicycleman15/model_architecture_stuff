import einops
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from transformer import TransformerBlock, RMSNorm, build_rope_cache, find_multiple


@dataclass
class Config:
    block_size: int = 2048
    vocab_size: int = 32000
    padded_vocab_size: Optional[int] = None

    # shared transformer dimensions
    dim: int = 768
    n_head: int = 12
    n_local_heads: int = -1
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None

    # per-stage layer counts
    n_compressor_layers: int = 4
    n_processor_layers: int = 8
    n_decoder_layers: int = 4

    # compressor
    compress_threshold: float = 0.5

    # processor: None = flat transformer blocks, or a Config for recursive nesting
    processor_config: Optional['Config'] = None

    # rope / norm
    rope_base: float = 10000
    norm_eps: float = 1e-5
    rope_n_elem: Optional[int] = None

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)

        assert self.dim % self.n_head == 0
        self.head_dim = self.dim // self.n_head

        self.padded_vocab_size = find_multiple(self.vocab_size, 256)
        self.rope_n_elem = self.head_dim


class Compressor(nn.Module):
    # compressor "implicitly" decides how to chunk up the sequence
    # it acts as a router
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.threshold = config.compress_threshold

        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_compressor_layers)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.should_chunk_mlp = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, 1),
        )

    def chunk_and_compress(self, x):
        # x: [B, L, D]

        probs = torch.sigmoid(self.should_chunk_mlp(x).squeeze(-1))  # [B, L]

        boundaries = torch.where(probs > self.threshold, dim=-1)  # [B, K]

        # "compress" by picking up representations at boundary positions
        compressed_x = x[boundaries]  # [B, K, D] hopefully!

        return compressed_x, boundaries, probs

    def forward(self, x, cos, sin):
        # x: [B, L, D]

        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)

        compressed_x, boundaries, probs = self.chunk_and_compress(x)

        return x, compressed_x, boundaries
        

class Decoder(nn.Module):
    # decoder decodes compressed chunks back to tokens
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_decoder_layers)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x_processed, boundaries, x_residual, cos, sin, seq_len):
        # x_processed: [B, K, D]
        # boundaries: [B, K] — positions in [0, seq_len) where we placed boundaries
        # x_residual: [B, L, D] — full-length output from compressor
        B, K, D = x_processed.shape

        # expand each compressed token back to its chunk of the original sequence
        # token 0 covers [0, boundaries[1])
        # token i covers [boundaries[i], boundaries[i+1])
        # token K-1 covers [boundaries[K-1], seq_len)
        batch_expanded = []
        for b in range(B):
            chunks = []
            for i in range(K):
                start = 0 if i == 0 else boundaries[b, i].item()
                end = seq_len if i == K - 1 else boundaries[b, i + 1].item()
                chunk_size = end - start
                chunks.append(einops.repeat(x_processed[b, i], 'd -> c d', c=chunk_size))
            batch_expanded.append(torch.cat(chunks, dim=0))  # [L, D]
        x = torch.stack(batch_expanded, dim=0)  # [B, L, D]

        # the expanded compressed tokens (i.e. `x` above) are identical within each chunk
        # so the decoder would have no way to know what the previous token was
        # decoder needs this since there can stochasty in decoding, and it must know what token
        # before is to accurately decode the current token
        # Adding the compressor's full-length output as a residual
        # provides this info about previous token without requiring input_ids
        # everything in compressor/processor/decoder should operate purely on embeddings only
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
            TransformerBlock(config) for _ in range(config.n_processor_layers)
        )
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
    # forward(x) -> x, making it usable as a processor in a parent HierarchicalModel

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.compressor = Compressor(config)
        if config.processor_config is not None:
            self.processor = HierarchicalModel(config.processor_config)
        else:
            self.processor = Processor(config)
        self.decoder = Decoder(config)

    def setup_cache(self, device=None):
        cos, sin = build_rope_cache(
            self.config.block_size, self.config.rope_n_elem,
            device=device, base=self.config.rope_base,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.processor.setup_cache(device=device)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape

        cos = self.cos[:, :L]
        sin = self.sin[:, :L]

        x, x_compressed, boundaries = self.compressor(x, cos, sin)

        x_processed = self.processor(x_compressed)

        out = self.decoder(x_processed, boundaries, x, cos, sin, L)

        return out


class HierarchicalLM(nn.Module):
    # language model wrapper: handles tokenization and vocab projection

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.emb = nn.Embedding(config.padded_vocab_size, config.dim)
        self.model = HierarchicalModel(config)
        self.vocab = nn.Linear(config.dim, config.padded_vocab_size, bias=False)

    def setup_cache(self, device=None):
        self.model.setup_cache(device=device)

    def forward(self, input_ids):
        x = self.emb(input_ids)  # [B, L, D]
        out = self.model(x)  # [B, L, D]
        logits = self.vocab(out)  # [B, L, V]
        return logits