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
    dim: int = 512
    n_head: int = 8
    n_local_heads: int = -1
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None

    # per-stage layer counts
    n_compressor_layers: int = 3
    n_processor_layers: int = 6
    n_decoder_layers: int = 3

    # compressor
    gumbel_tau: float = 1.0
    max_chunk_size: int = 16

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
        self.tau = config.gumbel_tau

        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_compressor_layers)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.should_chunk_mlp = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, 1),
        )

    def gumbel_sigmoid(self, logits):
        # Gumbel-Sigmoid (Binary Concrete) with straight-through estimator
        if self.training:
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            gumbel_noise = torch.log(u) - torch.log(1 - u)
            soft = torch.sigmoid((logits + gumbel_noise) / self.tau)
        else:
            soft = torch.sigmoid(logits)

        hard = (soft > 0.5).float()
        # straight-through: hard in forward, soft in backward
        return hard - soft.detach() + soft

    def _enforce_max_chunk_size(self, sampled_idx, seq_len):
        """Insert forced boundaries so no decoder chunk exceeds max_chunk_size.

        The decoder assigns chunks as:
          token 0:     [0, boundaries[0])                    size = boundaries[0]
          token i:     [boundaries[i-1], boundaries[i])      size = boundaries[i] - boundaries[i-1]
          token K-1:   [boundaries[K-2], seq_len)            size = seq_len - boundaries[K-2]
        """
        M = self.config.max_chunk_size
        result = list(sampled_idx.tolist())

        # guarantee at least one boundary so the loop has something to work with
        if not result:
            result.append(min(M, seq_len - 1))

        # iteratively insert forced boundaries until all chunks are within M
        changed = True
        while changed:
            changed = False
            result = sorted(set(result))
            K = len(result)
            insertions = []
            for i in range(K):
                start = 0 if i == 0 else result[i - 1]
                end = result[i] if i < K - 1 else seq_len
                chunk_size = end - start
                if chunk_size > M:
                    insertions.append(start + M)
                    changed = True
            result.extend(insertions)

        result = sorted(set(result))
        return torch.tensor(result, dtype=torch.long, device=sampled_idx.device)

    def chunk_and_compress(self, x):
        # x: [B, L, D]
        B, L, D = x.shape

        logits = self.should_chunk_mlp(x).squeeze(-1)  # [B, L]
        gate = self.gumbel_sigmoid(logits)  # [B, L], 0/1 forward, differentiable backward

        # gate the input so selected positions carry STE gradients
        gated_x = x * gate.unsqueeze(-1)  # [B, L, D]

        mask = gate > 0.5  # [B, L] hard boolean for indexing

        # per-batch: collect sampled boundaries, enforce max chunk size, gather tokens
        all_idx = []
        for b in range(B):
            sampled_idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            idx = self._enforce_max_chunk_size(sampled_idx, L)
            all_idx.append(idx)

        K = max(idx.shape[0] for idx in all_idx)
        counts = torch.zeros(B, dtype=torch.long, device=x.device)
        boundaries = torch.zeros(B, K, dtype=torch.long, device=x.device)
        compressed_x = torch.zeros(B, K, D, device=x.device, dtype=x.dtype)

        for b in range(B):
            idx = all_idx[b]
            n = idx.shape[0]
            counts[b] = n
            boundaries[b, :n] = idx
            # forced boundaries use x directly (no STE gate); sampled ones use gated_x
            is_sampled = mask[b][idx]
            compressed_x[b, :n] = torch.where(
                is_sampled.unsqueeze(-1), gated_x[b, idx], x[b, idx]
            )
            if n < K:
                boundaries[b, n:] = L

        return compressed_x, boundaries, gate, counts

    def forward(self, x, cos, sin):
        # x: [B, L, D]

        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)

        compressed_x, boundaries, probs, counts = self.chunk_and_compress(x)

        return x, compressed_x, boundaries, counts
        

class Decoder(nn.Module):
    # decoder decodes compressed chunks back to tokens
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_decoder_layers)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x_processed, boundaries, counts, x_residual, cos, sin, seq_len):
        # x_processed: [B, K, D]
        # boundaries: [B, K] — positions in [0, seq_len), padded with seq_len for invalid slots
        # counts: [B] — actual number of valid boundaries per batch element
        # x_residual: [B, L, D] — full-length output from compressor
        B, K, D = x_processed.shape

        # expand each compressed token back to its chunk of the original sequence
        # token i covers [boundaries[i-1], boundaries[i])  (boundary = chunk end)
        # token 0 covers [0, boundaries[0])
        # token K-1 covers [boundaries[K-2], seq_len)  (absorbs the tail)
        batch_expanded = []
        for b in range(B):
            n = counts[b].item()
            chunks = []
            for i in range(n):
                start = 0 if i == 0 else boundaries[b, i - 1].item()
                end = boundaries[b, i].item() if i < n - 1 else seq_len
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

        x, x_compressed, boundaries, counts = self.compressor(x, cos, sin)

        x_processed = self.processor(x_compressed)

        out = self.decoder(x_processed, boundaries, counts, x, cos, sin, L)

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


if __name__ == "__main__":

    config = Config()

    model = HierarchicalLM(config)

    print(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.setup_cache(device=device)

    input_ids = torch.randint(0, config.vocab_size, (2, 128), device=device)
    print("input_ids.shape:", input_ids.shape)

    logits = model(input_ids)
    print("logits.shape:", logits.shape)