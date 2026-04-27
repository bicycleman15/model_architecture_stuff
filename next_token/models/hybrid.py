"""Hybrid transformer that interleaves attention and GatedDeltaNet layers.

Each layer is one of:
  * `TransformerBlock` (workspace [`models/transformer.py`](../../models/transformer.py)) -- pre-norm
    causal attention + pre-norm `LLaMAMLP`.
  * `DeltaNetBlock` (defined here) -- pre-norm `fla.layers.GatedDeltaNet` +
    pre-norm `LLaMAMLP`.

The model shell (embedding -> layers -> final norm -> output) and the
public surface (`forward(input_ids, labels=None)` returning `(loss, stats)`
or `(logits, stats)`, and `setup_cache(device)`) match the workspace
`Transformer`, so the trainer in `next_token/train.py` is unchanged.

Layer pattern is strict alternation; `pattern_start_with` selects whether
layer 0 is a `DeltaNetBlock` or a `TransformerBlock`.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from fla.layers import GatedDeltaNet


_WS_ROOT = pathlib.Path(__file__).resolve().parents[2]
_WS_TRANSFORMER = _WS_ROOT / "models" / "transformer.py"


def _load_ws_transformer_module():
    spec = importlib.util.spec_from_file_location("_ws_root_transformer_hybrid", _WS_TRANSFORMER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ws = _load_ws_transformer_module()
TransformerConfig = _ws.TransformerConfig
TransformerBlock = _ws.TransformerBlock
RMSNorm = _ws.RMSNorm
LLaMAMLP = _ws.LLaMAMLP
build_rope_cache = _ws.build_rope_cache
find_multiple = _ws.find_multiple


@dataclass
class _DeltaNetCfg:
    num_heads: int
    head_dim: int
    expand_v: float
    use_short_conv: bool
    use_gate: bool
    conv_size: int


@dataclass
class HybridConfig:
    vocab_size: int
    block_size: int
    dim: int = 384
    n_layer: int = 6
    pattern_start_with: str = "deltanet"  # or "attention"

    # Attention sub-block (TransformerBlock)
    n_head: int = 6
    use_qk_norm: bool = False

    # DeltaNet sub-block (GatedDeltaNet)
    dn_num_heads: int = 6
    dn_head_dim: int = 48
    dn_expand_v: float = 2.0
    dn_use_short_conv: bool = True
    dn_use_gate: bool = True
    dn_conv_size: int = 4

    # Shared
    initializer_range: float = 0.02
    norm_eps: float = 1e-5
    use_fused_ops: bool = False
    rope_base: float = 10000

    # Filled in __post_init__:
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None
    padded_vocab_size: Optional[int] = None
    rope_n_elem: Optional[int] = None

    def __post_init__(self):
        assert self.dim % self.n_head == 0, (
            f"dim={self.dim} must be divisible by n_head={self.n_head}"
        )
        self.head_dim = self.dim // self.n_head
        hidden_dim = 4 * self.dim
        n_hidden = int(2 * hidden_dim / 3)
        self.intermediate_size = find_multiple(n_hidden, 256)
        self.padded_vocab_size = find_multiple(self.vocab_size, 256)
        self.rope_n_elem = self.head_dim

        if self.pattern_start_with not in ("deltanet", "attention"):
            raise ValueError(
                f"pattern_start_with must be 'deltanet' or 'attention', got {self.pattern_start_with!r}"
            )

        # Sanity check for GatedDeltaNet sizing constraints.
        if self.dn_use_gate:
            # GatedDeltaNet docstring: num_heads * head_dim == 0.75 * hidden_size
            expected = 0.75 * self.dim
            actual = self.dn_num_heads * self.dn_head_dim
            if not math.isclose(actual, expected, rel_tol=1e-5):
                raise ValueError(
                    f"GatedDeltaNet with use_gate=True requires "
                    f"dn_num_heads * dn_head_dim == 0.75 * dim. "
                    f"Got {self.dn_num_heads} * {self.dn_head_dim} = {actual}, "
                    f"expected {expected} (dim={self.dim})."
                )


def _attn_subconfig(cfg: HybridConfig) -> TransformerConfig:
    """Build the workspace TransformerConfig used by TransformerBlock layers."""
    tcfg = TransformerConfig(
        block_size=cfg.block_size,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        dim=cfg.dim,
        rope_base=cfg.rope_base,
        norm_eps=cfg.norm_eps,
        initializer_range=cfg.initializer_range,
        use_fused_ops=cfg.use_fused_ops,
        use_qk_norm=cfg.use_qk_norm,
    )
    return tcfg


def _ffn_cfg(cfg: HybridConfig) -> SimpleNamespace:
    """Minimal config namespace LLaMAMLP needs (`dim`, `intermediate_size`)."""
    return SimpleNamespace(dim=cfg.dim, intermediate_size=cfg.intermediate_size)


class DeltaNetBlock(nn.Module):
    """Pre-norm GatedDeltaNet + pre-norm LLaMAMLP, mirroring TransformerBlock."""

    def __init__(self, cfg: HybridConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        self.attention_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.gdn = GatedDeltaNet(
            hidden_size=cfg.dim,
            num_heads=cfg.dn_num_heads,
            head_dim=cfg.dn_head_dim,
            expand_v=cfg.dn_expand_v,
            use_short_conv=cfg.dn_use_short_conv,
            use_gate=cfg.dn_use_gate,
            conv_size=cfg.dn_conv_size,
            norm_eps=cfg.norm_eps,
            layer_idx=layer_idx,
        )
        self.ffn_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.feed_forward = LLaMAMLP(_ffn_cfg(cfg))

    def forward(
        self,
        x: Tensor,
        cos: Optional[Tensor] = None,
        sin: Optional[Tensor] = None,
        is_causal: Optional[bool] = True,
        mask=None,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        # GatedDeltaNet returns (output, attn_weights, past_kv); we only need output.
        o, _, _ = self.gdn(self.attention_norm(x))
        h = x + o
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class HybridModel(nn.Module):
    """Stack of alternating TransformerBlock / DeltaNetBlock layers."""

    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        self.config = config
        self._attn_subcfg = _attn_subconfig(config)

        self.wte = nn.Embedding(config.padded_vocab_size, config.dim)

        layer_types = self._layer_types()
        self.layer_types = layer_types
        self.layers = nn.ModuleList()
        for i, t in enumerate(layer_types):
            if t == "attention":
                self.layers.append(TransformerBlock(self._attn_subcfg, layer_idx=i))
            else:
                self.layers.append(DeltaNetBlock(config, layer_idx=i))

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.padded_vocab_size, bias=False)

        self._init_weights(config.initializer_range)

    def _layer_types(self) -> list[str]:
        a, b = (
            ("deltanet", "attention")
            if self.config.pattern_start_with == "deltanet"
            else ("attention", "deltanet")
        )
        return [a if i % 2 == 0 else b for i in range(self.config.n_layer)]

    def _init_weights(self, initializer_range: float = 0.02) -> None:
        n_residuals = self.config.n_layer * 2
        out_std = initializer_range / math.sqrt(n_residuals)
        n_attn = sum(1 for t in self.layer_types if t == "attention")
        n_dn = self.config.n_layer - n_attn
        print(
            f"[Init] HybridModel: layers={self.layer_types} "
            f"(attn={n_attn}, deltanet={n_dn}), n_residuals={n_residuals}, "
            f"output proj std={out_std:.6f}, other linear std={initializer_range}"
        )

        nn.init.normal_(self.wte.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.output.weight, mean=0.0, std=initializer_range)

        # Residual-output projections in TransformerBlock are `attention.wo`
        # and `feed_forward.proj`; in DeltaNetBlock they are `gdn.o_proj`
        # and `feed_forward.proj`. Apply the scaled init to those, normal init
        # to other Linear layers in the layer stack. Leave GatedDeltaNet's own
        # internal Parameters (A_log, dt_bias, ShortConvolution weights)
        # untouched -- those are initialized by fla itself.
        for name, m in self.layers.named_modules():
            if isinstance(m, nn.Linear):
                if name.endswith(".wo") or name.endswith(".proj") or name.endswith(".o_proj"):
                    nn.init.normal_(m.weight, mean=0.0, std=out_std)
                else:
                    nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def setup_cache(self, device=None) -> None:
        cos, sin = build_rope_cache(
            self.config.block_size,
            self.config.rope_n_elem,
            device=device,
            base=self.config.rope_base,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        print("created cos and sin cache (hybrid) ...")
        print("cos shape:", self.cos.shape)
        print("cos dtype:", self.cos.dtype)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        input_pos: Optional[Tensor] = None,
        mask=None,
        **kwargs,
    ):
        bsz, seqlen = input_ids.shape
        stats: dict = {}

        if input_pos is not None:
            cos = self.cos[:, input_pos]
            sin = self.sin[:, input_pos]
        else:
            cos = self.cos[:, :seqlen]
            sin = self.sin[:, :seqlen]

        x = self.wte(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin, mask=mask, input_pos=input_pos)

        x = self.norm(x)

        if labels is not None:
            logits = self.output(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return loss, stats

        logits = self.output(x)
        return logits, stats


def build_hybrid(vocab_size: int, block_size: int, mcfg) -> HybridModel:
    config = HybridConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        dim=mcfg.dim,
        n_layer=mcfg.n_layer,
        pattern_start_with=mcfg.get("pattern_start_with", "deltanet"),
        n_head=mcfg.attn.n_head,
        use_qk_norm=mcfg.attn.get("use_qk_norm", False),
        dn_num_heads=mcfg.deltanet.num_heads,
        dn_head_dim=mcfg.deltanet.head_dim,
        dn_expand_v=mcfg.deltanet.expand_v,
        dn_use_short_conv=mcfg.deltanet.use_short_conv,
        dn_use_gate=mcfg.deltanet.use_gate,
        dn_conv_size=mcfg.deltanet.get("conv_size", 4),
        initializer_range=mcfg.get("initializer_range", 0.02),
        norm_eps=mcfg.get("norm_eps", 1e-5),
        use_fused_ops=mcfg.get("use_fused_ops", False),
        rope_base=mcfg.get("rope_base", 10000),
    )
    return HybridModel(config)
