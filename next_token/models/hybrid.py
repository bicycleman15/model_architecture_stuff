"""Hybrid transformer that interleaves attention, GatedDeltaNet, and RNN layers.

Each layer is one of:
  * `TransformerBlock` (workspace [`models/transformer.py`](../../models/transformer.py)) -- pre-norm
    causal attention + pre-norm `LLaMAMLP`.
  * `DeltaNetBlock` (defined here) -- pre-norm `fla.layers.GatedDeltaNet` +
    pre-norm `LLaMAMLP`.
  * `RnnBlock` (defined here) -- pre-norm `M2RNNLayer` (vendored from
    `state_tracking/models/rnn/`) + pre-norm `LLaMAMLP`.

The model shell (embedding -> layers -> final norm -> output) and the
public surface (`forward(input_ids, labels=None)` returning `(loss, stats)`
or `(logits, stats)`, and `setup_cache(device)`) match the workspace
`Transformer`, so the trainer in `next_token/train.py` is unchanged.

Layer pattern: either an explicit list `pattern: [type, type, ...]` that
cycles through the layers (e.g. `[deltanet, attention]`,
`[rnn, attention, deltanet]`), or, when `pattern` is null, strict 2-way
alternation between `attention` and `deltanet` with `pattern_start_with`
selecting layer 0 (legacy default).
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from fla.layers import GatedDeltaNet

# Make sure the workspace root is on sys.path so we can import the vendored
# state_tracking M2RNN layer regardless of whether hybrid.py is loaded
# directly or through next_token.models.
_WS_ROOT_FOR_RNN = pathlib.Path(__file__).resolve().parents[2]
if str(_WS_ROOT_FOR_RNN) not in sys.path:
    sys.path.insert(0, str(_WS_ROOT_FOR_RNN))

from state_tracking.models.rnn import M2RNNLayer  # noqa: E402


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


_VALID_MIXERS = ("deltanet", "attention", "rnn")


@dataclass
class HybridConfig:
    vocab_size: int
    block_size: int
    dim: int = 384
    n_layer: int = 6
    # Per-layer mixer assignment (preferred). Two ways to use it:
    #   1. Length == n_layer  -> used verbatim, layer i is `pattern[i]`.
    #   2. Length <  n_layer  -> cycled: layer i is `pattern[i % len(pattern)]`.
    # When None, falls back to alternating {attention, deltanet} starting
    # with `pattern_start_with`.
    pattern: Optional[list] = None
    # Optional per-index overrides applied AFTER the cycle is materialized.
    # Maps layer index (int, supports negative Python-style indexing) to a
    # mixer type. Use this to pin specific positions, e.g. force the last
    # layer to be `rnn` while the rest follows a short cycle:
    #   pattern: [rnn, attention]
    #   pattern_overrides: {-1: rnn}
    pattern_overrides: Optional[dict] = None
    pattern_start_with: str = "deltanet"  # legacy 2-way default; ignored if pattern is set

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

    # RNN sub-block (M2RNN)
    rnn_num_heads: int = 6
    rnn_key_head_dim: int = 32
    rnn_value_head_dim: int = 32
    rnn_backend: str = "triton"  # or "torch"
    rnn_gradient_clipping: Optional[float] = None

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

        if self.pattern is not None:
            if len(self.pattern) == 0:
                raise ValueError("`pattern` must be a non-empty list of mixer types")
            if len(self.pattern) > self.n_layer:
                raise ValueError(
                    f"`pattern` has length {len(self.pattern)} > n_layer={self.n_layer}; "
                    f"either shorten it or increase n_layer."
                )
            for t in self.pattern:
                if t not in _VALID_MIXERS:
                    raise ValueError(
                        f"`pattern` entries must be one of {_VALID_MIXERS}, got {t!r}"
                    )
        else:
            if self.pattern_start_with not in ("deltanet", "attention"):
                raise ValueError(
                    f"pattern_start_with must be 'deltanet' or 'attention', "
                    f"got {self.pattern_start_with!r}"
                )

        if self.pattern_overrides is not None:
            normalized: dict[int, str] = {}
            for k, v in dict(self.pattern_overrides).items():
                try:
                    idx = int(k)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"`pattern_overrides` keys must be ints (got {k!r})"
                    ) from e
                if v not in _VALID_MIXERS:
                    raise ValueError(
                        f"`pattern_overrides` values must be one of {_VALID_MIXERS}, "
                        f"got {v!r} at index {idx}"
                    )
                if idx < 0:
                    idx = self.n_layer + idx
                if not (0 <= idx < self.n_layer):
                    raise ValueError(
                        f"`pattern_overrides` index {k} out of range for n_layer={self.n_layer}"
                    )
                normalized[idx] = v
            self.pattern_overrides = normalized

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


class RnnBlock(nn.Module):
    """Pre-norm M2RNN mixer + pre-norm LLaMAMLP, mirroring TransformerBlock.

    M2RNN is an autoregressive recurrence and does not use RoPE / causal
    masks, so the cos/sin/mask/input_pos kwargs are accepted for interface
    parity with TransformerBlock and DeltaNetBlock but are ignored.
    """

    def __init__(self, cfg: HybridConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        self.attention_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.rnn = M2RNNLayer(
            input_size=cfg.dim,
            key_head_dim=cfg.rnn_key_head_dim,
            value_head_dim=cfg.rnn_value_head_dim,
            output_size=cfg.dim,
            num_query_heads=cfg.rnn_num_heads,
            num_key_heads=cfg.rnn_num_heads,
            num_value_heads=cfg.rnn_num_heads,
            num_forget_input_heads=cfg.rnn_num_heads,
            num_weight_heads=cfg.rnn_num_heads,
            add_bias=False,
            gradient_clipping=cfg.rnn_gradient_clipping,
            backend=cfg.rnn_backend,
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
        o, _ = self.rnn(self.attention_norm(x))
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
            elif t == "deltanet":
                self.layers.append(DeltaNetBlock(config, layer_idx=i))
            elif t == "rnn":
                self.layers.append(RnnBlock(config, layer_idx=i))
            else:
                raise ValueError(f"unknown layer type {t!r}")

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.padded_vocab_size, bias=False)

        self._init_weights(config.initializer_range)

    def _layer_types(self) -> list[str]:
        if self.config.pattern is not None:
            cycle = list(self.config.pattern)
            types = [cycle[i % len(cycle)] for i in range(self.config.n_layer)]
        else:
            a, b = (
                ("deltanet", "attention")
                if self.config.pattern_start_with == "deltanet"
                else ("attention", "deltanet")
            )
            types = [a if i % 2 == 0 else b for i in range(self.config.n_layer)]

        if self.config.pattern_overrides:
            for idx, t in self.config.pattern_overrides.items():
                types[idx] = t
        return types

    def _init_weights(self, initializer_range: float = 0.02) -> None:
        n_residuals = self.config.n_layer * 2
        out_std = initializer_range / math.sqrt(n_residuals)
        n_attn = sum(1 for t in self.layer_types if t == "attention")
        n_dn = sum(1 for t in self.layer_types if t == "deltanet")
        n_rnn = sum(1 for t in self.layer_types if t == "rnn")
        print(
            f"[Init] HybridModel: layers={self.layer_types} "
            f"(attn={n_attn}, deltanet={n_dn}, rnn={n_rnn}), "
            f"n_residuals={n_residuals}, "
            f"output proj std={out_std:.6f}, other linear std={initializer_range}"
        )

        nn.init.normal_(self.wte.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.output.weight, mean=0.0, std=initializer_range)

        # Residual-output projections per block:
        #   * TransformerBlock: `attention.wo`, `feed_forward.proj`
        #   * DeltaNetBlock:    `gdn.o_proj`,   `feed_forward.proj`
        #   * RnnBlock:         `rnn.output_projection`, `feed_forward.proj`
        # Apply the scaled init to those, and normal init to other Linears in
        # the layer stack. Leave each mixer's own internal init paths intact:
        #   * GatedDeltaNet: A_log, dt_bias, ShortConvolution weights
        #     (initialized by fla).
        #   * M2RNNLayer: state_weight (initialized by M2RNNLayer's
        #     reset_parameters with std=1/sqrt(value_head_dim) for BPTT
        #     stability -- do NOT overwrite).
        scaled_suffixes = (".wo", ".proj", ".o_proj", ".output_projection")
        for name, m in self.layers.named_modules():
            if isinstance(m, nn.Linear):
                if any(name.endswith(s) for s in scaled_suffixes):
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
    pattern_raw = mcfg.get("pattern", None)
    pattern = list(pattern_raw) if pattern_raw is not None else None

    overrides_raw = mcfg.get("pattern_overrides", None)
    pattern_overrides = dict(overrides_raw) if overrides_raw is not None else None

    rnn_cfg = mcfg.get("rnn", None)

    config = HybridConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        dim=mcfg.dim,
        n_layer=mcfg.n_layer,
        pattern=pattern,
        pattern_overrides=pattern_overrides,
        pattern_start_with=mcfg.get("pattern_start_with", "deltanet"),
        n_head=mcfg.attn.n_head,
        use_qk_norm=mcfg.attn.get("use_qk_norm", False),
        dn_num_heads=mcfg.deltanet.num_heads,
        dn_head_dim=mcfg.deltanet.head_dim,
        dn_expand_v=mcfg.deltanet.expand_v,
        dn_use_short_conv=mcfg.deltanet.use_short_conv,
        dn_use_gate=mcfg.deltanet.use_gate,
        dn_conv_size=mcfg.deltanet.get("conv_size", 4),
        rnn_num_heads=rnn_cfg.get("num_heads", 6) if rnn_cfg is not None else 6,
        rnn_key_head_dim=rnn_cfg.get("key_head_dim", 32) if rnn_cfg is not None else 32,
        rnn_value_head_dim=rnn_cfg.get("value_head_dim", 32) if rnn_cfg is not None else 32,
        rnn_backend=rnn_cfg.get("backend", "triton") if rnn_cfg is not None else "triton",
        rnn_gradient_clipping=rnn_cfg.get("gradient_clipping", None) if rnn_cfg is not None else None,
        initializer_range=mcfg.get("initializer_range", 0.02),
        norm_eps=mcfg.get("norm_eps", 1e-5),
        use_fused_ops=mcfg.get("use_fused_ops", False),
        rope_base=mcfg.get("rope_base", 10000),
    )
    return HybridModel(config)
