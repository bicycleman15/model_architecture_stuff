"""MTP (Multi-Token Prediction) auxiliary objective wrapper.

Per-position autoregressive head: for each base position `t`, a small causal
transformer takes ``[h_t, emb(x_{t+1}), ..., emb(x_{t+H-1})]`` and predicts
``x_{t+1}, x_{t+2}, ..., x_{t+H}`` in parallel via causal self-attention.
The head is run as a single batched call at sequence length ``H`` over a
flattened ``(B*T)`` batch, so the extra cost is roughly ``n_layer * H``
block-token applications per base token.

Inference is unchanged: when ``labels is None``, the wrapper delegates straight
to the wrapped model. Training adds the small transformer head (and optionally
its own ``wte`` / ``lm_head``) and mixes the auxiliary cross-entropy into the
total loss as

    L_total = L_next + lambda_mtp * L_mtp

Targets for the head are sourced from the dataset's ``labels`` tensor, NOT
from ``input_ids``. This matters under teacherless training where
``input_ids[prefix_len:]`` is the dummy ``$`` token while ``labels`` always
holds the true target tokens.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import warnings
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


_WS_ROOT = pathlib.Path(__file__).resolve().parents[1]
_WS_TRANSFORMER = _WS_ROOT / "models" / "transformer.py"


def _load_ws_transformer_module():
    spec = importlib.util.spec_from_file_location("_ws_root_transformer_mtp", _WS_TRANSFORMER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ws = _load_ws_transformer_module()
TransformerConfig = _ws.TransformerConfig
TransformerBlock = _ws.TransformerBlock
RMSNorm = _ws.RMSNorm
LigerRMSNorm = _ws.LigerRMSNorm
build_rope_cache = _ws.build_rope_cache


class MTPHead(nn.Module):
    """Small causal transformer that runs at sequence length ``horizon``.

    Inputs/outputs are flattened across ``(B, T)`` so the head sees a batch of
    ``(B*T, H, D)`` micro-sequences in a single SDPA call. Token embeddings
    and the lm-head live on the wrapper -- this module only owns the stack of
    transformer blocks, the final RMSNorm, and the RoPE cache for length
    ``horizon``.
    """

    def __init__(
        self,
        dim: int,
        n_layer: int,
        n_head: int,
        horizon: int,
        norm_eps: float = 1e-5,
        rope_base: float = 10000.0,
        use_qk_norm: bool = False,
        use_fused_ops: bool = False,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        if dim % n_head != 0:
            raise ValueError(f"MTPHead: dim={dim} must be divisible by n_head={n_head}")

        tcfg = TransformerConfig(
            block_size=horizon,
            vocab_size=1,  # unused; the head doesn't own wte/lm-head
            n_layer=n_layer,
            n_head=n_head,
            dim=dim,
            rope_base=rope_base,
            norm_eps=norm_eps,
            initializer_range=initializer_range,
            use_fused_ops=use_fused_ops,
            use_qk_norm=use_qk_norm,
        )
        self.config = tcfg
        self.horizon = int(horizon)

        self.blocks = nn.ModuleList(
            TransformerBlock(tcfg, layer_idx=i) for i in range(n_layer)
        )
        # Match the base Transformer's convention: LigerRMSNorm under
        # use_fused_ops, plain RMSNorm otherwise. The workspace RMSNorm forces
        # fp32 internally; LigerRMSNorm is a fused triton kernel.
        if use_fused_ops:
            self.norm = LigerRMSNorm(dim, eps=norm_eps)
        else:
            self.norm = RMSNorm(dim, eps=norm_eps)

        self._init_weights(initializer_range)

    def _init_weights(self, initializer_range: float) -> None:
        # Match the workspace Transformer init: scaled init for residual-output
        # projections (.wo, .proj), normal init for the rest. Embedding /
        # lm-head are owned by the wrapper, not here.
        n_residuals = self.config.n_layer * 2
        out_std = initializer_range / math.sqrt(n_residuals)
        for name, m in self.blocks.named_modules():
            if isinstance(m, nn.Linear):
                if name.endswith(".wo") or name.endswith(".proj"):
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

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, H, D) -- flattened across (B, T).
        if not hasattr(self, "cos"):
            raise RuntimeError(
                "MTPHead: RoPE cache not set. Call `setup_cache(device)` after "
                "moving the wrapper to its device."
            )
        seqlen = x.size(1)
        cos = self.cos[:, :seqlen]
        sin = self.sin[:, :seqlen]
        for block in self.blocks:
            x = block(x, cos, sin, is_causal=True, mask=None, input_pos=None)
        x = self.norm(x)
        return x


class MTPWrapper(nn.Module):
    """Wraps a base model and adds the MTP auxiliary loss during training.

    Args:
        model: base model with ``forward(input_ids, labels=None) -> (logits, stats)``
            and a final RMSNorm exposed as ``model.norm`` whose output is the
            pre-logits hidden state fed to ``model.output``. When ``tie_wte``
            is set, the wrapped model must also expose ``model.wte``.
        dim: hidden-state dimension of the base model.
        vocab_size: tokenizer's actual (un-padded) vocabulary size; kept on
            the wrapper for diagnostics only.
        padded_vocab_size: vocab dim used by the base model's ``wte`` /
            ``output`` Linear; matters for the untied head_wte / head_lm so
            their shapes match.
        horizon: number of depths the head predicts per base position. The
            small transformer's micro-sequence has length ``horizon`` and
            covers depths ``1..horizon`` (the next ``horizon`` tokens after
            position ``t``).
        n_layer, n_head: shape of the small transformer.
        lambda_mtp: coefficient on ``L_mtp`` in the total loss.
        skip_depth_1: if True, mask out depth-1 (k=0) targets so they don't
            enter the loss (the base lm-head already computes that
            prediction). The depth-1 diagnostic is still reported.
        tie_wte: if True, share input embeddings with ``model.wte``.
        tie_lm_head: if True, share the lm-head with ``model.output``.
    """

    def __init__(
        self,
        model: nn.Module,
        dim: int,
        vocab_size: int,
        padded_vocab_size: int,
        horizon: int,
        n_layer: int = 2,
        n_head: int = 6,
        lambda_mtp: float = 1.0,
        skip_depth_1: bool = True,
        tie_wte: bool = True,
        tie_lm_head: bool = True,
        use_qk_norm: bool = False,
        use_fused_ops: bool = False,
        norm_eps: float = 1e-5,
        rope_base: float = 10000.0,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        if not hasattr(model, "norm") or not hasattr(model, "output"):
            raise AttributeError(
                "MTPWrapper requires the wrapped model to expose `model.norm` "
                "(final RMSNorm) and `model.output` (lm-head)."
            )
        if tie_wte and not hasattr(model, "wte"):
            raise AttributeError(
                "MTPWrapper(tie_wte=True) requires the wrapped model to expose "
                "`model.wte` (input embedding)."
            )
        if int(horizon) < 1:
            raise ValueError(f"MTPWrapper: horizon must be >= 1, got {horizon}")

        self.model = model
        self.horizon = int(horizon)
        self.lambda_mtp = float(lambda_mtp)
        self.skip_depth_1 = bool(skip_depth_1)
        self.tie_wte = bool(tie_wte)
        self.tie_lm_head = bool(tie_lm_head)
        self.use_fused_ops = bool(use_fused_ops)
        self.dim = int(dim)
        self.vocab_size = int(vocab_size)
        self.padded_vocab_size = int(padded_vocab_size)

        self.head = MTPHead(
            dim=dim,
            n_layer=n_layer,
            n_head=n_head,
            horizon=self.horizon,
            norm_eps=norm_eps,
            rope_base=rope_base,
            use_qk_norm=use_qk_norm,
            use_fused_ops=self.use_fused_ops,
            initializer_range=initializer_range,
        )

        # Untied weight modules; tied paths read straight off the base model.
        if self.tie_wte:
            self.head_wte = None
        else:
            self.head_wte = nn.Embedding(padded_vocab_size, dim)
            nn.init.normal_(self.head_wte.weight, mean=0.0, std=initializer_range)

        if self.tie_lm_head:
            self.head_lm = None
        else:
            self.head_lm = nn.Linear(dim, padded_vocab_size, bias=False)
            nn.init.normal_(self.head_lm.weight, mean=0.0, std=initializer_range)

        # NOTE: we deliberately do NOT use LigerFusedLinearCrossEntropyLoss
        # for the head's lm-head + CE. Mixing the fused linear+CE kernel into
        # an auxiliary loss path can interfere with gradient routing through
        # the (potentially tied) lm-head weight, so we keep that step as a
        # plain matmul + F.cross_entropy. Liger ops INSIDE the head's
        # TransformerBlocks (MLP, RMSNorm, RoPE) are still used.

        self._hidden_buf: Optional[Tensor] = None

        # WARNING: MTPWrapper assumes the wrapped model exposes a single final
        # RMSNorm at `model.norm` whose output is the post-final-norm /
        # pre-logits hidden state fed directly to `model.output` (the lm_head).
        # If you rename `.norm`, run it more than once per forward, or move the
        # lm-head input elsewhere, this hook will silently capture the wrong
        # tensor (or nothing). Update `_capture` / this registration if the
        # base model definition changes.
        warnings.warn(
            "MTPWrapper: capturing hidden states via forward-hook on "
            "`model.norm` (assumed to be the final RMSNorm whose output is the "
            "pre-logits input to `model.output`). If the base model definition "
            "changes, update next_token/mtp.py accordingly.",
            stacklevel=2,
        )
        model.norm.register_forward_hook(self._capture)

    # ---- internals -----------------------------------------------------

    def _capture(self, module: nn.Module, inp: Any, out: Tensor) -> None:
        self._hidden_buf = out

    def _wte(self) -> nn.Module:
        return self.head_wte if self.head_wte is not None else self.model.wte

    def _lm(self) -> nn.Module:
        return self.head_lm if self.head_lm is not None else self.model.output

    # ---- delegation ----------------------------------------------------

    def setup_cache(self, device=None):
        self.model.setup_cache(device=device)
        self.head.setup_cache(device=device)

    def setup_kv_cache(self, *args, **kwargs):
        return self.model.setup_kv_cache(*args, **kwargs)

    # ---- forward -------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # Eval / generation: untouched fast path.
        if labels is None:
            return self.model(input_ids, labels=None, **kwargs)

        # 1) Run the base model with labels=None so we get logits_post and can
        #    compute L_next ourselves with a plain F.cross_entropy. Calling
        #    with labels=labels would route through the base's
        #    LigerFusedLinearCrossEntropyLoss when use_fused_ops is on, which
        #    can interfere with gradient routing through the (possibly tied)
        #    lm-head weight; we explicitly avoid that here.
        self._hidden_buf = None
        logits_post, stats = self.model(input_ids, labels=None, **kwargs)
        h = self._hidden_buf
        if h is None:
            raise RuntimeError(
                "MTPWrapper: forward hook on `model.norm` did not fire. "
                "Has the base model definition changed?"
            )

        loss_next = F.cross_entropy(
            logits_post.reshape(-1, logits_post.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        B, T, D = h.shape
        H = self.horizon
        V = self.padded_vocab_size

        # 2) Build head_input: (B, T, H, D)
        #    head_input[:, t, 0, :] = h_t
        #    head_input[:, t, k, :] = wte(input_ids[:, t+k]) for k = 1..H-1
        #    Right-edge positions where t+k >= T are zero-padded; their
        #    targets are -100 below so they contribute nothing to the loss.
        emb_x = self._wte()(input_ids)                                  # (B, T, D)
        if H > 1:
            emb_pad = F.pad(emb_x, (0, 0, 0, H - 1))                    # (B, T+H-1, D)
            chunks = [h] + [emb_pad[:, k : k + T, :] for k in range(1, H)]
            head_input = torch.stack(chunks, dim=2)                     # (B, T, H, D)
        else:
            head_input = h.unsqueeze(2)                                 # (B, T, 1, D)
        head_input_flat = head_input.reshape(B * T, H, D)

        # 3) Small transformer + plain lm-head -> per-(t, depth) logits.
        head_out = self.head(head_input_flat)                           # (B*T, H, D)
        head_logits = self._lm()(head_out).view(B, T, H, V)

        # 4) Build target_ids from labels (the ground-truth next-token tensor;
        #    safe under teacherless since input_ids contains $ dummies there).
        labels_pad = F.pad(labels, (0, H - 1), value=-100)
        target_ids_raw = torch.stack(
            [labels_pad[:, k : k + T] for k in range(H)], dim=2
        )                                                                # (B, T, H)

        if self.skip_depth_1:
            target_ids = target_ids_raw.clone()
            target_ids[:, :, 0] = -100
        else:
            target_ids = target_ids_raw

        loss_mtp = F.cross_entropy(
            head_logits.reshape(-1, V),
            target_ids.reshape(-1),
            ignore_index=-100,
        )

        total = loss_next + self.lambda_mtp * loss_mtp

        stats = dict(stats) if stats else {}
        stats["mtp/loss_next"] = float(loss_next.detach())
        stats["mtp/loss_mtp"] = float(loss_mtp.detach())
        stats["mtp/loss_total"] = float(total.detach())
        # for k in range(H):
        #     stats[f"mtp/loss_d{k + 1}"] = float(per_depth_losses[k])
        return total, stats


def build_mtp(
    model: nn.Module,
    vocab_size: int,
    padded_vocab_size: int,
    cfg_mtp,
    horizon: int,
) -> MTPWrapper:
    """Construct an `MTPWrapper` around `model` from a `cfg.mtp` block.

    `vocab_size` is the tokenizer's actual vocab (kept for diagnostics).
    `padded_vocab_size` must match the base model's lm-head / wte width so
    the untied head_wte / head_lm can be constructed with the right shape.
    `dim`, `norm_eps`, `rope_base`, `initializer_range` are read off
    ``model.config`` so the head matches the base model's residual stream.
    """
    if not hasattr(model, "config") or not hasattr(model.config, "dim"):
        raise AttributeError(
            "build_mtp: wrapped model must expose `model.config.dim`."
        )
    dim = int(model.config.dim)
    norm_eps = float(getattr(model.config, "norm_eps", 1e-5))
    rope_base = float(getattr(model.config, "rope_base", 10000.0))
    initializer_range = float(getattr(model.config, "initializer_range", 0.02))

    # Default to the base model's fused-ops setting when the cfg leaves it
    # null. The base ships with use_fused_ops=true so the head follows suit
    # unless the user opts out explicitly.
    base_fused = bool(getattr(model.config, "use_fused_ops", False))
    fused_raw = cfg_mtp.get("use_fused_ops", None)
    use_fused_ops = base_fused if fused_raw is None else bool(fused_raw)

    return MTPWrapper(
        model=model,
        dim=dim,
        vocab_size=int(vocab_size),
        padded_vocab_size=int(padded_vocab_size),
        horizon=int(horizon),
        n_layer=int(cfg_mtp.get("n_layer", 2)),
        n_head=int(cfg_mtp.get("n_head", 6)),
        lambda_mtp=float(cfg_mtp.get("lambda_mtp", 1.0)),
        skip_depth_1=bool(cfg_mtp.get("skip_depth_1", True)),
        tie_wte=bool(cfg_mtp.get("tie_wte", True)),
        tie_lm_head=bool(cfg_mtp.get("tie_lm_head", True)),
        use_qk_norm=bool(cfg_mtp.get("use_qk_norm", False)),
        use_fused_ops=use_fused_ops,
        norm_eps=norm_eps,
        rope_base=rope_base,
        initializer_range=initializer_range,
    )
