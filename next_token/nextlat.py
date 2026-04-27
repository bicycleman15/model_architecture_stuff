"""NextLat auxiliary objective wrapper.

Implements the training objective from
[Joint Training of Latent World Models for Transformers](https://arxiv.org/abs/2511.05963)
as a thin wrapper around any next-token model that exposes the workspace
contract `forward(input_ids, labels=None, **kwargs) -> (loss_or_logits, stats)`
and a final RMSNorm at `self.norm` whose output is the pre-logits hidden
state fed to `self.output`.

Inference is unchanged: when `labels is None`, the wrapper delegates straight
to the wrapped model. Training adds a residual MLP latent-dynamics model and
two extra losses (smooth-L1 next-h regression + KL divergence against a
detached lm-head), aggregated as

    L_total = L_next + lambda_h * L_next_h + lambda_kl * L_kl

following Algorithm 1 of the paper.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class LatentDynamics(nn.Module):
    """Residual MLP latent dynamics model `p_psi(h, x) -> h_next`.

    The action embedding maps the next-token id to a `dim`-vector; this is
    concatenated with the current hidden state and fed through a small SiLU
    MLP whose output is added back to the input hidden state (residual).
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        n_hidden_layers: int = 2,
        hidden_mult: int = 4,
    ) -> None:
        super().__init__()
        self.action_emb = nn.Embedding(vocab_size, dim)
        layers: list[nn.Module] = []
        in_dim = 2 * dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_mult * dim))
            layers.append(nn.SiLU())
            in_dim = hidden_mult * dim
        layers.append(nn.Linear(in_dim, dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, h: Tensor, tokens: Tensor) -> Tensor:
        a = self.action_emb(tokens)
        return h + self.mlp(torch.cat([h, a], dim=-1))


class NextLatWrapper(nn.Module):
    """Wraps a base model and adds the NextLat auxiliary loss during training.

    Args:
        model: base model with `forward(input_ids, labels=None) -> (logits, stats)`
            and a final RMSNorm exposed as `model.norm` whose output is the
            pre-logits hidden state.
        dim: hidden-state dimension.
        vocab_size: action-embedding vocabulary size (the action is the
            next input token id).
        horizon: rollout depth `d` for the multi-step latent prediction.
        lambda_h, lambda_kl: loss-mixing coefficients.
        n_hidden_layers, hidden_mult: shape of the dynamics MLP.
        stop_grad_target: if True, detach the target hidden states (matches
            Algorithm 1's `sg(.)`).
        mask_kl: if True, average KL only over positions whose label is not
            `-100` (skip the prompt). `loss_next_h` is unmasked since the paper
            text only places stop-gradient on the KL targets.
    """

    def __init__(
        self,
        model: nn.Module,
        dim: int,
        vocab_size: int,
        horizon: int,
        lambda_h: float = 1.0,
        lambda_kl: float = 1.0,
        n_hidden_layers: int = 2,
        hidden_mult: int = 4,
        stop_grad_target: bool = True,
        mask_kl: bool = True,
    ) -> None:
        super().__init__()
        if not hasattr(model, "norm") or not hasattr(model, "output"):
            raise AttributeError(
                "NextLatWrapper requires the wrapped model to expose `model.norm` "
                "(final RMSNorm) and `model.output` (lm-head)."
            )
        self.model = model
        self.dynamics = LatentDynamics(dim, vocab_size, n_hidden_layers, hidden_mult)
        self.horizon = int(horizon)
        self.lambda_h = float(lambda_h)
        self.lambda_kl = float(lambda_kl)
        self.stop_grad_target = bool(stop_grad_target)
        self.mask_kl = bool(mask_kl)
        self._hidden_buf: Optional[Tensor] = None

        # WARNING: NextLat assumes the wrapped model exposes a single final
        # RMSNorm at `model.norm` whose output is the post-final-norm /
        # pre-logits hidden state fed directly to `model.output` (the lm_head).
        # If you rename `.norm`, run it more than once per forward, or move the
        # lm-head input elsewhere, this hook will silently capture the wrong
        # tensor (or nothing). Update `_capture` / this registration if the
        # base model definition changes.
        warnings.warn(
            "NextLatWrapper: capturing hidden states via forward-hook on "
            "`model.norm` (assumed to be the final RMSNorm whose output is the "
            "pre-logits input to `model.output`). If the base model definition "
            "changes, update next_token/nextlat.py accordingly.",
            stacklevel=2,
        )
        model.norm.register_forward_hook(self._capture)

    # ---- internals -----------------------------------------------------

    def _capture(self, module: nn.Module, inp: Any, out: Tensor) -> None:
        self._hidden_buf = out

    # ---- delegation ----------------------------------------------------

    def setup_cache(self, device=None):
        return self.model.setup_cache(device=device)

    def setup_kv_cache(self, *args, **kwargs):
        # Generation helpers from the base transformer.
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

        # 1) Run the base model to get logits_post and capture h via the hook.
        self._hidden_buf = None
        logits_post, stats = self.model(input_ids, labels=None, **kwargs)
        h = self._hidden_buf
        if h is None:
            raise RuntimeError(
                "NextLatWrapper: forward hook on `model.norm` did not fire. "
                "Has the base model definition changed?"
            )

        # 2) Standard next-token CE on the actual logits.
        loss_next = F.cross_entropy(
            logits_post.reshape(-1, logits_post.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        # 3) Multi-step latent rollout (Algorithm 1).
        loss_next_h = h.new_zeros(())
        loss_kl = h.new_zeros(())

        if self.horizon > 0:
            B, T, D = h.shape
            # Algorithm 1 writes (B,1,1) for the dummy initial state; we use
            # (B,1,D) so the cat along dim=1 actually produces a (B,T,D) shifted
            # tensor. Initial state is zeros either way.
            dummy = h.new_zeros(B, 1, D)

            target_states = h.detach() if self.stop_grad_target else h
            logits_target = logits_post.detach()
            log_p_target = F.log_softmax(logits_target, dim=-1)
            p_target = log_p_target.exp()

            # Detached output head: equivalent to a frozen copy of the lm-head
            # without the cost of an actual deepcopy each forward.
            head_w = self.model.output.weight.detach()
            head_b = (
                self.model.output.bias.detach()
                if self.model.output.bias is not None
                else None
            )

            if self.mask_kl:
                kl_mask = (labels != -100).to(h.dtype)
                kl_denom = kl_mask.sum().clamp_min(1.0)
            else:
                kl_mask = None
                kl_denom = None

            current = h
            for _ in range(self.horizon):
                shifted = torch.cat([dummy, current[:, :-1]], dim=1)
                pred = self.dynamics(shifted, input_ids)

                loss_next_h = loss_next_h + F.smooth_l1_loss(pred, target_states)

                logits_prior = F.linear(pred, head_w, head_b)
                log_p_prior = F.log_softmax(logits_prior, dim=-1)
                # KL(p_target || p_prior) summed over vocab.
                kl_pos = (p_target * (log_p_target - log_p_prior)).sum(dim=-1)
                if kl_mask is not None:
                    kl_pos = (kl_pos * kl_mask).sum() / kl_denom
                else:
                    kl_pos = kl_pos.mean()
                loss_kl = loss_kl + kl_pos

                current = pred

            loss_next_h = loss_next_h / self.horizon
            loss_kl = loss_kl / self.horizon

        total = loss_next + self.lambda_h * loss_next_h + self.lambda_kl * loss_kl

        stats = dict(stats) if stats else {}
        stats["nextlat/loss_next"] = float(loss_next.detach())
        stats["nextlat/loss_next_h"] = float(loss_next_h.detach())
        stats["nextlat/loss_kl"] = float(loss_kl.detach())
        stats["nextlat/loss_total"] = float(total.detach())
        return total, stats


def build_nextlat(
    model: nn.Module,
    vocab_size: int,
    cfg_nextlat,
    horizon: int,
) -> NextLatWrapper:
    """Construct a `NextLatWrapper` around `model` from a `cfg.nextlat` block.

    `vocab_size` should be the action vocabulary (the tokenizer's actual
    `vocab_size`, not the padded one). `dim` is read from the wrapped model's
    config so the dynamics MLP matches the residual stream size.
    """
    if not hasattr(model, "config") or not hasattr(model.config, "dim"):
        raise AttributeError(
            "build_nextlat: wrapped model must expose `model.config.dim`."
        )
    dim = int(model.config.dim)

    return NextLatWrapper(
        model=model,
        dim=dim,
        vocab_size=int(vocab_size),
        horizon=int(horizon),
        lambda_h=float(cfg_nextlat.get("lambda_h", 1.0)),
        lambda_kl=float(cfg_nextlat.get("lambda_kl", 1.0)),
        n_hidden_layers=int(cfg_nextlat.get("n_hidden_layers", 2)),
        hidden_mult=int(cfg_nextlat.get("hidden_mult", 4)),
        stop_grad_target=bool(cfg_nextlat.get("stop_grad_target", True)),
        mask_kl=bool(cfg_nextlat.get("mask_kl", True)),
    )
