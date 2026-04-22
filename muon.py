"""
Pure PyTorch Muon optimizer.

Muon (MomentUm Orthogonalized by Newton-schulz), Keller Jordan, 2024.
  https://kellerjordan.github.io/posts/muon/

The update rule is standard SGD-with-(Nesterov)-momentum, except every update is
replaced with its nearest semi-orthogonal matrix via a quintic Newton-Schulz
iteration run in bfloat16. Muon is applied only to the 2D "hidden" weight
matrices of the network (attention/MLP projections). Everything else --
embeddings, lm_head, norm gains, biases, scalars, gates -- gets AdamW.

This file is a single-device, pure-PyTorch port of the reference. No Triton,
no distributed collectives, no FP8, no Polar Express. Use with any nn.Module.

Usage (see ``overfit.py`` for a working example)::

    from muon import MuonWithAuxAdamW, build_muon_param_groups

    param_groups = build_muon_param_groups(
        model,
        muon_lr=cfg.optimizer.lr,
        muon_weight_decay=0.0,
        adamw_lr=cfg.optimizer.lr * 0.1,
        adamw_betas=tuple(cfg.optimizer.betas),
        adamw_weight_decay=cfg.optimizer.weight_decay,
    )
    optimizer = MuonWithAuxAdamW(param_groups)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
from torch import Tensor, nn


# Quintic coefficients tuned by @KellerJordan so that the composition of 5
# iterations reliably pushes all singular values into [1 - eps, 1 + eps].
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """Compute the nearest semi-orthogonal matrix to ``G`` via Newton-Schulz.

    Runs entirely in bfloat16 on the GPU, which is both faster and numerically
    adequate for the Muon update. For rectangular matrices we transpose so the
    iteration always runs with the "short" dimension first, keeping the inner
    matmul small.

    Args:
        G: At least 2D tensor. The iteration is applied to the last 2 dims.
        steps: Number of Newton-Schulz iterations (5 is the canonical choice).

    Returns:
        Tensor with the same shape and dtype as ``G`` whose singular values
        are all ~1 (so ``out`` is close to ``U @ V^T`` from the SVD of ``G``).
    """
    assert G.ndim >= 2, "Newton-Schulz expects matrices"
    a, b, c = _NS_COEFFS

    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT

    # Normalize so the largest singular value is <= 1 before iterating.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Pure PyTorch Muon optimizer for 2D weight matrices.

    Intended for the "hidden" projection weights of a transformer. Do NOT put
    embeddings, output heads, biases, norm gains, or any 1D parameter here --
    use AdamW for those (see ``MuonWithAuxAdamW``).

    Args:
        params: Iterable of 2D parameters (or param groups).
        lr: Base learning rate.
        weight_decay: Decoupled weight decay (applied multiplicatively per step).
        momentum: Heavy-ball / Nesterov momentum coefficient.
        nesterov: Whether to use Nesterov-style lookahead momentum.
        ns_steps: Number of Newton-Schulz iterations per step.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.ndim < 2:
                    raise ValueError(
                        "Muon only supports matrix (ndim >= 2) parameters; got "
                        f"shape {tuple(p.shape)}. Route 1D params to AdamW."
                    )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf: Tensor = state["momentum_buffer"]

                # SGD-style momentum update in FP32-compatible dtype.
                buf.lerp_(grad, 1 - mom)
                update = grad.lerp_(buf, mom) if nesterov else buf

                # Flatten any trailing dims so Newton-Schulz sees a 2D matrix.
                flat = update.reshape(update.shape[0], -1)
                ortho = zeropower_via_newtonschulz5(flat, steps=ns_steps)
                ortho = ortho.view_as(p)

                # Scale the orthogonalized update so it has roughly unit RMS.
                # A spectral-norm-1 update has RMS ~ sqrt(min(m,n) / (m*n));
                # this factor (from Keller) restores ~1 regardless of shape.
                fan_out, fan_in = p.shape[-2], p.shape[-1]
                scale = max(1.0, fan_out / fan_in) ** 0.5

                # Decoupled weight decay + update.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(ortho, alpha=-lr * scale)

        return loss


def _is_muon_param(name: str, param: nn.Parameter) -> bool:
    """Default policy: 2D tensors from the transformer body go to Muon.

    Excludes anything that looks like an embedding, output head, norm gain,
    bias, or 1D scalar.
    """
    if not param.requires_grad:
        return False
    if param.ndim < 2:
        return False
    lowered = name.lower()
    for needle in ("embed", "embedding", "wte", "wpe", "lm_head", "head", "norm", "bias"):
        if needle in lowered:
            return False
    return True


def build_muon_param_groups(
    model: nn.Module,
    *,
    muon_lr: float,
    muon_weight_decay: float = 0.0,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_steps: int = 5,
    adamw_lr: float,
    adamw_betas: Sequence[float] = (0.9, 0.95),
    adamw_eps: float = 1e-8,
    adamw_weight_decay: float = 0.0,
    is_muon_param=_is_muon_param,
) -> List[dict]:
    """Split ``model``'s parameters into Muon and AdamW groups.

    Returns a list of param groups suitable for ``MuonWithAuxAdamW``. Each
    group is tagged with ``use_muon: bool`` so the wrapper optimizer knows
    which update rule to apply.
    """
    muon_params: List[nn.Parameter] = []
    adamw_params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_muon_param(name, param):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    groups: List[dict] = []
    if muon_params:
        groups.append(dict(
            params=muon_params,
            use_muon=True,
            lr=muon_lr,
            weight_decay=muon_weight_decay,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
        ))
    if adamw_params:
        groups.append(dict(
            params=adamw_params,
            use_muon=False,
            lr=adamw_lr,
            betas=tuple(adamw_betas),
            eps=adamw_eps,
            weight_decay=adamw_weight_decay,
        ))
    return groups


class MuonWithAuxAdamW(torch.optim.Optimizer):
    """Single-device optimizer that routes each param group to Muon or AdamW.

    Param groups must carry a boolean ``use_muon`` flag. Groups with
    ``use_muon=True`` are updated by :class:`Muon`; the rest use standard
    AdamW with decoupled weight decay.

    This mirrors Keller Jordan's reference ``SingleDeviceMuonWithAuxAdam`` but
    is written as a straight ``torch.optim.Optimizer`` so that existing
    training loops (``optimizer.param_groups``, ``optimizer.step()``,
    ``optimizer.zero_grad()``) work unchanged.
    """

    def __init__(self, param_groups: Iterable[dict]) -> None:
        groups = list(param_groups)
        for g in groups:
            if "use_muon" not in g:
                raise ValueError("every param group must set 'use_muon' (True/False)")
            if g["use_muon"]:
                g.setdefault("weight_decay", 0.0)
                g.setdefault("momentum", 0.95)
                g.setdefault("nesterov", True)
                g.setdefault("ns_steps", 5)
            else:
                g.setdefault("betas", (0.9, 0.95))
                g.setdefault("eps", 1e-8)
                g.setdefault("weight_decay", 0.0)

        defaults: dict = {}
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    @torch.no_grad()
    def _muon_step(self, group: dict) -> None:
        lr = group["lr"]
        wd = group["weight_decay"]
        mom = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.ndim < 2:
                raise ValueError(
                    "Muon group received a non-matrix parameter of shape "
                    f"{tuple(p.shape)}; route 1D params to the AdamW group."
                )

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)
            buf: Tensor = state["momentum_buffer"]

            buf.lerp_(grad, 1 - mom)
            update = grad.lerp_(buf, mom) if nesterov else buf

            flat = update.reshape(update.shape[0], -1)
            ortho = zeropower_via_newtonschulz5(flat, steps=ns_steps).view_as(p)

            fan_out, fan_in = p.shape[-2], p.shape[-1]
            scale = max(1.0, fan_out / fan_in) ** 0.5

            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.add_(ortho, alpha=-lr * scale)

    @torch.no_grad()
    def _adamw_step(self, group: dict) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

            state["step"] += 1
            t = state["step"]
            exp_avg: Tensor = state["exp_avg"]
            exp_avg_sq: Tensor = state["exp_avg_sq"]

            g_f32 = grad.float()
            exp_avg.mul_(beta1).add_(g_f32, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g_f32, g_f32, value=1.0 - beta2)

            bias1 = 1.0 - beta1 ** t
            bias2 = 1.0 - beta2 ** t
            step_size = lr * (bias2 ** 0.5) / bias1

            denom = exp_avg_sq.sqrt().add_(eps)
            update = (exp_avg / denom).to(p.dtype)

            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.add_(update, alpha=-step_size)
