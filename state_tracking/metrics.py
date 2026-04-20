"""Self-contained metric helpers for the state-tracking task.

Replaces the subset of `sfirah.metrics` and `state_tracking/src/utils.py` used by
the reference training script (main.py). No third-party metric deps beyond torch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _pad_to(x: Tensor, length: int, value) -> Tensor:
    pad = length - x.shape[1]
    if pad <= 0:
        return x
    return F.pad(x, (0, pad) if x.dim() == 2 else (0, 0, 0, pad), value=value)


def detach_and_pad(records, pad_token_id: int) -> dict[str, Tensor]:
    """Concatenate a list of ``(predictions, targets)`` tuples along the batch dim.

    Right-pads the sequence dimension of both predictions (logits) and targets to
    the longest sequence length across all records. Pads logits with 0 and
    targets with ``pad_token_id``.
    """
    preds_list, tgts_list = [], []
    max_len = 0
    for batch in records:
        for p, t in batch if isinstance(batch, list) else [batch]:
            preds_list.append(p.detach())
            tgts_list.append(t.detach())
            max_len = max(max_len, t.shape[1])

    padded_preds = []
    padded_tgts = []
    for p, t in zip(preds_list, tgts_list):
        # p: [B, T, V], t: [B, T]
        if p.shape[1] < max_len:
            pad_len = max_len - p.shape[1]
            p = F.pad(p, (0, 0, 0, pad_len), value=0.0)
            t = F.pad(t, (0, pad_len), value=pad_token_id)
        padded_preds.append(p)
        padded_tgts.append(t)

    return {
        "predictions": torch.cat(padded_preds, dim=0),
        "targets": torch.cat(padded_tgts, dim=0),
    }


def reduce_metrics(list_of_dicts: list[dict]) -> dict:
    """Mean-reduce each key across a list of per-batch metric dicts.

    Tensors and python scalars are averaged numerically. For dict values (e.g.
    ``cumulative_sequence_accuracies`` which returns ``{"value": np.array, ...}``),
    the ``value`` array is averaged element-wise across batches (assuming equal
    length; otherwise the shortest prefix is used).
    """
    if not list_of_dicts:
        return {}

    keys = list_of_dicts[0].keys()
    out: dict = {}
    for k in keys:
        vals = [d[k] for d in list_of_dicts if k in d]
        if not vals:
            continue
        first = vals[0]
        if isinstance(first, dict):
            # e.g. cumulative_sequence_accuracies -> {"value": ndarray, "n_samples": int}
            arrs = [np.asarray(v["value"]) for v in vals]
            min_len = min(a.shape[0] for a in arrs)
            arrs = np.stack([a[:min_len] for a in arrs], axis=0)
            out[k] = arrs.mean(axis=0)
        elif isinstance(first, Tensor):
            out[k] = torch.stack([v.detach().float().cpu() for v in vals]).mean().item()
        else:
            out[k] = float(np.mean([float(v) for v in vals]))
    return out


def ce_loss(logits: Tensor, targets: Tensor, ignore_index: int) -> Tensor:
    """Token-level cross-entropy over [B, T, V] vs [B, T]."""
    return F.cross_entropy(
        logits.flatten(end_dim=-2),
        targets.flatten(),
        ignore_index=ignore_index,
    )


def token_accuracy(logits: Tensor, targets: Tensor, ignore_index: int) -> Tensor:
    """Fraction of non-ignored tokens predicted correctly."""
    if logits.dim() == targets.dim() + 1:
        preds = logits.argmax(dim=-1)
    else:
        preds = logits
    mask = targets != ignore_index
    if mask.sum() == 0:
        return torch.zeros((), device=targets.device)
    return ((preds == targets) & mask).sum().float() / mask.sum().clamp_min(1)


def sequence_accuracy(logits: Tensor, targets: Tensor, ignore_index: int) -> Tensor:
    """Fraction of rows whose (non-ignored) tokens are all predicted correctly."""
    if logits.dim() == targets.dim() + 1:
        preds = logits.argmax(dim=-1)
    else:
        preds = logits
    mask = targets != ignore_index
    wrong = ((preds != targets) & mask).any(dim=1)
    return (~wrong).float().mean()


def cumulative_sequence_accuracies(
    logits: Tensor, targets: Tensor, ignore_index: int | None = None
) -> dict:
    """Per-position prefix accuracy curve averaged over the batch.

    For each position ``t`` in the sequence, computes the fraction of rows in
    which every (non-ignored) prediction up to and including position ``t`` was
    correct.

    Returns ``{"value": np.ndarray of shape [T], "n_samples": int}`` to match the
    reference ``state_tracking/src/utils.py::cumulative_sequence_accuracies``.
    """
    if logits.dim() != targets.dim():
        preds = logits.argmax(dim=-1)
    else:
        preds = logits

    assert preds.size() == targets.size(), f"{preds.size()} != {targets.size()}"

    if ignore_index is not None:
        valid_mask = targets != ignore_index
    else:
        valid_mask = torch.ones_like(targets, dtype=torch.bool)

    correct = (preds == targets) & valid_mask
    cumulative_correct = correct.cumsum(dim=1)
    cumulative_valid = valid_mask.cumsum(dim=1).clamp_min(1)

    ratio = cumulative_correct.float() / cumulative_valid.float()
    accuracies = ratio.floor().mean(dim=0).cpu().numpy()

    return {"value": accuracies, "n_samples": targets.size(0)}
