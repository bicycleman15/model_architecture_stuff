import math
import random
import os

import hydra
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Iterable, Optional
from typing_extensions import Self
from tqdm import tqdm


def get_experiment_name(cfg, datetime_str) -> str:
    name = cfg.wandb.exp_name
    name += f" {datetime_str}"
    return name


def create_results_dir(cfg, datetime_str) -> str:
    name = f"{datetime_str}"
    name = "/".join(name.split(" "))

    original_cwd = hydra.utils.get_original_cwd()
    result_dir = os.path.join(original_cwd, cfg.results_dir, cfg.wandb.project, name)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def _decode_chunk(ids, tokenizer, bos_id, eos_id):
    """Decode a chunk of token ids, rendering BOS/EOS as <s>/<\s>."""
    parts = []
    text_buf = []
    for t in ids:
        if t == bos_id:
            if text_buf:
                parts.append(tokenizer.decode(text_buf, errors="replace"))
                text_buf = []
            parts.append("<s>")
        elif t == eos_id:
            if text_buf:
                parts.append(tokenizer.decode(text_buf, errors="replace"))
                text_buf = []
            parts.append("</s>")
        else:
            text_buf.append(t)
    if text_buf:
        parts.append(tokenizer.decode(text_buf, errors="replace"))
    return "".join(parts)


@torch.no_grad()
def visualize_boundaries(model, val_dataloader, tokenizer, n=3):
    """Show how the compressor chunks a sample, using | as delimiters."""
    model.eval()
    input_ids, _ = next(iter(val_dataloader))
    input_ids = input_ids[:n].to(next(model.parameters()).device)

    x = model.emb(input_ids)
    B, L, D = x.shape
    cos = model.model.cos[:, :L]
    sin = model.model.sin[:, :L]
    compressor_out = model.model.compressor(x, cos, sin, input_ids=input_ids)
    boundaries, counts = compressor_out[2], compressor_out[3]

    bos_id = getattr(tokenizer, "bos_idx", None) or getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_idx", None) or getattr(tokenizer, "eos_token_id", None)

    for b in range(min(n, B)):
        ids = input_ids[b].tolist()
        nc = counts[b].item()
        bnd = sorted(boundaries[b, :nc].tolist())

        parts = []
        prev = 0
        for boundary in bnd:
            chunk = ids[prev:boundary]
            parts.append(_decode_chunk(chunk, tokenizer, bos_id, eos_id))
            prev = boundary
        chunk = ids[prev:]
        parts.append(_decode_chunk(chunk, tokenizer, bos_id, eos_id))

        print(f"[boundaries] chunks={nc}\n{'|'.join(parts)}\n")
    print()

    model.train()


@torch.no_grad()
def validate(model, val_dataloader, device, eval_iters=None, bytes_per_token=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0

    for i, (input_ids, targets) in enumerate(tqdm(val_dataloader, desc="Evaluating", total=eval_iters)):
        if eval_iters is not None and i >= eval_iters:
            break
        input_ids, targets = input_ids.to(device), targets.to(device)
        num_tokens = (targets != -100).sum().item()
        loss, stats = model(input_ids, labels=targets)
        ce_loss = stats.get("reinforce/ce_loss", loss.item())
        total_loss += ce_loss * num_tokens
        total_tokens += num_tokens

        if bytes_per_token is not None:
            valid_targets = targets[targets != -100]
            total_bytes += bytes_per_token[valid_targets].sum().item()

    val_loss = total_loss / total_tokens
    perplexity = math.exp(val_loss)

    val_bpb = None
    if bytes_per_token is not None and total_bytes > 0:
        val_bpb = total_loss / total_bytes / math.log(2)

    model.train()
    return val_loss, perplexity, val_bpb


@torch.no_grad()
def validate_char_only(model, val_dataloader, device, eval_iters=None, bytes_per_token=None, pad_zero_idx=0):
    """Like validate(), but masks out zero-padding tokens (target == pad_zero_idx) from loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0

    for i, (input_ids, targets) in enumerate(tqdm(val_dataloader, desc="Evaluating (char-only)", total=eval_iters)):
        if eval_iters is not None and i >= eval_iters:
            break
        input_ids, targets = input_ids.to(device), targets.to(device)
        targets = targets.clone()
        targets[targets == pad_zero_idx] = -100
        num_tokens = (targets != -100).sum().item()
        loss, stats = model(input_ids, labels=targets)
        ce_loss = stats.get("reinforce/ce_loss", loss.item())
        total_loss += ce_loss * num_tokens
        total_tokens += num_tokens

        if bytes_per_token is not None:
            valid_targets = targets[targets != -100]
            total_bytes += bytes_per_token[valid_targets].sum().item()

    val_loss = total_loss / total_tokens
    perplexity = math.exp(val_loss)

    val_bpb = None
    if bytes_per_token is not None and total_bytes > 0:
        val_bpb = total_loss / total_bytes / math.log(2)

    model.train()
    return val_loss, perplexity, val_bpb


def group_params(model, weight_decay):
    """Create optimizer param groups respecting per-param _optim annotations.

    Groups parameters by unique (lr_multiplier, weight_decay) tuples.
    Bias and norm parameters always get weight_decay=0.
    Parameters without _optim annotations get default values
    (lr_multiplier=1.0, weight_decay=weight_decay).
    """
    from models.hourglass import apply_optimization_params

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not hasattr(param, "_optim"):
            param._optim = {}
        if name.endswith(".bias") or ".norm." in name or "norm.weight" in name:
            apply_optimization_params(param, weight_decay=0.0)

    all_keys = set()
    for param in model.parameters():
        if param.requires_grad and hasattr(param, "_optim"):
            all_keys.update(param._optim.keys())
    all_keys = sorted(all_keys)

    all_tuples = []
    param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        optim_dict = getattr(param, "_optim", {})
        current_tuple = tuple(optim_dict.get(key, None) for key in all_keys)
        if current_tuple not in all_tuples:
            all_tuples.append(current_tuple)
            group = {"params": [param], **optim_dict}
            group.setdefault("weight_decay", weight_decay)
            group.setdefault("lr_multiplier", 1.0)
            param_groups.append(group)
        else:
            idx = all_tuples.index(current_tuple)
            param_groups[idx]["params"].append(param)

    return param_groups


def get_lr(learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float) -> float:
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def num_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    total = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            total += p.numel()
    return total


class CycleIterator:
    """An iterator that cycles through an iterable indefinitely without caching."""

    def __init__(self, iterable: Iterable) -> None:
        self.iterable = iterable
        self.epoch = 0
        self._iterator = None

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self.iterable)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterable)
            self.epoch += 1
            return next(self._iterator)

    def __iter__(self) -> Self:
        return self


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_bytes_per_token(tokenizer):
    """Build a lookup table mapping each token ID to its UTF-8 byte length.

    For byte-level tokenizers every token is exactly 1 byte.
    For BPE / sentencepiece tokenizers we decode each vocab entry and
    measure its UTF-8 byte length, giving an exact per-token count.
    """
    from byte_tokenizer import ByteTokenizer

    if isinstance(tokenizer, ByteTokenizer):
        return torch.ones(tokenizer.vocab_size, dtype=torch.float32)

    vocab_size = tokenizer.vocab_size
    table = torch.zeros(vocab_size, dtype=torch.float32)
    for token_id in range(vocab_size):
        try:
            text = tokenizer.decode([token_id])
            table[token_id] = len(text.encode("utf-8"))
        except Exception:
            table[token_id] = 0
    return table
