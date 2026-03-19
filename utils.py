import math
import random
import os

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Iterable, Optional
from typing_extensions import Self
from tqdm import tqdm


def build_hourglass_config(vocab_size, block_size, n_levels, dim=768):
    # TODO: fix this ugly dim arg
    """Recursively build a nested Config chain with n_levels of hierarchy."""
    from hnet import Config

    proc_dim = (dim * 3) // 2

    if n_levels <= 1:
        return Config(
            vocab_size=vocab_size,
            block_size=block_size,
            dim=dim,
            processor_dim=proc_dim,
            processor_config=None,
        )

    inner_block_size = block_size

    inner = build_hourglass_config(
        vocab_size=vocab_size,
        block_size=inner_block_size,
        n_levels=n_levels - 1,
        dim=proc_dim,
    )

    return Config(
        vocab_size=vocab_size,
        block_size=block_size,
        dim=dim,
        processor_dim=proc_dim,
        n_compressor_layers=3,
        n_processor_layers=6,
        n_decoder_layers=3,
        processor_config=inner,
    )


def get_model(config):

    if config.model == "transformer":
        from transformer import TransformerConfig, TransformerLM

        model_config = TransformerConfig(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
        )
        model = TransformerLM(model_config)
        return model_config, model

    elif config.model == "hourglass":
        from hnet import HierarchicalLM

        model_config = build_hourglass_config(
            config.vocab_size, config.block_size, config.n_levels,
        )
        model = HierarchicalLM(model_config)
        return model_config, model

    else:
        raise ValueError(f"Unknown model: {config.model}")


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
    _, _, boundaries, counts, _ = model.model.compressor(x, cos, sin)

    for b in range(min(n, B)):
        tokens = tokenizer.convert_ids_to_tokens(input_ids[b].tolist())
        nc = counts[b].item()
        bnd = set(boundaries[b, :nc].tolist())

        parts = []
        for i, tok in enumerate(tokens):
            if i in bnd and i > 0:
                parts.append("|")
            parts.append(tok)
        print(f"[boundaries] chunks={nc}\n{''.join(parts)}\n")
    print()

    model.train()


@torch.no_grad()
def validate(model, val_dataloader, device, eval_iters=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, (input_ids, targets) in enumerate(tqdm(val_dataloader, desc="Evaluating", total=eval_iters)):
        if eval_iters is not None and i >= eval_iters:
            break
        input_ids, targets = input_ids.to(device), targets.to(device)
        num_tokens = (targets != -100).sum().item()
        logits, _ = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    val_loss = total_loss / total_tokens
    perplexity = math.exp(val_loss)
    model.train()
    return val_loss, perplexity


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
