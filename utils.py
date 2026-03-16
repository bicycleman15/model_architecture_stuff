import math
import random
import os

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Iterable, Optional
from typing_extensions import Self
from tqdm import tqdm


@torch.no_grad()
def validate(model, val_dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, targets in tqdm(val_dataloader, desc="Evaluating"):
        input_ids, targets = input_ids.to(device), targets.to(device)
        num_tokens = (targets != -100).sum().item()
        loss = model(input_ids, targets)
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
