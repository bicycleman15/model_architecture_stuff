"""Model registry for the next-token star-graph experiment.

Currently only ``transformer`` is registered; new architectures can be
added later without touching the trainer.
"""

from __future__ import annotations

from omegaconf import DictConfig


def get_model(cfg: DictConfig, vocab_size: int, block_size: int):
    name = cfg.model.name
    if name == "transformer":
        from next_token.models.transformer import build_transformer

        return build_transformer(vocab_size=vocab_size, block_size=block_size, mcfg=cfg.model)

    if name == "hybrid":
        from next_token.models.hybrid import build_hybrid

        return build_hybrid(vocab_size=vocab_size, block_size=block_size, mcfg=cfg.model)

    raise ValueError(f"unknown model: {name!r}")
