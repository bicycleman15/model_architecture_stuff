"""Self-contained data pipeline for the S_n running-product task.

Reads ``data_dir/{group}={k}.csv`` (produced by ``src/generate_data.py``), builds
a HuggingFace WordLevel tokenizer with BOS-prepended templates, and returns
PyTorch DataLoaders for training and evaluation.

No imports from ``state_tracking/src/*`` or ``sfirah.*`` — this file can outlive
``state_tracking/src/``.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


def _default_num_proc() -> int:
    """Return a reasonable default worker count for dataset ``.map()``.

    Prefers ``os.sched_getaffinity`` (respects HPC/SLURM core pinning) and
    falls back to ``os.cpu_count``. Caps at 16 to avoid diminishing returns
    and excess memory from Arrow shards.
    """
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    return max(1, min(16, n - 1 if n > 1 else 1))


class SpecialTokens:
    PAD = "[PAD]"
    BOS = "[BOS]"
    UNK = "[UNK]"
    EOS = "[EOS]"
    SEP = "[SEP]"
    CLS = "[CLS]"
    MASK = "[MASK]"

    @classmethod
    def values(cls):
        return [cls.PAD, cls.BOS, cls.UNK, cls.EOS, cls.SEP, cls.CLS, cls.MASK]


def pad_collate(samples: list[dict[str, Tensor]], pad_token_id: int) -> dict[str, Tensor]:
    """Right-pad ``input_ids`` and ``labels`` of each sample to the batch max."""
    channels_to_pad = ["input_ids"]
    if samples[0]["labels"].dim() > 0:
        channels_to_pad.append("labels")

    max_lens = {c: max(s[c].shape[0] for s in samples) for c in channels_to_pad}

    for s in samples:
        for c in channels_to_pad:
            if max_lens[c] > s[c].shape[0]:
                s[c] = F.pad(s[c], (0, max_lens[c] - s[c].shape[0]), value=pad_token_id)

    return {
        "input_ids": torch.stack([s["input_ids"] for s in samples]),
        "labels": torch.stack([s["labels"] for s in samples]),
    }


def _tokenize(example, tokenizer: PreTrainedTokenizerFast):
    """Tokenize a batch of examples (tagging task: per-token labels)."""
    tokenized = tokenizer(example["input"], return_tensors="pt", padding=True)
    tokenized.pop("attention_mask", None)
    tokenized["labels"] = tokenizer(example["target"], return_tensors="pt", padding=True)[
        "input_ids"
    ]
    return tokenized


def _build_tokenizer(unique_tokens: list[str]) -> PreTrainedTokenizerFast:
    """Build a WordLevel tokenizer over the provided vocabulary tokens."""
    tok = Tokenizer(WordLevel())
    tok.pre_tokenizer = WhitespaceSplit()
    tok.add_tokens(sorted(unique_tokens, key=lambda x: int(x)))
    tok.add_special_tokens(SpecialTokens.values())
    tok.post_processor = TemplateProcessing(
        single=f"{SpecialTokens.BOS} $A",
        special_tokens=[(SpecialTokens.BOS, tok.token_to_id(SpecialTokens.BOS))],
    )
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=SpecialTokens.BOS,
        unk_token=SpecialTokens.UNK,
        eos_token=SpecialTokens.EOS,
        sep_token=SpecialTokens.SEP,
        cls_token=SpecialTokens.CLS,
        mask_token=SpecialTokens.MASK,
        pad_token=SpecialTokens.PAD,
    )
    hf_tok.padding_side = "right"
    return hf_tok


def _load_tokenized(
    path: Path,
    tokenizer: PreTrainedTokenizerFast,
    max_samples: int | None,
    num_proc: int = 1,
):
    # Fast tokenizers already use Rust-level parallelism via rayon; when we
    # also fork for datasets.map(), the inner parallelism is disabled and HF
    # prints a noisy warning unless we silence it explicitly.
    if num_proc > 1:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ds = (
        load_dataset("csv", data_files=str(path), split="all")
        .remove_columns(["seed"])
        .map(
            partial(_tokenize, tokenizer=tokenizer),
            batched=True,
            num_proc=num_proc if num_proc > 1 else None,
        )
    )
    drop = [c for c in ["input", "target", "token_type_ids"] if c in ds.column_names]
    ds = ds.remove_columns(drop)
    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))
    return ds.with_format("torch")


def build_dataloaders(
    group: str,
    k: int,
    k_test: int | None,
    data_dir: str | Path,
    batch_size: int,
    train_size: float = 0.99,
    max_samples: int | None = None,
    num_workers: int = 0,
    map_num_proc: int | None = None,
) -> dict:
    """Build train/eval DataLoaders for ``{group}={k}.csv`` / ``{group}={k_test}.csv``.

    Returns a dict with keys ``train_loader``, ``eval_loader``, ``tokenizer``,
    and ``n_vocab`` (tokenizer vocab size including special tokens).
    """
    data_dir = Path(data_dir)
    train_path = data_dir / f"{group}={k}.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train CSV: {train_path}")

    # Vocab comes from the unique tokens in the train split's `input` column.
    unique_tokens = (
        pl.read_csv(train_path)
        .select(pl.col("input").map_batches(lambda x: x.str.split(" ")))
        .explode("input")
        .unique()["input"]
        .to_list()
    )
    tokenizer = _build_tokenizer(unique_tokens)
    collate = partial(pad_collate, pad_token_id=tokenizer.pad_token_id)

    n_proc = _default_num_proc() if map_num_proc is None else max(1, int(map_num_proc))
    train_ds_full = _load_tokenized(train_path, tokenizer, max_samples, num_proc=n_proc)

    # Train/val split on the training file (val from same length as train).
    if train_size < 1.0:
        split = train_ds_full.train_test_split(train_size=train_size)
        train_ds = split["train"]
        inlen_val_ds = split["test"]
    else:
        train_ds = train_ds_full
        inlen_val_ds = train_ds_full  # fallback; eval on the longer k_test below

    # Evaluation set: prefer a separate k_test file when present (length extrapolation).
    if k_test is not None and k_test != k:
        test_path = data_dir / f"{group}={k_test}.csv"
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test CSV: {test_path}")
        eval_ds = _load_tokenized(test_path, tokenizer, max_samples, num_proc=n_proc)
    else:
        eval_ds = inlen_val_ds

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True,
    )

    # `len(tokenizer)` returns the full vocab size (base + added special tokens).
    n_vocab = len(tokenizer)

    return {
        "train_loader": train_loader,
        "eval_loader": eval_loader,
        "tokenizer": tokenizer,
        "n_vocab": n_vocab,
    }
