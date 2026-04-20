"""Hydra-based training entry point for the state-tracking task.

Inspired by the workspace-root ``train.py`` (Hydra + Accelerate) and the
reference ``state_tracking/src/main.py`` (curriculum + per-position metrics),
but does not import from either.

Run from the workspace root:
    python -m state_tracking.train
    python -m state_tracking.train model=deltanet
    python -m state_tracking.train model=transformer schedule.epochs=5
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

# Ensure the workspace root is on sys.path so the transformer shim (which in turn
# uses importlib to load models/transformer.py) can resolve any transitive
# file-local imports it might have.
_WS_ROOT = Path(__file__).resolve().parents[1]
if str(_WS_ROOT) not in sys.path:
    sys.path.insert(0, str(_WS_ROOT))

from state_tracking.data_module import build_dataloaders  # noqa: E402
from state_tracking.metrics import (  # noqa: E402
    cumulative_sequence_accuracies,
    reduce_metrics,
    sequence_accuracy,
    token_accuracy,
)
from state_tracking.models import get_model  # noqa: E402

log = logging.getLogger(__name__)
_DATE_STR = datetime.now().strftime("%Y-%m-%d")


def _num_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_pretrained(accelerator: Accelerator, model, cfg: DictConfig, seed: int):
    """Save the unwrapped model's state_dict and config to a dated directory.

    Adapted (inlined, not imported) from the reference
    ``state_tracking/src/main.py::save_pretrained``.
    """
    if not accelerator.is_main_process:
        return

    model_name = cfg.model.name
    group = cfg.data.group
    k = cfg.data.k
    hh = cfg.model.get("num_householder", 0)

    save_root = Path(cfg.checkpoint.save_dir)
    if not save_root.is_absolute():
        save_root = Path(hydra.core.hydra_config.HydraConfig.get().runtime.cwd) / save_root

    save_dir = save_root / f"{group}_{model_name}_k{k}_hh{hh}_{_DATE_STR}_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)

    hf_config = getattr(unwrapped, "config", None)
    if hf_config is not None and hasattr(hf_config, "to_dict"):
        with open(save_dir / "config.json", "w") as f:
            json.dump(hf_config.to_dict(), f, indent=2)
    else:
        with open(save_dir / "config.yaml", "w") as f:
            OmegaConf.save(cfg, f)

    torch.save(unwrapped.state_dict(), save_dir / "pytorch_model.bin")
    log.info(f"Saved checkpoint to {save_dir}")


def _run_eval(model, eval_loader, accelerator: Accelerator, pad_token_id: int) -> dict:
    """Run evaluation and return a dict of averaged metrics."""
    model.eval()
    records = []
    for batch in tqdm(eval_loader, desc="Eval", leave=False, disable=not accelerator.is_main_process):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        with torch.no_grad(), accelerator.autocast():
            logits = model(input_ids)

        logits_g, labels_g = accelerator.gather_for_metrics((logits, labels))

        records.append(
            {
                "val/token_accuracy": token_accuracy(logits_g, labels_g, pad_token_id),
                "val/sequence_accuracy": sequence_accuracy(logits_g, labels_g, pad_token_id),
                "val/cumulative_sequence_accuracies": cumulative_sequence_accuracies(
                    logits_g, labels_g, pad_token_id
                ),
            }
        )

    return reduce_metrics(records)


@hydra.main(config_path="config", config_name="state_tracking", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        log_with="wandb" if cfg.logging.wandb else None,
    )
    if accelerator.is_main_process:
        log.setLevel(logging.INFO)
        log.info(OmegaConf.to_yaml(cfg))

    # Data
    dm = build_dataloaders(
        group=cfg.data.group,
        k=cfg.data.k,
        k_test=cfg.data.k_test,
        data_dir=cfg.data.data_dir,
        batch_size=cfg.batch_size,
        train_size=cfg.data.train_size,
        max_samples=cfg.data.max_samples,
        map_num_proc=cfg.data.get("map_num_proc", None),
    )
    tokenizer = dm["tokenizer"]
    n_vocab = dm["n_vocab"]
    pad_id = tokenizer.pad_token_id

    if accelerator.is_main_process:
        log.info(f"vocab size: {n_vocab} | pad_id: {pad_id}")
        log.info(f"train batches: {len(dm['train_loader'])} | eval batches: {len(dm['eval_loader'])}")

    # Model
    model = get_model(cfg, vocab_size=n_vocab)
    if accelerator.is_main_process:
        log.info(f"Model: {cfg.model.name} | params: {_num_params(model):,}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )

    steps_per_epoch = max(1, len(dm["train_loader"]))
    total_steps = cfg.schedule.epochs * steps_per_epoch
    warmup_steps = max(1, int(cfg.schedule.warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if not cfg.schedule.use_cosine:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Prepare with Accelerate
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, dm["train_loader"], dm["eval_loader"], scheduler
    )

    # RoPE cache for the Transformer baseline (no-op for DeltaProduct/DeltaNet)
    unwrapped = accelerator.unwrap_model(model)
    if hasattr(unwrapped, "setup_cache"):
        unwrapped.setup_cache(device=accelerator.device)

    # Init trackers (wandb)
    if cfg.logging.wandb:
        init_kwargs = {}
        if cfg.logging.entity:
            init_kwargs["wandb"] = {"entity": cfg.logging.entity}
        accelerator.init_trackers(
            cfg.project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs=init_kwargs,
        )

    # Shape sanity check on the very first batch
    first_batch = next(iter(train_loader))
    if accelerator.is_main_process:
        log.info(
            f"first batch shapes: input_ids={tuple(first_batch['input_ids'].shape)} "
            f"labels={tuple(first_batch['labels'].shape)}"
        )
    assert first_batch["input_ids"].shape == first_batch["labels"].shape, (
        "Expected input_ids and labels to have the same shape (both BOS-prefixed)."
    )

    # Curriculum
    max_curr = first_batch["labels"].shape[1]  # k + 1 with BOS
    if cfg.curriculum.enabled:
        curriculum_idx = min(cfg.curriculum.start_idx, max_curr)
    else:
        curriculum_idx = max_curr

    global_step = 0
    best_val_acc = 0.0

    for epoch in tqdm(range(cfg.schedule.epochs), desc="Epochs", disable=not accelerator.is_main_process):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_bar = tqdm(
            train_loader,
            desc=f"Train e{epoch}",
            leave=False,
            disable=not accelerator.is_main_process,
        )
        for batch in t_bar:
            global_step += 1
            optimizer.zero_grad()

            input_ids = batch["input_ids"]
            labels = batch["labels"]

            with accelerator.autocast():
                logits = model(input_ids)  # [B, T, V]

                T = labels.shape[1]
                idx = min(curriculum_idx, T)
                tgt = labels[:, :idx].reshape(-1)
                pred = logits[:, :idx, :].reshape(-1, logits.size(-1))
                loss = F.cross_entropy(pred, tgt, ignore_index=pad_id)

            accelerator.backward(loss)
            if cfg.optimizer.grad_clip and cfg.optimizer.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            t_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}", curr=idx)

            if cfg.logging.wandb and accelerator.is_main_process:
                accelerator.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/curriculum_idx": idx,
                    },
                    step=global_step,
                )

        mean_train_loss = epoch_loss / max(1, n_batches)

        if (
            cfg.curriculum.enabled
            and mean_train_loss < cfg.curriculum.loss_threshold
            and curriculum_idx < max_curr
        ):
            curriculum_idx = min(curriculum_idx + cfg.curriculum.step, max_curr)
            if accelerator.is_main_process:
                log.info(f"Curriculum -> {curriculum_idx}")

        # Eval
        if (epoch + 1) % cfg.eval.every_n_epochs == 0 or epoch == cfg.schedule.epochs - 1:
            eval_metrics = _run_eval(model, eval_loader, accelerator, pad_id)
            eval_metrics["train/epoch_loss"] = mean_train_loss
            if accelerator.is_main_process:
                log.info(
                    f"epoch {epoch}: train_loss={mean_train_loss:.4f} "
                    f"val_tok_acc={eval_metrics.get('val/token_accuracy', 0):.4f} "
                    f"val_seq_acc={eval_metrics.get('val/sequence_accuracy', 0):.4f}"
                )
                if cfg.logging.wandb:
                    loggable = {
                        k: v for k, v in eval_metrics.items() if not hasattr(v, "shape")
                    }
                    accelerator.log(loggable, step=global_step)

            cur_seq_acc = eval_metrics.get("val/sequence_accuracy", 0.0)
            if cur_seq_acc > best_val_acc:
                best_val_acc = cur_seq_acc
                if cfg.checkpoint.save_on_improve:
                    save_pretrained(accelerator, model, cfg, cfg.seed)

    # Final save
    save_pretrained(accelerator, model, cfg, cfg.seed)

    if cfg.logging.wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()
