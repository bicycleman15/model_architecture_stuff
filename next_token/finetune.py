"""Hydra + Accelerate supervised finetuner for the CoT star-graph data.

Loads a checkpoint produced by :mod:`next_token.pretrain` and continues
training with the same masked next-token cross-entropy objective on CoT
traces, optionally on a different dataset. Saved checkpoints are
GRPO-compatible (only the base transformer's weights are written, identical
layout to ``pretrain.py``).

Run from the workspace root::

    accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 \
        --num_processes=1 -m next_token.finetune \
        init.ckpt_path=Results/next_token/pretrain/transformer/.../ckpt/final.pt \
        data.dataset=star_5x5_mixed_3M
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_WS_ROOT = Path(__file__).resolve().parents[1]
if str(_WS_ROOT) not in sys.path:
    sys.path.insert(0, str(_WS_ROOT))

from next_token.data import (  # noqa: E402
    cot_pad_collate,
    compute_lengths,
    make_cot_train_targets,
)
from next_token.grpo import _build_model_from_ckpt, _load_ckpt  # noqa: E402
from next_token.pretrain import (  # noqa: E402
    _build_datasets,
    _eval_cot,
    _num_params,
    _save_checkpoint,
)

log = logging.getLogger(__name__)
_DATE_STR = datetime.now().strftime("%Y-%m-%d")


@hydra.main(config_path="config", config_name="finetune", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)

    if accelerator.is_main_process:
        log.setLevel(logging.INFO)
        log.info("\n" + OmegaConf.to_yaml(cfg))

    # ------------------------------------------------------------------ data
    tokenizer, train_ds, test_ds, meta = _build_datasets(cfg)
    deg = int(meta["_resolved"]["deg"])
    path_len = int(meta["_resolved"]["path_len"])
    num_nodes = int(meta["_resolved"]["num_nodes"])

    prefix_len, _ = compute_lengths(deg, path_len, num_nodes)
    assert prefix_len == train_ds.prefix_len == test_ds.prefix_len
    max_target_len = int(meta["max_target_len"])

    # ------------------------------------------------------------------ ckpt
    ckpt_path = Path(cfg.init.ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.cwd) / ckpt_path
    ckpt = _load_ckpt(ckpt_path)
    if accelerator.is_main_process:
        log.info(f"Loaded checkpoint: {ckpt_path}")

    if int(ckpt["vocab_size"]) != tokenizer.vocab_size:
        raise ValueError(
            f"vocab_size mismatch: ckpt={ckpt['vocab_size']} "
            f"tokenizer={tokenizer.vocab_size} (dataset num_nodes={num_nodes})"
        )

    block_size = int(ckpt["block_size"])
    required = prefix_len + max_target_len - 1
    if required > block_size:
        raise ValueError(
            f"Dataset needs block_size>={required} (prefix_len={prefix_len} + "
            f"max_target_len={max_target_len} - 1) but checkpoint was trained "
            f"with block_size={block_size}. Re-pretrain with a larger block "
            f"or pick a smaller dataset."
        )

    model, vocab_size, _ = _build_model_from_ckpt(
        cfg, ckpt, accelerator.device,
        freeze=False, strict=bool(cfg.init.strict),
    )

    if accelerator.is_main_process:
        log.info(f"Base model: {cfg.model.name} | params: {_num_params(model):,}")
        log.info(
            f"Star-graph CoT: deg={deg} path_len={path_len} num_nodes={num_nodes}"
        )
        log.info(
            f"Tokens: prefix={prefix_len} max_target={max_target_len} "
            f"max_seq={prefix_len + max_target_len} (block_size={block_size}) | "
            f"vocab={vocab_size}"
        )
        log.info(
            f"Train samples: {len(train_ds):,} (max trace seen: {train_ds.max_target_len}) | "
            f"Test samples: {len(test_ds):,} (max trace seen: {test_ds.max_target_len})"
        )
        print(model)

    # --------------------------------------------------------------- loaders
    collate = partial(cot_pad_collate, pad_id=tokenizer.DUMMY)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
    )

    # --------------------------------------------------------- optimizer + sched
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.schedule.epochs * steps_per_epoch
    warmup_steps = max(1, int(cfg.schedule.warmup_steps))
    peak_lr = float(cfg.optimizer.lr)
    min_lr = float(cfg.optimizer.get("min_lr", 0.0))
    min_lr_ratio = (min_lr / peak_lr) if peak_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if not cfg.schedule.use_cosine:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )
    accelerator.unwrap_model(model).setup_cache(device=accelerator.device)

    # ----------------------------------------------------------------- wandb
    if cfg.logging.wandb and accelerator.is_main_process:
        run_name = (
            f"ft_{cfg.model.name}_{cfg.data.dataset}"
            f"_lr{cfg.optimizer.lr:g}"
        )
        arch_bits = []
        if int(cfg.model.get("n_layer", 6)) != 6:
            arch_bits.append(f"L{cfg.model.n_layer}")
        if int(cfg.model.get("n_head", 6)) != 6:
            arch_bits.append(f"H{cfg.model.n_head}")
        if int(cfg.model.get("dim", 384)) != 384:
            arch_bits.append(f"D{cfg.model.dim}")
        if arch_bits:
            run_name += "_" + "".join(arch_bits)
        if int(cfg.get("batch_size", 256)) != 256:
            run_name += f"_bs{cfg.batch_size}"
        run_name += f"_T{cfg.eval.temperature:g}_seed{cfg.seed}"
        custom_name = cfg.logging.get("name", None)
        if custom_name:
            run_name = f"{custom_name} {run_name}"
        wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # -------------------------------------------------------------- loop
    eval_every_pct = float(cfg.eval.get("every_pct", 0.1))
    eval_interval = max(1, int(round(total_steps * eval_every_pct)))
    if accelerator.is_main_process:
        log.info(
            f"Eval cadence: every {eval_interval} steps "
            f"({eval_every_pct * 100:.1f}% of {total_steps} total)"
        )

    window_loss = 0.0
    window_count = 0
    global_step = 0

    def run_eval(step: int, epoch: int):
        metrics, samples_log = _eval_cot(
            model,
            test_loader,
            accelerator,
            prefix_len=prefix_len,
            path_len=path_len,
            deg=deg,
            max_target_len=max_target_len,
            pad_id=tokenizer.DUMMY,
            eos_id=tokenizer.EOS,
            temperature=float(cfg.eval.temperature),
            top_k=cfg.eval.get("top_k", None),
            max_batches=cfg.eval.get("max_batches", None),
            tokenizer=tokenizer,
            log_samples=int(cfg.eval.get("log_samples", 0)),
            vocab_size=vocab_size,
        )
        mean_window_loss = window_loss / max(1, window_count)
        metrics["train/window_loss"] = mean_window_loss

        if accelerator.is_main_process:
            report_idx = sorted({0, 1, path_len - 2, path_len - 1})
            tok_str = " ".join(
                f"t{j}={metrics.get(f'val/sample_token_{j}', float('nan')):.3f}"
                for j in report_idx
            )
            log.info(
                f"step {step}/{total_steps} (ep {epoch}): "
                f"train_loss={mean_window_loss:.4f} "
                f"forced_loss={metrics.get('val/forced_loss', float('nan')):.4f} "
                f"sample_seq={metrics.get('val/sample_seq_acc', float('nan')):.4f} "
                f"no_path={metrics.get('val/no_complete_path_rate', float('nan')):.3f} "
                f"eos_emit={metrics.get('val/eos_emitted_rate', float('nan')):.3f} "
                f"cot_valid={metrics.get('val/cot_chains_valid_rate', float('nan')):.3f} "
                f"cot_goal={metrics.get('val/cot_ends_at_goal_rate', float('nan')):.3f} "
                f"len={metrics.get('val/avg_response_len', float('nan')):.2f} "
                f"len_ok={metrics.get('val/avg_response_len_correct', float('nan')):.2f} "
                f"len_bad={metrics.get('val/avg_response_len_incorrect', float('nan')):.2f} "
                f"{tok_str}"
            )
            for k, s in enumerate(samples_log):
                log.info(
                    "  sample[%d] correct=%s eos_emitted=%s cot_valid=%s cot_goal=%s\n"
                    "    prefix:    %s\n"
                    "    gt_trace:  %s\n"
                    "    gen:       %s\n"
                    "    gt_path:   %s\n"
                    "    pred_path: %s",
                    k, s["correct"], s["eos_emitted"], s["cot_valid"], s["cot_ends_at_goal"],
                    s["prefix"], s["gt_trace"], s["gen"], s["gt_path"], s["pred_path"],
                )
            if cfg.logging.wandb:
                wandb.log(metrics, step=step)
                if samples_log:
                    table = wandb.Table(
                        columns=[
                            "step", "idx", "correct", "eos_emitted",
                            "cot_valid", "cot_ends_at_goal",
                            "gt_path", "pred_path", "gen", "gt_trace",
                        ]
                    )
                    for k, s in enumerate(samples_log):
                        table.add_data(
                            step, k, s["correct"], s["eos_emitted"],
                            s["cot_valid"], s["cot_ends_at_goal"],
                            s["gt_path"], s["pred_path"], s["gen"], s["gt_trace"],
                        )
                    wandb.log({"val/samples": table}, step=step)
        model.train()

    bar = tqdm(
        total=total_steps,
        desc="finetune",
        disable=not accelerator.is_main_process,
    )
    model.train()
    for epoch in range(cfg.schedule.epochs):
        for full_ids, lengths in train_loader:
            global_step += 1
            optimizer.zero_grad()
            input_ids, labels = make_cot_train_targets(full_ids, lengths, prefix_len)
            with accelerator.autocast():
                loss, _step_stats = model(input_ids, labels=labels)
            accelerator.backward(loss)
            clip_val = (
                cfg.optimizer.grad_clip
                if cfg.optimizer.grad_clip and cfg.optimizer.grad_clip > 0
                else float("inf")
            )
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), clip_val)
            grad_norm_val = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
            optimizer.step()
            scheduler.step()

            window_loss += loss.item()
            window_count += 1

            bar.update(1)
            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                grad_norm=f"{grad_norm_val:.2f}",
                lr=f"{scheduler.get_last_lr()[0]:.5f}",
                epoch=epoch,
            )

            if cfg.logging.wandb and accelerator.is_main_process:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/grad_norm": grad_norm_val,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/step": global_step,
                    },
                    step=global_step,
                )

            is_last_step = global_step == total_steps
            if global_step % eval_interval == 0 or is_last_step:
                run_eval(step=global_step, epoch=epoch)
                window_loss = 0.0
                window_count = 0
                if cfg.checkpoint.get("save_every_eval", False):
                    _save_checkpoint(
                        accelerator, model, cfg, meta, tokenizer,
                        block_size=block_size, prefix_len=prefix_len,
                        tag=f"step_{global_step}",
                    )

    bar.close()

    if cfg.checkpoint.get("save_final", True):
        _save_checkpoint(
            accelerator, model, cfg, meta, tokenizer,
            block_size=block_size, prefix_len=prefix_len, tag="final",
        )

    if cfg.logging.wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
