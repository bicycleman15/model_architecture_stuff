"""Hydra + Accelerate trainer for the star-graph next-token task.

Mirrors `state_tracking/train.py`. Run from the workspace root::

    python -m next_token.generate_data --deg=2 --path_len=5 --num_nodes=50 \
        --n_train=200000 --n_test=20000

    accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 \
        --num_processes=1 -m next_token.train

Override anything via Hydra::

    -m next_token.train data=small schedule.epochs=10
    -m next_token.train data.reverse=true
    -m next_token.train data.teacherless=true
    -m next_token.train model.dim=512 model.n_layer=8 optimizer.lr=3e-4
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import hydra
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Make sure the workspace root is on sys.path so the importlib-based
# transformer wrapper can resolve any transitive imports.
_WS_ROOT = Path(__file__).resolve().parents[1]
if str(_WS_ROOT) not in sys.path:
    sys.path.insert(0, str(_WS_ROOT))

from next_token.data import (  # noqa: E402
    NumeralTokenizer,
    StarGraphDataset,
    compute_lengths,
)
from next_token.generate_data import dataset_dirname  # noqa: E402
from next_token.models import get_model  # noqa: E402

log = logging.getLogger(__name__)
_DATE_STR = datetime.now().strftime("%Y-%m-%d")


def _num_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _build_datasets(cfg: DictConfig):
    deg = cfg.data.deg
    path_len = cfg.data.path_len
    num_nodes = cfg.data.num_nodes
    reverse = cfg.data.reverse

    data_dir = Path(cfg.data.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.cwd) / data_dir
    sub = data_dir / dataset_dirname(deg, path_len, num_nodes, reverse)
    train_path = sub / "train.txt"
    test_path = sub / "test.txt"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Could not find {train_path} / {test_path}. "
            f"Run `python -m next_token.generate_data --deg={deg} --path_len={path_len} "
            f"--num_nodes={num_nodes} --n_train={cfg.data.n_train} --n_test={cfg.data.n_test} "
            f"--reverse={reverse}` first."
        )

    tokenizer = NumeralTokenizer(num_nodes=num_nodes)
    train_ds = StarGraphDataset(
        train_path,
        tokenizer=tokenizer,
        n_samples=cfg.data.n_train,
        teacherless=cfg.data.teacherless,
        eval_mode=False,
    )
    test_ds = StarGraphDataset(
        test_path,
        tokenizer=tokenizer,
        n_samples=cfg.data.n_test,
        teacherless=cfg.data.teacherless,
        eval_mode=False,
    )

    # All rows have the same length, so the batch tensors come out shaped
    # (B, L-1) directly via the default collate_fn.
    return tokenizer, train_ds, test_ds


@torch.no_grad()
def _eval_teacher_forced(
    model,
    loader: DataLoader,
    accelerator: Accelerator,
    target_len: int,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    total_correct_seq = 0
    total_seq = 0
    total_correct_per_pos = torch.zeros(target_len, dtype=torch.long)
    total_per_pos = torch.zeros(target_len, dtype=torch.long)
    total_loss = 0.0
    total_loss_tokens = 0

    for i, (input_ids, labels) in enumerate(
        tqdm(loader, desc="eval-forced", leave=False, disable=not accelerator.is_main_process)
    ):
        if max_batches is not None and i >= max_batches:
            break
        with accelerator.autocast():
            logits, _ = model(input_ids, labels=None)
            loss, _ = model(input_ids, labels=labels)

        # Per-position correctness over the last `target_len` predictions.
        preds = logits[:, -target_len:, :].argmax(dim=-1)            # (B, T)
        targets = labels[:, -target_len:]                            # (B, T)
        correct = preds.eq(targets)                                  # (B, T) bool

        correct_g, _ = accelerator.gather_for_metrics((correct, targets))
        total_correct_per_pos += correct_g.sum(dim=0).cpu().long()
        total_per_pos += correct_g.shape[0]
        total_correct_seq += correct_g.all(dim=1).sum().item()
        total_seq += correct_g.shape[0]

        n_tok = (labels != -100).sum().item()
        total_loss += loss.item() * n_tok
        total_loss_tokens += n_tok

    metrics: dict[str, float] = {}
    if total_seq > 0:
        metrics["val/forced_seq_acc"] = total_correct_seq / total_seq
        for j in range(target_len):
            metrics[f"val/forced_token_{j}"] = (
                total_correct_per_pos[j].item() / max(1, total_per_pos[j].item())
            )
    if total_loss_tokens > 0:
        metrics["val/forced_loss"] = total_loss / total_loss_tokens
    return metrics


@torch.no_grad()
def _eval_free_generation(
    model,
    loader: DataLoader,
    accelerator: Accelerator,
    prefix_len: int,
    target_len: int,
    teacherless: bool,
    dummy_id: int,
    max_batches: int | None,
) -> dict[str, float]:
    """Free-generation eval.

    * Standard mode (teacherless=False): greedy autoregressive decode.
      We start from the prefix and roll the model forward, feeding back its
      own prediction at each step so the prediction at step `t` depends on
      the predictions at steps `< t`.
    * Teacherless mode (teacherless=True): the model was trained with the
      target inputs replaced by '$', so "generation" is just a single forward
      where every target position is predicted in parallel from the dummies.
    """
    model.eval()
    total_correct_seq = 0
    total_seq = 0
    total_correct_per_pos = torch.zeros(target_len, dtype=torch.long)
    total_per_pos = torch.zeros(target_len, dtype=torch.long)

    for i, (input_ids, labels) in enumerate(
        tqdm(loader, desc="eval-gen", leave=False, disable=not accelerator.is_main_process)
    ):
        if max_batches is not None and i >= max_batches:
            break

        # Recover the full sequence: input_ids = full[:-1], labels[:, -1] is full[-1].
        full = torch.cat([input_ids, labels[:, -1:].clone()], dim=1)             # (B, L)
        targets = full[:, prefix_len:]                                            # (B, target_len)
        seq = full.clone()

        if teacherless:
            seq[:, prefix_len:] = dummy_id
            with accelerator.autocast():
                logits, _ = model(seq[:, :-1], labels=None)
            preds_full = logits[:, prefix_len - 1 : prefix_len - 1 + target_len, :].argmax(dim=-1)
        else:
            for t in range(target_len):
                with accelerator.autocast():
                    logits, _ = model(seq[:, :-1], labels=None)
                pred = logits[:, prefix_len - 1 + t, :].argmax(dim=-1)            # (B,)
                # Write into the next slot so step t+1 sees the model's own
                # prediction at position prefix_len + t.
                if t < target_len - 1:
                    seq[:, prefix_len + t] = pred
            preds_full = torch.cat(
                [seq[:, prefix_len : prefix_len + target_len - 1], pred.unsqueeze(1)],
                dim=1,
            )

        correct = preds_full.eq(targets)                                          # (B, T)
        correct_g, _ = accelerator.gather_for_metrics((correct, targets))
        total_correct_per_pos += correct_g.sum(dim=0).cpu().long()
        total_per_pos += correct_g.shape[0]
        total_correct_seq += correct_g.all(dim=1).sum().item()
        total_seq += correct_g.shape[0]

    metrics: dict[str, float] = {}
    if total_seq > 0:
        metrics["val/gen_seq_acc"] = total_correct_seq / total_seq
        for j in range(target_len):
            metrics[f"val/gen_token_{j}"] = (
                total_correct_per_pos[j].item() / max(1, total_per_pos[j].item())
            )
    return metrics


def _save_checkpoint(accelerator: Accelerator, model, cfg: DictConfig, tag: str):
    if not accelerator.is_main_process:
        return
    save_root = Path(cfg.checkpoint.save_dir)
    if not save_root.is_absolute():
        save_root = Path(hydra.core.hydra_config.HydraConfig.get().runtime.cwd) / save_root
    sub = (
        f"{cfg.model.name}_deg{cfg.data.deg}_path{cfg.data.path_len}_"
        f"nodes{cfg.data.num_nodes}{'_rev' if cfg.data.reverse else ''}"
        f"{'_tless' if cfg.data.teacherless else ''}_{_DATE_STR}_seed{cfg.seed}"
    )
    save_dir = save_root / sub
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    torch.save(unwrapped.state_dict(), save_dir / f"{tag}.pt")
    with open(save_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    log.info(f"Saved checkpoint -> {save_dir / f'{tag}.pt'}")


@hydra.main(config_path="config", config_name="star_graph", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)

    if accelerator.is_main_process:
        log.setLevel(logging.INFO)
        log.info("\n" + OmegaConf.to_yaml(cfg))

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------
    tokenizer, train_ds, test_ds = _build_datasets(cfg)
    prefix_len, target_len = compute_lengths(cfg.data.deg, cfg.data.path_len, cfg.data.num_nodes)
    assert prefix_len == train_ds.prefix_len
    assert target_len == train_ds.target_len
    block_size = train_ds.seq_len - 1  # input length after the shift

    if accelerator.is_main_process:
        log.info(
            f"Star graph: deg={cfg.data.deg} path_len={cfg.data.path_len} "
            f"num_nodes={cfg.data.num_nodes} reverse={cfg.data.reverse} "
            f"teacherless={cfg.data.teacherless}"
        )
        log.info(
            f"Tokens: prefix={prefix_len} target={target_len} "
            f"seq={train_ds.seq_len} (block_size={block_size}) | vocab={tokenizer.vocab_size}"
        )
        log.info(f"Train samples: {len(train_ds):,} | Test samples: {len(test_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # ---------------------------------------------------------------------
    # Model + optimizer
    # ---------------------------------------------------------------------
    model = get_model(cfg, vocab_size=tokenizer.vocab_size, block_size=block_size)
    print(model)
    if accelerator.is_main_process:
        log.info(f"Model: {cfg.model.name} | params: {_num_params(model):,}")

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

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if not cfg.schedule.use_cosine:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )

    # RoPE cache lives on the model's device; must be set up after .prepare().
    accelerator.unwrap_model(model).setup_cache(device=accelerator.device)

    # ---------------------------------------------------------------------
    # wandb
    # ---------------------------------------------------------------------
    if cfg.logging.wandb and accelerator.is_main_process:
        run_name = (
            f"{cfg.model.name}_deg{cfg.data.deg}_path{cfg.data.path_len}_"
            f"nodes{cfg.data.num_nodes}"
            f"{'_rev' if cfg.data.reverse else ''}"
            f"{'_tless' if cfg.data.teacherless else ''}"
            f"_lr{cfg.optimizer.lr:g}_seed{cfg.seed}"
        )
        wandb.init(
            project=cfg.project_name,
            entity=cfg.logging.entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ---------------------------------------------------------------------
    # Train loop (single tqdm over total_steps)
    # ---------------------------------------------------------------------
    global_step = 0
    best_seq_acc = -1.0

    bar = tqdm(
        total=total_steps,
        desc="train",
        disable=not accelerator.is_main_process,
    )
    for epoch in range(cfg.schedule.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for input_ids, labels in train_loader:
            global_step += 1
            optimizer.zero_grad()
            with accelerator.autocast():
                loss, _ = model(input_ids, labels=labels)
            accelerator.backward(loss)
            clip_val = cfg.optimizer.grad_clip if cfg.optimizer.grad_clip and cfg.optimizer.grad_clip > 0 else float("inf")
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), clip_val)
            grad_norm_val = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
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

        mean_train_loss = epoch_loss / max(1, n_batches)

        # -----------------------------------------------------------------
        # Eval
        # -----------------------------------------------------------------
        is_last = epoch == cfg.schedule.epochs - 1
        if (epoch + 1) % cfg.eval.every_n_epochs == 0 or is_last:
            forced = _eval_teacher_forced(
                model,
                test_loader,
                accelerator,
                target_len=target_len,
                max_batches=cfg.eval.get("max_batches", None),
            )
            metrics = dict(forced)
            if cfg.eval.get("free_generation", True):
                gen = _eval_free_generation(
                    model,
                    test_loader,
                    accelerator,
                    prefix_len=prefix_len,
                    target_len=target_len,
                    teacherless=cfg.data.teacherless,
                    dummy_id=tokenizer.DUMMY,
                    max_batches=cfg.eval.get("max_batches", None),
                )
                metrics.update(gen)
            metrics["train/epoch_loss"] = mean_train_loss

            if accelerator.is_main_process:
                # tok0 (source, trivial), tok1 (first hop, the hard one),
                # tok_{T-2} (second-last), tok_{T-1} (goal, trivial).
                report_idx = sorted({0, 1, target_len - 2, target_len - 1})
                forced_str = " ".join(
                    f"forced_t{j}={metrics.get(f'val/forced_token_{j}', float('nan')):.3f}"
                    for j in report_idx
                )
                gen_str = " ".join(
                    f"gen_t{j}={metrics.get(f'val/gen_token_{j}', float('nan')):.3f}"
                    for j in report_idx
                )
                log.info(
                    f"epoch {epoch}: train_loss={mean_train_loss:.4f} "
                    f"forced_seq={metrics.get('val/forced_seq_acc', float('nan')):.4f} "
                    f"{forced_str} "
                    f"gen_seq={metrics.get('val/gen_seq_acc', float('nan')):.4f} "
                    f"{gen_str}"
                )
                if cfg.logging.wandb:
                    wandb.log(metrics, step=global_step)

            cur_seq_acc = metrics.get("val/gen_seq_acc", metrics.get("val/forced_seq_acc", 0.0))
            if cur_seq_acc > best_seq_acc:
                best_seq_acc = cur_seq_acc
                if cfg.checkpoint.save_on_improve:
                    _save_checkpoint(accelerator, model, cfg, tag="best")

    bar.close()

    if cfg.checkpoint.get("save_final", True):
        _save_checkpoint(accelerator, model, cfg, tag="final")

    if cfg.logging.wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
