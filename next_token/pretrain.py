"""Hydra + Accelerate pre-trainer for the CoT-with-backtracking star-graph data.

A minimal vanilla-transformer trainer over variable-length CoT traces produced
by :mod:`next_token.generate_data_pretrain`. The train objective is masked
next-token cross-entropy on the trace tokens (everything after ``=``). Eval
samples ``meta.max_target_len`` tokens with temperature, scans for the
``</think>`` tag, and reads the next ``path_len`` tokens as the predicted path.

Run from the workspace root::

    python -m next_token.generate_data_pretrain --deg=5 --path_len=5 --num_nodes=50 \
        --n_train=2000000 --n_test=20000

    accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 \
        --num_processes=1 -m next_token.pretrain
"""

from __future__ import annotations

import json
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
    NumeralTokenizer,
    StarGraphCoTDataset,
    compute_lengths,
    cot_pad_collate,
    make_cot_train_targets,
)
from next_token.models import get_model  # noqa: E402

log = logging.getLogger(__name__)
_DATE_STR = datetime.now().strftime("%Y-%m-%d")


def _num_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _build_datasets(cfg: DictConfig):
    """Resolve a dataset folder, read its ``meta.json`` and build the datasets.

    All graph/CoT params (``deg``, ``path_len``, ``num_nodes``, ``cot.*``,
    ``max_target_len``) are read from ``<dataset>/meta.json`` -- the trainer
    config only needs to point at the folder.
    """
    data_dir = Path(cfg.data.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.cwd) / data_dir
    sub = data_dir / str(cfg.data.dataset)
    train_path = sub / "train.txt"
    test_path = sub / "test.txt"
    meta_path = sub / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing meta.json at {meta_path}. Generate the dataset first:\n"
            f"  python -m next_token.generate_data_pretrain --name={cfg.data.dataset} ..."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    deg = int(meta["deg"])
    path_len = int(meta["path_len"])
    num_nodes = int(meta["num_nodes"])
    cot = meta.get("cot", {})
    min_b = int(cot.get("min_backtracks", 0))
    max_b = int(cot.get("max_backtracks", deg - 1))
    min_d = int(cot.get("min_depth", path_len - 1))
    max_d = int(cot.get("max_depth", path_len - 1))

    log.info(
        "Dataset paths -- dir: %s | train: %s | test: %s "
        "(deg=%d path_len=%d num_nodes=%d cot=b%d-%d/d%d-%d "
        "n_train_cap=%s n_test_cap=%s max_target_len=%d)",
        sub, train_path, test_path,
        deg, path_len, num_nodes,
        min_b, max_b, min_d, max_d,
        cfg.data.get("n_train", None), cfg.data.get("n_test", None),
        meta["max_target_len"],
    )

    tokenizer = NumeralTokenizer(num_nodes=num_nodes)
    n_train_cap = cfg.data.get("n_train", None)
    n_test_cap = cfg.data.get("n_test", None)
    train_ds = StarGraphCoTDataset(
        train_path, tokenizer, deg=deg, path_len=path_len, num_nodes=num_nodes,
        n_samples=int(n_train_cap) if n_train_cap is not None else None,
    )
    test_ds = StarGraphCoTDataset(
        test_path, tokenizer, deg=deg, path_len=path_len, num_nodes=num_nodes,
        n_samples=int(n_test_cap) if n_test_cap is not None else None,
    )

    # Stash the resolved ints back so downstream code can read them.
    meta["_resolved"] = {
        "deg": deg, "path_len": path_len, "num_nodes": num_nodes,
        "min_b": min_b, "max_b": max_b, "min_d": min_d, "max_d": max_d,
    }
    return tokenizer, train_ds, test_ds, meta


def _sample_next(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    vocab_size: int | None = None,
) -> torch.Tensor:
    """Temperature (+ optional top-k) sampling on a (B, V) logit slice.

    If ``vocab_size`` is provided, positions ``>= vocab_size`` (padding from
    the model's padded lm-head) are masked out so we never sample undefined
    ids.
    """
    if vocab_size is not None and vocab_size < logits.size(-1):
        logits = logits.clone()
        logits[..., vocab_size:] = float("-inf")
    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1)
    scaled = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(int(top_k), scaled.size(-1))
        topk_vals, _ = scaled.topk(k, dim=-1)
        kth = topk_vals[:, -1].unsqueeze(-1)
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def _eval_cot(
    model,
    loader: DataLoader,
    accelerator: Accelerator,
    *,
    prefix_len: int,
    path_len: int,
    max_target_len: int,
    pad_id: int,
    think_close_id: int,
    eos_id: int,
    temperature: float,
    top_k: int | None,
    max_batches: int | None,
    tokenizer: NumeralTokenizer | None = None,
    log_samples: int = 0,
    vocab_size: int | None = None,
) -> tuple[dict[str, float], list[dict[str, str | bool]]]:
    """Sampling-based eval + teacher-forced loss diagnostic.

    For each test row:
      1. Take the prefix tokens (everything up to and including ``=``).
      2. AR-sample `max_target_len` tokens with temperature.
      3. Find first ``</think>`` in the generated tokens; the next `path_len`
         tokens are the predicted answer. Otherwise mark wrong.
      4. GT answer = ``full_ids[length - path_len : length]`` (last
         `path_len` un-padded tokens; by construction the second copy of the
         correct path).
    """
    model.eval()
    total_correct_seq = 0
    total_seq = 0
    total_no_close = 0
    total_eos_ok = 0
    total_correct_per_pos = torch.zeros(path_len, dtype=torch.long)
    total_per_pos = torch.zeros(path_len, dtype=torch.long)
    total_loss = 0.0
    total_loss_tokens = 0
    samples_log: list[dict[str, str | bool]] = []

    device = accelerator.device

    for i, (full_ids, lengths) in enumerate(
        tqdm(loader, desc="eval", leave=False, disable=not accelerator.is_main_process)
    ):
        if max_batches is not None and i >= max_batches:
            break
        B = full_ids.size(0)

        input_ids, labels = make_cot_train_targets(full_ids, lengths, prefix_len)
        with accelerator.autocast():
            loss, _ = model(input_ids, labels=labels)
        n_tok = (labels != -100).sum().item()
        total_loss += loss.item() * n_tok
        total_loss_tokens += n_tok

        full_ids_d = full_ids.to(device)
        lengths_d = lengths.to(device)

        prefix = full_ids_d[:, :prefix_len]
        seq = torch.cat(
            [prefix, torch.full((B, max_target_len), pad_id, dtype=torch.long, device=device)],
            dim=1,
        )

        # Per-row early-stop on <eos>: once a row emits EOS, the EOS itself is
        # kept (so `eos_ok` works) but subsequent positions are filled with
        # ``pad_id``. We also break out of the loop entirely once every row in
        # the batch is done, which saves model forwards on later positions.
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for t in range(max_target_len):
            with accelerator.autocast():
                logits, _ = model(seq[:, : prefix_len + t], labels=None)
            nxt = _sample_next(logits[:, -1, :], temperature, top_k, vocab_size)
            nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
            seq[:, prefix_len + t] = nxt
            done = done | (nxt == eos_id)
            if bool(done.all().item()):
                break

        gen = seq[:, prefix_len:]                                                # (B, max_target_len)

        # GT path: last path_len un-padded tokens of full_ids per row.
        ar = torch.arange(path_len, device=device)
        gather_idx = (lengths_d.unsqueeze(1) - path_len + ar).clamp_min(0)         # (B, path_len)
        gt = full_ids_d.gather(1, gather_idx)                                      # (B, path_len)

        # Predicted path: next path_len tokens after the first </think> in gen.
        # eos_ok: token immediately after that path slice is `<eos>`.
        pred = torch.full((B, path_len), pad_id, dtype=torch.long, device=device)
        valid = torch.zeros(B, dtype=torch.bool, device=device)
        eos_ok = torch.zeros(B, dtype=torch.bool, device=device)
        L_gen = gen.size(1)
        for b in range(B):
            row = gen[b]
            close_pos = (row == think_close_id).nonzero(as_tuple=False)
            if close_pos.numel() == 0:
                continue
            start = int(close_pos[0].item()) + 1
            end = start + path_len
            if end > L_gen:
                continue
            pred[b] = row[start:end]
            valid[b] = True
            if end < L_gen and row[end].item() == eos_id:
                eos_ok[b] = True

        correct = pred.eq(gt) & valid.unsqueeze(1)                                # (B, path_len)
        seq_ok = correct.all(dim=1) & valid

        correct_g, valid_g, seq_ok_g, eos_ok_g = accelerator.gather_for_metrics(
            (correct, valid, seq_ok, eos_ok)
        )
        total_correct_per_pos += correct_g.sum(dim=0).cpu().long()
        total_per_pos += correct_g.shape[0]
        total_correct_seq += seq_ok_g.sum().item()
        total_no_close += (~valid_g).sum().item()
        total_eos_ok += eos_ok_g.sum().item()
        total_seq += correct_g.shape[0]

        # Capture a few decoded samples from the very first batch (main process).
        if (
            accelerator.is_main_process
            and tokenizer is not None
            and log_samples > 0
            and len(samples_log) == 0
        ):
            n_log = min(int(log_samples), B)
            full_cpu = full_ids.cpu()
            lengths_cpu = lengths.cpu()
            gen_cpu = gen.cpu()
            pred_cpu = pred.cpu()
            gt_cpu = gt.cpu()
            valid_cpu = valid.cpu()
            seq_ok_cpu = seq_ok.cpu()
            eos_ok_cpu = eos_ok.cpu()
            for b in range(n_log):
                L_b = int(lengths_cpu[b].item())
                prefix_str = tokenizer.decode(full_cpu[b, :prefix_len].tolist())
                gt_trace_str = tokenizer.decode(full_cpu[b, prefix_len:L_b].tolist())
                gen_str = tokenizer.decode(gen_cpu[b].tolist())
                gt_path_str = tokenizer.decode(gt_cpu[b].tolist())
                pred_path_str = (
                    tokenizer.decode(pred_cpu[b].tolist())
                    if bool(valid_cpu[b].item())
                    else "<no </think>>"
                )
                samples_log.append(
                    {
                        "prefix": prefix_str,
                        "gt_trace": gt_trace_str,
                        "gen": gen_str,
                        "gt_path": gt_path_str,
                        "pred_path": pred_path_str,
                        "correct": bool(seq_ok_cpu[b].item()),
                        "eos_ok": bool(eos_ok_cpu[b].item()),
                    }
                )

    metrics: dict[str, float] = {}
    if total_seq > 0:
        metrics["val/sample_seq_acc"] = total_correct_seq / total_seq
        metrics["val/no_think_close_rate"] = total_no_close / total_seq
        metrics["val/eos_correct_pos_rate"] = total_eos_ok / total_seq
        for j in range(path_len):
            metrics[f"val/sample_token_{j}"] = (
                total_correct_per_pos[j].item() / max(1, total_per_pos[j].item())
            )
    if total_loss_tokens > 0:
        metrics["val/forced_loss"] = total_loss / total_loss_tokens
    return metrics, samples_log


@hydra.main(config_path="config", config_name="pretrain", version_base=None)
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
    block_size = prefix_len + max_target_len - 1

    if accelerator.is_main_process:
        log.info(
            f"Star-graph CoT: deg={deg} path_len={path_len} num_nodes={num_nodes}"
        )
        log.info(
            f"Tokens: prefix={prefix_len} max_target={max_target_len} "
            f"max_seq={prefix_len + max_target_len} (block_size={block_size}) | "
            f"vocab={tokenizer.vocab_size}"
        )
        log.info(
            f"Train samples: {len(train_ds):,} (max trace seen: {train_ds.max_target_len}) | "
            f"Test samples: {len(test_ds):,} (max trace seen: {test_ds.max_target_len})"
        )

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

    # ------------------------------------------------------------ model + opt
    model = get_model(cfg, vocab_size=tokenizer.vocab_size, block_size=block_size)
    if accelerator.is_main_process:
        log.info(f"Model: {cfg.model.name} | params: {_num_params(model):,}")
        print(model)

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
            f"pre_{cfg.model.name}_{cfg.data.dataset}"
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
            max_target_len=max_target_len,
            pad_id=tokenizer.DUMMY,
            think_close_id=tokenizer.THINK_CLOSE,
            eos_id=tokenizer.EOS,
            temperature=float(cfg.eval.temperature),
            top_k=cfg.eval.get("top_k", None),
            max_batches=cfg.eval.get("max_batches", None),
            tokenizer=tokenizer,
            log_samples=int(cfg.eval.get("log_samples", 0)),
            vocab_size=tokenizer.vocab_size,
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
                f"no_close={metrics.get('val/no_think_close_rate', float('nan')):.3f} "
                f"eos_pos={metrics.get('val/eos_correct_pos_rate', float('nan')):.3f} "
                f"{tok_str}"
            )
            for k, s in enumerate(samples_log):
                log.info(
                    "  sample[%d] correct=%s eos_ok=%s\n"
                    "    prefix:    %s\n"
                    "    gt_trace:  %s\n"
                    "    gen:       %s\n"
                    "    gt_path:   %s\n"
                    "    pred_path: %s",
                    k, s["correct"], s["eos_ok"],
                    s["prefix"], s["gt_trace"], s["gen"], s["gt_path"], s["pred_path"],
                )
            if cfg.logging.wandb:
                wandb.log(metrics, step=step)
                if samples_log:
                    table = wandb.Table(
                        columns=[
                            "step", "idx", "correct", "eos_ok",
                            "gt_path", "pred_path", "gen", "gt_trace",
                        ]
                    )
                    for k, s in enumerate(samples_log):
                        table.add_data(
                            step, k, s["correct"], s["eos_ok"],
                            s["gt_path"], s["pred_path"], s["gen"], s["gt_trace"],
                        )
                    wandb.log({"val/samples": table}, step=step)
        model.train()

    bar = tqdm(
        total=total_steps,
        desc="pretrain",
        disable=not accelerator.is_main_process,
    )
    model.train()
    for epoch in range(cfg.schedule.epochs):
        for full_ids, lengths in train_loader:
            global_step += 1
            optimizer.zero_grad()
            input_ids, labels = make_cot_train_targets(full_ids, lengths, prefix_len)
            with accelerator.autocast():
                loss, _ = model(input_ids, labels=labels)
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

    bar.close()

    if cfg.logging.wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
