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

    log.info(
        "Dataset paths -- dir: %s | train: %s | test: %s "
        "(deg=%d, path_len=%d, num_nodes=%d, reverse=%s, teacherless=%s, "
        "n_train=%d, n_test=%d)",
        sub,
        train_path,
        test_path,
        deg,
        path_len,
        num_nodes,
        reverse,
        cfg.data.teacherless,
        cfg.data.n_train,
        cfg.data.n_test,
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
        # NOTE: in teacherless mode, `input_ids[:, prefix_len:]` was overwritten
        # with `$` by the dataset, so we must read targets from `labels` (which
        # is unchanged in both modes) rather than from `full`.
        full = torch.cat([input_ids, labels[:, -1:].clone()], dim=1)             # (B, L)
        targets = labels[:, -target_len:]                                         # (B, target_len)
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
    if accelerator.is_main_process:
        log.info(f"Base model: {cfg.model.name} | params: {_num_params(model):,}")

    nextlat_enabled = (
        cfg.get("nextlat", None) is not None and cfg.nextlat.get("enabled", False)
    )
    mtp_enabled = (
        cfg.get("mtp", None) is not None and cfg.mtp.get("enabled", False)
    )
    if nextlat_enabled and mtp_enabled:
        raise ValueError(
            "nextlat.enabled and mtp.enabled are mutually exclusive; pick one."
        )

    if nextlat_enabled:
        from next_token.nextlat import build_nextlat

        horizon = cfg.nextlat.get("horizon", None)
        if horizon is None:
            # Path-star paper default (Sec 4.3): d = path_len - 2 so the goal
            # node sits within the multi-step horizon of the source.
            horizon = max(1, int(cfg.data.path_len) - 2)
        if accelerator.is_main_process:
            log.info(
                f"NextLat enabled: horizon={horizon} "
                f"lambda_h={cfg.nextlat.get('lambda_h', 1.0)} "
                f"lambda_kl={cfg.nextlat.get('lambda_kl', 1.0)} "
                f"n_hidden_layers={cfg.nextlat.get('n_hidden_layers', 2)} "
                f"hidden_mult={cfg.nextlat.get('hidden_mult', 4)} "
                f"stop_grad_target={cfg.nextlat.get('stop_grad_target', True)} "
                f"mask_kl={cfg.nextlat.get('mask_kl', True)}"
            )
        model = build_nextlat(
            model,
            vocab_size=tokenizer.vocab_size,
            cfg_nextlat=cfg.nextlat,
            horizon=horizon,
        )
        if accelerator.is_main_process:
            log.info(f"NextLat-wrapped params: {_num_params(model):,}")

    if mtp_enabled:
        from next_token.mtp import build_mtp

        mtp_horizon = cfg.mtp.get("horizon", None)
        if mtp_horizon is None:
            mtp_horizon = max(2, int(cfg.data.path_len) - 1)
        padded_vocab_size = int(model.config.padded_vocab_size)
        if accelerator.is_main_process:
            mtp_fused_raw = cfg.mtp.get("use_fused_ops", None)
            mtp_fused_resolved = (
                bool(getattr(model.config, "use_fused_ops", False))
                if mtp_fused_raw is None
                else bool(mtp_fused_raw)
            )
            log.info(
                f"MTP enabled: horizon={mtp_horizon} "
                f"n_layer={cfg.mtp.get('n_layer', 2)} "
                f"n_head={cfg.mtp.get('n_head', 6)} "
                f"lambda_mtp={cfg.mtp.get('lambda_mtp', 1.0)} "
                f"skip_depth_1={cfg.mtp.get('skip_depth_1', True)} "
                f"tie_wte={cfg.mtp.get('tie_wte', True)} "
                f"tie_lm_head={cfg.mtp.get('tie_lm_head', True)} "
                f"use_qk_norm={cfg.mtp.get('use_qk_norm', False)} "
                f"use_fused_ops={mtp_fused_resolved} (cfg={mtp_fused_raw}) "
                f"padded_vocab_size={padded_vocab_size}"
            )
        model = build_mtp(
            model,
            vocab_size=tokenizer.vocab_size,
            padded_vocab_size=padded_vocab_size,
            cfg_mtp=cfg.mtp,
            horizon=mtp_horizon,
        )
        if accelerator.is_main_process:
            log.info(f"MTP-wrapped params: {_num_params(model):,}")

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
    # LambdaLR returns a multiplier on the optimizer's base lr (= peak_lr),
    # so the floor we apply at the bottom of the cosine is min_lr / peak_lr.
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

    # RoPE cache lives on the model's device; must be set up after .prepare().
    accelerator.unwrap_model(model).setup_cache(device=accelerator.device)

    # ---------------------------------------------------------------------
    # wandb
    # ---------------------------------------------------------------------
    if cfg.logging.wandb and accelerator.is_main_process:
        # Base: <model>_deg{}_path{}_nodes{} [_rev] [_tless] _lr{}_seed{}
        # Optional suffixes -- only added when they deviate from defaults so
        # baseline runs keep short names but ablations are distinguishable.
        run_name = (
            f"{cfg.model.name}_deg{cfg.data.deg}_path{cfg.data.path_len}_"
            f"nodes{cfg.data.num_nodes}"
            f"{'_rev' if cfg.data.reverse else ''}"
            f"{'_tless' if cfg.data.teacherless else ''}"
            f"_lr{cfg.optimizer.lr:g}"
        )
        # Architecture overrides (only when non-default 6L/6H/384D).
        arch_bits = []
        if int(cfg.model.get("n_layer", 6)) != 6:
            arch_bits.append(f"L{cfg.model.n_layer}")
        if int(cfg.model.get("n_head", 6)) != 6:
            arch_bits.append(f"H{cfg.model.n_head}")
        if int(cfg.model.get("dim", 384)) != 384:
            arch_bits.append(f"D{cfg.model.dim}")
        if arch_bits:
            run_name += "_" + "".join(arch_bits)
        # Optimizer extras.
        if float(cfg.optimizer.get("weight_decay", 0.0)) > 0:
            run_name += f"_wd{cfg.optimizer.weight_decay:g}"
        if int(cfg.get("batch_size", 256)) != 256:
            run_name += f"_bs{cfg.batch_size}"
        # NextLat suffix.
        if cfg.get("nextlat", None) is not None and cfg.nextlat.get("enabled", False):
            nl_horizon = cfg.nextlat.get("horizon", None)
            nl_horizon = nl_horizon if nl_horizon is not None else max(1, int(cfg.data.path_len) - 2)
            run_name += (
                f"_nl"
                f"_h{nl_horizon}"
                f"_lh{cfg.nextlat.get('lambda_h', 1.0):g}"
                f"_lkl{cfg.nextlat.get('lambda_kl', 1.0):g}"
                f"_pL{cfg.nextlat.get('n_hidden_layers', 2)}"
                f"_pM{cfg.nextlat.get('hidden_mult', 4)}"
            )
        # MTP suffix.
        if cfg.get("mtp", None) is not None and cfg.mtp.get("enabled", False):
            m_horizon = cfg.mtp.get("horizon", None)
            m_horizon = m_horizon if m_horizon is not None else max(2, int(cfg.data.path_len) - 1)
            run_name += (
                f"_mtp"
                f"_h{m_horizon}"
                f"_lm{cfg.mtp.get('lambda_mtp', 1.0):g}"
                f"_L{cfg.mtp.get('n_layer', 2)}"
                f"_H{cfg.mtp.get('n_head', 6)}"
                f"_sd1{int(bool(cfg.mtp.get('skip_depth_1', True)))}"
                f"_tw{int(bool(cfg.mtp.get('tie_wte', True)))}"
                f"_tlm{int(bool(cfg.mtp.get('tie_lm_head', True)))}"
            )
        run_name += f"_seed{cfg.seed}"
        # User-supplied tag is prepended to the auto-built name.
        custom_name = cfg.logging.get("name", None)
        if custom_name:
            run_name = f"{custom_name} {run_name}"
        wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ---------------------------------------------------------------------
    # Train loop (single tqdm over total_steps)
    # ---------------------------------------------------------------------
    global_step = 0
    best_seq_acc = -1.0

    # Eval cadence: every `eval.every_pct` of total_steps (default 10%).
    eval_every_pct = float(cfg.eval.get("every_pct", 0.1))
    eval_interval = max(1, int(round(total_steps * eval_every_pct)))
    if accelerator.is_main_process:
        log.info(
            f"Eval cadence: every {eval_interval} steps "
            f"({eval_every_pct * 100:.1f}% of {total_steps} total)"
        )

    # Rolling-window train loss accumulated since the last eval.
    window_loss = 0.0
    window_count = 0

    def run_eval(step: int, epoch: int):
        nonlocal best_seq_acc
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
        mean_window_loss = window_loss / max(1, window_count)
        metrics["train/window_loss"] = mean_window_loss

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
                f"step {step}/{total_steps} (ep {epoch}): "
                f"train_loss={mean_window_loss:.4f} "
                f"forced_seq={metrics.get('val/forced_seq_acc', float('nan')):.4f} "
                f"{forced_str} "
                f"gen_seq={metrics.get('val/gen_seq_acc', float('nan')):.4f} "
                f"{gen_str}"
            )
            if cfg.logging.wandb:
                wandb.log(metrics, step=step)

        cur_seq_acc = metrics.get("val/gen_seq_acc", metrics.get("val/forced_seq_acc", 0.0))
        if cur_seq_acc > best_seq_acc:
            best_seq_acc = cur_seq_acc
            if cfg.checkpoint.save_on_improve:
                _save_checkpoint(accelerator, model, cfg, tag="best")

        model.train()  # _eval_* set model.eval(); restore train mode

    bar = tqdm(
        total=total_steps,
        desc="train",
        disable=not accelerator.is_main_process,
    )
    model.train()
    for epoch in range(cfg.schedule.epochs):
        for input_ids, labels in train_loader:
            global_step += 1
            optimizer.zero_grad()
            with accelerator.autocast():
                loss, step_stats = model(input_ids, labels=labels)
            accelerator.backward(loss)
            clip_val = cfg.optimizer.grad_clip if cfg.optimizer.grad_clip and cfg.optimizer.grad_clip > 0 else float("inf")
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), clip_val)
            grad_norm_val = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
            optimizer.step()
            scheduler.step()

            window_loss += loss.item()
            window_count += 1

            # Per-token train accuracy on the current batch (cheap teacher-forced
            # readout). Throttled by `eval.train_acc_every_steps` since it costs
            # one extra forward when the main path uses fused-linear-CE.
            train_acc_every = int(cfg.eval.get("train_acc_every_steps", 1))
            train_acc_stats: dict[str, float] = {}
            if train_acc_every > 0 and (global_step % train_acc_every == 0):
                with torch.no_grad():
                    with accelerator.autocast():
                        train_logits, _ = model(input_ids, labels=None)
                    preds_t = train_logits[:, -target_len:, :].argmax(dim=-1)
                    targets_t = labels[:, -target_len:]
                    correct_t = preds_t.eq(targets_t)                       # (B,T) bool
                per_pos = correct_t.float().mean(dim=0).cpu().tolist()
                seq_acc = correct_t.all(dim=1).float().mean().item()
                train_acc_stats["train/forced_seq_acc"] = seq_acc
                for j in range(target_len):
                    train_acc_stats[f"train/forced_token_{j}"] = per_pos[j]

            bar.update(1)
            postfix = dict(
                loss=f"{loss.item():.4f}",
                grad_norm=f"{grad_norm_val:.2f}",
                lr=f"{scheduler.get_last_lr()[0]:.5f}",
                epoch=epoch,
            )
            if step_stats and "nextlat/loss_next" in step_stats:
                postfix["nl_h"] = f"{step_stats['nextlat/loss_next_h']:.3f}"
                postfix["nl_kl"] = f"{step_stats['nextlat/loss_kl']:.3f}"
            if step_stats and "mtp/loss_mtp" in step_stats:
                postfix["mtp"] = f"{step_stats['mtp/loss_mtp']:.3f}"
                # Deepest depth: loss_d{H} (most informative far-horizon signal).
                deep_keys = [k for k in step_stats if k.startswith("mtp/loss_d")]
                if deep_keys:
                    deepest = max(deep_keys, key=lambda k: int(k.split("_d")[-1]))
                    depth = deepest.split("_d")[-1]
                    postfix[f"mtp_d{depth}"] = f"{step_stats[deepest]:.3f}"
            if train_acc_stats:
                # 0, 1, t-2, t-1 (matches the eval log)
                report_idx = sorted({0, 1, target_len - 2, target_len - 1})
                postfix["acc"] = "/".join(
                    f"{train_acc_stats[f'train/forced_token_{j}']:.2f}" for j in report_idx
                )
            bar.set_postfix(**postfix)

            if cfg.logging.wandb and accelerator.is_main_process:
                log_payload = {
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm_val,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/step": global_step,
                }
                if step_stats:
                    for k, v in step_stats.items():
                        if k.startswith("nextlat/") or k.startswith("mtp/"):
                            log_payload[f"train/{k}"] = v
                if train_acc_stats:
                    log_payload.update(train_acc_stats)
                wandb.log(log_payload, step=global_step)

            # Step-based eval
            is_last_step = global_step == total_steps
            if global_step % eval_interval == 0 or is_last_step:
                run_eval(step=global_step, epoch=epoch)
                window_loss = 0.0
                window_count = 0

    bar.close()

    if cfg.checkpoint.get("save_final", True):
        _save_checkpoint(accelerator, model, cfg, tag="final")

    if cfg.logging.wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
