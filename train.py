import math
import os
import time
import datetime
from tqdm import tqdm

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import torch
import wandb

torch._dynamo.config.optimize_ddp = False

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from models.utils import get_model
from data_loader import ShardedDataLoader, TestDataset
from utils import (
    CycleIterator,
    validate,
    get_lr,
    group_params,
    num_parameters,
    seed_everything,
    visualize_boundaries,
    get_experiment_name,
    create_results_dir,
    build_bytes_per_token,
)


@hydra.main(config_path="config", config_name="byte", version_base=None)
def main(cfg: DictConfig):

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    seed_everything(cfg.seed)

    if accelerator.is_main_process:
        datetime_str = str(datetime.datetime.now())
        experiment_name = get_experiment_name(cfg, datetime_str)
        result_dir = create_results_dir(cfg, datetime_str)
        cfg.result_dir = result_dir

        OmegaConf.save(HydraConfig.get().overrides.task, os.path.join(result_dir, "overrides.yaml"))
        OmegaConf.save(cfg, os.path.join(result_dir, "config.yaml"))
        OmegaConf.save(cfg.model, os.path.join(result_dir, "model.yaml"))

        wandb.init(
            project=cfg.wandb.project,
            name=experiment_name,
            dir=result_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val/bpb", summary="min")

    accelerator.print(OmegaConf.to_container(cfg, resolve=True), "\n")

    # batch size / grad accum
    block_size = cfg.model.block_size
    batch_size = cfg.train.batch_size
    global_batch_size = cfg.train.global_batch_size
    num_processes = accelerator.num_processes

    assert global_batch_size % (batch_size * num_processes) == 0, (
        f"global_batch_size ({global_batch_size}) must be divisible by "
        f"batch_size * num_processes ({batch_size} * {num_processes} = {batch_size * num_processes})"
    )
    grad_accum = global_batch_size // (batch_size * num_processes)

    # data
    train_loader = ShardedDataLoader(
        data_root=cfg.dataset.path,
        block_size=block_size,
        batch_size=batch_size,
        split="train",
        process_rank=accelerator.process_index,
        num_processes=num_processes,
        seed=cfg.seed,
    )
    train_iterator = CycleIterator(train_loader)

    test_dataset = TestDataset(os.path.join(cfg.dataset.path, "test.npy"), block_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
    )

    if cfg.dataset.tokenizer_name == "byte":
        from byte_tokenizer import ByteTokenizer
        tokenizer = ByteTokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.tokenizer_name)

    accelerator.print(f"Building bytes-per-token table ...")
    bytes_per_token = build_bytes_per_token(tokenizer).to(accelerator.device)
    accelerator.print(f"Built bytes-per-token table (vocab_size={len(bytes_per_token)}, "
                       f"mean={bytes_per_token.mean():.2f} bytes/token)")

    # model
    model_config, model = get_model(cfg)

    accelerator.print(model)
    accelerator.print(model_config)

    train_steps = cfg.train.train_steps
    train_iters = train_steps * grad_accum

    accelerator.print("*****************************************************************")
    n_shards = len(train_loader.dataset.shards)
    accelerator.print(f"Train shards: {n_shards} | Test sequences: {len(test_dataset):,}")
    accelerator.print(f"Parameters: {num_parameters(model):,}")
    accelerator.print(f"Using #GPUs: {num_processes}")
    accelerator.print(f"Using Mixed Precision: {accelerator.mixed_precision}")
    accelerator.print(f"Using Model type: {cfg.model.name}")
    accelerator.print()

    accelerator.print(f"Block size: {block_size}")
    accelerator.print(f"Per-GPU batch size: {batch_size}")
    accelerator.print(f"Global batch size: {global_batch_size}")
    accelerator.print(f"Gradient accumulation steps: {grad_accum}")
    accelerator.print()

    tokens_per_step = global_batch_size * block_size
    accelerator.print(f"Train steps: {train_steps:,}")
    accelerator.print(f"Train iters (steps x grad_accum): {train_iters:,}")
    accelerator.print(f"Tokens per step: {tokens_per_step:,}")
    accelerator.print(f"Total tokens in training: {train_steps * tokens_per_step:,}")
    accelerator.print()

    if cfg.train.grad_norm > 0:
        accelerator.print(f"Gradient clipping: {cfg.train.grad_norm}")
    else:
        accelerator.print("Not using gradient clipping")

    if cfg.train.warmup_steps >= 0:
        warmup_steps = cfg.train.warmup_steps
        accelerator.print(f"Warmup steps: {warmup_steps}")
    else:
        warmup_steps = int(train_steps * cfg.train.warmup_steps_percentage)
        accelerator.print(f"Warmup steps ({cfg.train.warmup_steps_percentage} * {train_steps}): {warmup_steps}")

    accelerator.print("*****************************************************************")

    # optimizer
    param_groups = group_params(model, weight_decay=cfg.optimizer.weight_decay)
    accelerator.print(f"Optimizer param groups: {len(param_groups)}")
    for i, pg in enumerate(param_groups):
        n_params = sum(p.numel() for p in pg["params"])
        accelerator.print(f"  group {i}: {n_params:,} params, lr_mult={pg.get('lr_multiplier', 1.0)}, wd={pg.get('weight_decay', cfg.optimizer.weight_decay)}")
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.optimizer.lr, betas=cfg.optimizer.betas,
    )

    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.unwrap_model(model).setup_cache(device=accelerator.device)

    accelerator.wait_for_everyone()

    # training loop
    opt_step = 0
    bar = tqdm(range(train_iters), desc="Training", disable=(not accelerator.is_main_process))
    for it in bar:
        input_ids, targets = next(train_iterator)
        input_ids = input_ids.to(accelerator.device, non_blocking=True)
        targets = targets.to(accelerator.device, non_blocking=True)

        start_time = time.time()

        if it % grad_accum == 0:
            optimizer.zero_grad()

        lr = get_lr(cfg.optimizer.lr, opt_step, warmup_steps, train_steps, cfg.optimizer.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr * pg.get("lr_multiplier", 1.0)

        with accelerator.autocast():
            loss, stats = model(input_ids, labels=targets)

        accelerator.backward(loss / grad_accum)

        if (it + 1) % grad_accum == 0:
            if cfg.train.grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.train.grad_norm)
            optimizer.step()
            opt_step += 1

            time_taken = time.time() - start_time
            token_throughput = input_ids.shape[0] * input_ids.shape[1] / time_taken / 1000

            if accelerator.is_main_process:
                num_tokens = targets.numel()
                num_bytes = bytes_per_token[targets].sum().item()
                ce_for_bpb = stats.get("reinforce/ce_loss", stats.get("hnet/ce_loss", loss.item()))
                train_bpb = ce_for_bpb * num_tokens / num_bytes / math.log(2) if num_bytes > 0 else 0.0

                log_dict = {
                    "train/loss": loss.item(),
                    "train/bpb": train_bpb,
                    "train/lr": lr,
                    "train/step": opt_step,
                    "perf/Ktokens_s": token_throughput,
                }
                for k, v in stats.items():
                    log_dict[f"stat/{k}"] = v
                wandb.log(log_dict)

            _num_bytes = bytes_per_token[targets].sum().item()
            _ce = stats.get("reinforce/ce_loss", stats.get("hnet/ce_loss", loss.item()))
            _bpb = _ce * targets.numel() / _num_bytes / math.log(2) if _num_bytes > 0 else 0.0
            stats_str = " ".join(f"{k}={v:.2f}" for k, v in sorted(stats.items()))
            bar.set_postfix_str(f"loss={loss.item():.4f} bpb={_bpb:.4f} lr={lr:.6f} step={opt_step} {stats_str}")

        # eval (on optimizer steps)
        if opt_step > 0 and opt_step % cfg.eval.eval_interval == 0 and (it + 1) % grad_accum == 0:
            val_loss, val_ppl, val_bpb = validate(model, test_loader, accelerator.device, eval_iters=cfg.eval.eval_iters, bytes_per_token=bytes_per_token)
            accelerator.print(f"\nStep {opt_step}: val_loss={val_loss:.4f} ppl={val_ppl:.2f} bpb={val_bpb:.4f}")
            if accelerator.is_main_process:
                wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl, "val/bpb": val_bpb})

            if accelerator.is_main_process:
                if cfg.model.name in ("hourglass", "reinforce_hourglass", "hnet"):
                    visualize_boundaries(accelerator.unwrap_model(model), test_loader, tokenizer, n=3)

        # save (on optimizer steps)
        if opt_step > 0 and opt_step % cfg.train.save_interval == 0 and (it + 1) % grad_accum == 0:
            if accelerator.is_main_process:
                save_dir = cfg.result_dir if cfg.result_dir else "checkpoints"
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, f"step_{opt_step:07d}.pt")
                accelerator.save(accelerator.unwrap_model(model).state_dict(), path)
                accelerator.print(f"\nSaved checkpoint: {path}")
            accelerator.wait_for_everyone()

    # final save & eval
    if accelerator.is_main_process:
        save_dir = cfg.result_dir if cfg.result_dir else "checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, "state_dict.pt")
        accelerator.save(
            accelerator.unwrap_model(model).state_dict(),
            model_save_path,
        )
        accelerator.print(f"Training complete. Saved model at: {model_save_path}")

    val_loss, val_ppl, val_bpb = validate(model, test_loader, accelerator.device, eval_iters=cfg.eval.eval_iters, bytes_per_token=bytes_per_token)
    accelerator.print(f"Final: val_loss={val_loss:.4f} ppl={val_ppl:.2f} bpb={val_bpb:.4f}")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl, "val/bpb": val_bpb})
        wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
