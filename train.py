import math
import os
import time
import datetime
from tqdm import tqdm

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn.functional as F
import wandb

torch._dynamo.config.optimize_ddp = False

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from models.utils import get_model
from utils import (
    CycleIterator,
    validate,
    get_lr,
    num_parameters,
    seed_everything,
    visualize_boundaries,
    get_experiment_name,
    create_results_dir,
    build_bytes_per_token,
)


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, block_size):
        self.tokens = torch.load(data_path, weights_only=True)
        self.block_size = block_size

    def __len__(self):
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = self.tokens[start : start + self.block_size]
        y = self.tokens[start + 1 : start + self.block_size + 1]
        return x.long(), y.long()


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

    # data
    block_size = cfg.model.block_size
    train_dataset = TokenDataset(os.path.join(cfg.dataset.path, "train.pt"), block_size)
    test_dataset = TokenDataset(os.path.join(cfg.dataset.path, "test.pt"), block_size)

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True, pin_memory=True, generator=g,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.train.batch_size, shuffle=False, pin_memory=True,
    )

    train_loader = accelerator.prepare_data_loader(train_loader)
    train_iterator = CycleIterator(train_loader)

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

    accelerator.print("*****************************************************************")
    accelerator.print(f"Train tokens: {len(train_dataset.tokens):,} | Test tokens: {len(test_dataset.tokens):,}")
    accelerator.print(f"Parameters: {num_parameters(model):,}")
    accelerator.print(f"Using #GPUs: {accelerator.num_processes}")
    accelerator.print(f"Using Mixed Precision: {accelerator.mixed_precision}")
    accelerator.print(f"Using Model type: {cfg.model.name}")
    accelerator.print()

    accelerator.print(f"Using block size: {block_size}")
    accelerator.print(f"Batch size on single device: {cfg.train.batch_size}")
    accelerator.print(f"Total effective batch size: {cfg.train.batch_size * accelerator.num_processes * cfg.train.grad_accum}")
    accelerator.print(f"Gradient Accumulation steps: {cfg.train.grad_accum}")
    accelerator.print()

    # determine total training iterations
    if cfg.train.train_epochs > -1:
        train_iters = cfg.train.train_epochs * len(train_loader)
        accelerator.print(f"Using epoch-based training: {cfg.train.train_epochs} epochs x {len(train_loader)} iters/epoch = {train_iters:,} iters")
    else:
        train_iters = cfg.train.train_iters
        accelerator.print(f"Using iter-based training: {train_iters:,} iters")
    accelerator.print()

    accelerator.print(f"Train iters: {train_iters:,}")
    accelerator.print(f"Optimization steps: {train_iters // cfg.train.grad_accum:,}")
    accelerator.print(f"Tokens per iter: {cfg.train.batch_size * accelerator.num_processes * block_size:,}")
    accelerator.print(f"Tokens per optimization step: {cfg.train.batch_size * accelerator.num_processes * block_size * cfg.train.grad_accum:,}")
    accelerator.print(f"Total tokens in training: {train_iters * cfg.train.batch_size * accelerator.num_processes * block_size:,}")
    accelerator.print()

    if cfg.train.grad_norm > 0:
        accelerator.print(f"Using gradient clipping: {cfg.train.grad_norm}")
    else:
        accelerator.print("Not using gradient clipping")

    if cfg.train.train_epochs > -1:
        warmup_steps = int(train_iters * 0.1)
        accelerator.print(f"Warmup iters (10% of {train_iters}): {warmup_steps}")
    elif cfg.train.warmup_steps > 0:
        warmup_steps = cfg.train.warmup_steps
        accelerator.print(f"Warmup iters: {warmup_steps}")
    else:
        warmup_steps = int(train_iters * cfg.train.warmup_steps_percentage)
        accelerator.print(f"Warmup iters ({cfg.train.warmup_steps_percentage} * {train_iters}): {warmup_steps}")

    accelerator.print("*****************************************************************")

    # optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay, betas=cfg.optimizer.betas,
    )

    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.unwrap_model(model).setup_cache(device=accelerator.device)

    accelerator.wait_for_everyone()

    grad_accum = cfg.train.grad_accum

    # training loop
    bar = tqdm(range(train_iters), desc="Training", disable=(not accelerator.is_main_process))
    for step in bar:
        input_ids, targets = next(train_iterator)
        input_ids = input_ids.to(accelerator.device, non_blocking=True)
        targets = targets.to(accelerator.device, non_blocking=True)

        start_time = time.time()

        if step % grad_accum == 0:
            optimizer.zero_grad()

        lr = get_lr(cfg.optimizer.lr, step, warmup_steps, train_iters, cfg.optimizer.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with accelerator.autocast():
            logits, stats = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        accelerator.backward(loss / grad_accum)

        if (step + 1) % grad_accum == 0:
            if cfg.train.grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.train.grad_norm)
            optimizer.step()

            time_taken = time.time() - start_time
            token_throughput = input_ids.shape[0] * input_ids.shape[1] / time_taken / 1000

            if accelerator.is_main_process:
                num_tokens = targets.numel()
                num_bytes = bytes_per_token[targets].sum().item()
                train_bpb = loss.item() * num_tokens / num_bytes / math.log(2) if num_bytes > 0 else 0.0

                log_dict = {
                    "train/loss": loss.item(),
                    "train/bpb": train_bpb,
                    "train/lr": lr,
                    "perf/Ktokens_s": token_throughput,
                }
                for k, v in stats.items():
                    log_dict[f"train/{k}"] = v
                wandb.log(log_dict)

        _num_bytes = bytes_per_token[targets].sum().item()
        _bpb = loss.item() * targets.numel() / _num_bytes / math.log(2) if _num_bytes > 0 else 0.0
        chunks_str = " ".join(f"L{k.split('/')[0].split('_')[1]}={v:.1f}" for k, v in sorted(stats.items()))
        bar.set_postfix_str(f"loss={loss.item():.4f} bpb={_bpb:.4f} lr={lr:.6f} {chunks_str}")

        # eval
        if step > 0 and step % cfg.eval.eval_interval == 0:
            val_loss, val_ppl, val_bpb = validate(model, test_loader, accelerator.device, eval_iters=cfg.eval.eval_iters, bytes_per_token=bytes_per_token)
            accelerator.print(f"\nStep {step}: val_loss={val_loss:.4f} ppl={val_ppl:.2f} bpb={val_bpb:.4f}")
            if accelerator.is_main_process:
                wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl, "val/bpb": val_bpb})

            if accelerator.is_main_process:
                if cfg.model.name == "hourglass":
                    visualize_boundaries(accelerator.unwrap_model(model), test_loader, tokenizer, n=3)

        # save
        if step > 0 and step % cfg.train.save_interval == 0:
            if accelerator.is_main_process:
                save_dir = cfg.result_dir if cfg.result_dir else "checkpoints"
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, f"step_{step:07d}.pt")
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
