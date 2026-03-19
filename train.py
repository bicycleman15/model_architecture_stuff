import os
import time
from dataclasses import dataclass
from tqdm import tqdm

import torch
import torch.nn.functional as F
import wandb

torch._dynamo.config.optimize_ddp = False

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from hnet import HierarchicalLM, Config
from utils import CycleIterator, validate, get_lr, num_parameters, seed_everything, visualize_boundaries


@dataclass
class TrainConfig:
    # data
    data_dir: str = "data/tinystories-gpt4-clean"
    tokenizer_name: str = "bicycleman15/tinystories-gpt4-clean-tokenizer"

    # model
    block_size: int = 512
    vocab_size: int = 75
    dim: int = 768
    n_levels: int = 1

    batch_size: int = 128
    train_iters: int = 8000 # ~500M tokens
    grad_accum: int = 1
    grad_norm: float = 1.0
    seed: int = 42

    # optimizer
    lr: float = 6e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    warmup_iters: int = 500

    # eval
    eval_interval: int = 500
    eval_iters: int = 50

    # saving
    save_dir: str = "checkpoints"
    save_interval: int = 1000

    # wandb
    wandb_project: str = "hnet-tinystories"
    wandb_run_name: str = "test"


def build_config(vocab_size, block_size, n_levels, dim):
    """Recursively build a nested Config chain with n_levels of hierarchy."""
    proc_dim = (dim * 3) // 2

    if n_levels <= 1:
        return Config(
            vocab_size=vocab_size,
            block_size=block_size,
            dim=dim,
            processor_dim=proc_dim,
            processor_config=None,
        )

    inner_block_size = block_size # for now it matches the global block size

    inner = build_config(
        vocab_size=vocab_size,
        block_size=inner_block_size,
        n_levels=n_levels - 1,
        dim=proc_dim,
    )

    return Config(
        vocab_size=vocab_size,
        block_size=block_size,
        dim=dim,
        processor_dim=proc_dim,
        n_compressor_layers=3,
        n_processor_layers=6,
        n_decoder_layers=3,
        processor_config=inner,
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


def main():
    cfg = TrainConfig()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    seed_everything(cfg.seed)

    # data
    train_dataset = TokenDataset(os.path.join(cfg.data_dir, "train.pt"), cfg.block_size)
    test_dataset = TokenDataset(os.path.join(cfg.data_dir, "test.pt"), cfg.block_size)
    accelerator.print(f"Train tokens: {len(train_dataset.tokens):,} | Test tokens: {len(test_dataset.tokens):,}")

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, generator=g,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=True,
    )

    train_loader = accelerator.prepare_data_loader(train_loader)
    train_iterator = CycleIterator(train_loader)

    # tokenizer (for boundary visualization)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

    # model
    model_config = build_config(cfg.vocab_size, cfg.block_size, cfg.n_levels, cfg.dim)
    model = HierarchicalLM(model_config)

    accelerator.print(model)
    accelerator.print(model_config)

    accelerator.print("*****************************************************************")
    accelerator.print(f"Parameters: {num_parameters(model):,}")
    accelerator.print(f"Using #GPUs: {accelerator.num_processes}")
    accelerator.print(f"Using Mixed Precision: {accelerator.mixed_precision}")
    accelerator.print(f"Using block size: {cfg.block_size}")
    accelerator.print(f"Batch size on single device: {cfg.batch_size}")
    accelerator.print(f"Total effective batch size: {cfg.batch_size * accelerator.num_processes * cfg.grad_accum}")
    accelerator.print(f"Gradient Accumulation steps: {cfg.grad_accum}")
    accelerator.print(f"Train iters: {cfg.train_iters:,}")
    accelerator.print(f"Optimization steps: {cfg.train_iters // cfg.grad_accum:,}")
    accelerator.print(f"Tokens per iter: {cfg.batch_size * accelerator.num_processes * cfg.block_size:,}")
    accelerator.print(f"Tokens per optimization step: {cfg.batch_size * accelerator.num_processes * cfg.block_size * cfg.grad_accum:,}")
    accelerator.print(f"Total tokens in training: {cfg.train_iters * cfg.batch_size * accelerator.num_processes * cfg.block_size:,}")
    accelerator.print("*****************************************************************")

    # optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )

    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.unwrap_model(model).setup_cache(device=accelerator.device)

    # wandb (main process only)
    if accelerator.is_main_process:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=vars(cfg))

    accelerator.wait_for_everyone()

    # training loop
    bar = tqdm(range(cfg.train_iters), desc="Training", disable=(not accelerator.is_main_process))
    for step in bar:
        input_ids, targets = next(train_iterator)
        input_ids = input_ids.to(accelerator.device, non_blocking=True)
        targets = targets.to(accelerator.device, non_blocking=True)

        start_time = time.time()

        if step % cfg.grad_accum == 0:
            optimizer.zero_grad()

        lr = get_lr(cfg.lr, step, cfg.warmup_iters, cfg.train_iters, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with accelerator.autocast():
            logits, stats = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        accelerator.backward(loss / cfg.grad_accum)

        if (step + 1) % cfg.grad_accum == 0:
            if cfg.grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.grad_norm)
            optimizer.step()

            time_taken = time.time() - start_time
            token_throughput = input_ids.shape[0] * input_ids.shape[1] / time_taken / 1000

            if accelerator.is_main_process:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "perf/Ktokens_s": token_throughput,
                }
                for k, v in stats.items():
                    log_dict[f"train/{k}"] = v
                wandb.log(log_dict)

        chunks_str = " ".join(f"L{k.split('/')[0].split('_')[1]}={v:.1f}" for k, v in sorted(stats.items()))
        bar.set_postfix_str(f"loss={loss.item():.4f} lr={lr:.6f} {chunks_str}")

        # eval
        if step > 0 and step % cfg.eval_interval == 0:
            val_loss, val_ppl = validate(model, test_loader, accelerator.device, eval_iters=cfg.eval_iters)
            accelerator.print(f"\nStep {step}: val_loss={val_loss:.4f} ppl={val_ppl:.2f}")
            if accelerator.is_main_process:
                wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl})

            if accelerator.is_main_process:
                visualize_boundaries(accelerator.unwrap_model(model), test_loader, tokenizer, n=3)

        # save
        if step > 0 and step % cfg.save_interval == 0:
            if accelerator.is_main_process:
                os.makedirs(cfg.save_dir, exist_ok=True)
                path = os.path.join(cfg.save_dir, f"step_{step:07d}.pt")
                accelerator.save(accelerator.unwrap_model(model).state_dict(), path)
                accelerator.print(f"\nSaved checkpoint: {path}")
            accelerator.wait_for_everyone()

    # final save & eval
    if accelerator.is_main_process:
        os.makedirs(cfg.save_dir, exist_ok=True)
        accelerator.save(
            accelerator.unwrap_model(model).state_dict(),
            os.path.join(cfg.save_dir, "final.pt"),
        )

    val_loss, val_ppl = validate(model, test_loader, accelerator.device)
    accelerator.print(f"Final: val_loss={val_loss:.4f} ppl={val_ppl:.2f}")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl})
        wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
