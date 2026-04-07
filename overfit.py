"""
Single-batch overfitting experiment.

Grabs one batch from the sharded training data and repeatedly trains on it
until the loss (hopefully) drops to near zero. Useful as a sanity check that
the model can actually learn.

Usage:
    python overfit.py                                          # defaults (transformer)
    python overfit.py model_type=mean_residual_transformer     # mean residual variant
    python overfit.py overfit_steps=1000 optimizer.lr=1e-3     # custom steps / lr
    python overfit.py train.batch_size=8                       # smaller batch
"""

import time
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import wandb

from models.utils import get_model
from data_loader import ShardedDataLoader
from utils import seed_everything, num_parameters


@hydra.main(config_path="config", config_name="residual/fineweb", version_base=None)
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(cfg.seed)

    overfit_steps = cfg.train.train_steps
    log_interval = cfg.get("overfit_log_interval", 10)

    block_size = cfg.model.block_size
    batch_size = cfg.train.batch_size
    global_batch_size = cfg.train.global_batch_size

    assert global_batch_size % batch_size == 0, (
        f"global_batch_size ({global_batch_size}) must be divisible by "
        f"batch_size ({batch_size})"
    )
    grad_accum = global_batch_size // batch_size
    train_iters = overfit_steps * grad_accum

    print(OmegaConf.to_yaml(cfg))
    print(f"Per-device batch size: {batch_size}")
    print(f"Global batch size: {global_batch_size}")
    print(f"Gradient accumulation steps: {grad_accum}")
    print(f"Optimizer steps: {overfit_steps}")
    print(f"Total iterations: {train_iters}")

    # --- data: grab grad_accum micro-batches that form one global batch ---
    loader = ShardedDataLoader(
        data_root=cfg.dataset.path,
        block_size=cfg.model.block_size,
        batch_size=cfg.train.batch_size,
        split="train",
    )
    loader_iter = iter(loader)
    micro_batches = []
    for _ in range(grad_accum):
        xi, yi = next(loader_iter)
        micro_batches.append((xi.to(device), yi.to(device)))
    print(f"Loaded {grad_accum} micro-batches, each shape: x={list(micro_batches[0][0].shape)}, y={list(micro_batches[0][1].shape)}")

    # --- model ---
    model_config, model = get_model(cfg)
    model.to(device)
    model.setup_cache(device=device)
    n_params = num_parameters(model)
    print(f"Model: {cfg.model.name} | Parameters: {n_params:,}")
    print(model_config)

    # --- wandb ---
    wandb.init(
        project=cfg.wandb.project,
        name=f"overfit-{cfg.model.name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=["overfit"],
    )

    # --- optimizer (constant lr, no schedule) ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.betas),
        weight_decay=cfg.optimizer.weight_decay,
    )

    # --- overfit loop ---
    model.train()
    opt_step = 0
    bar = tqdm(total=overfit_steps, desc="Overfitting")
    for it in range(train_iters):
        t0 = time.time()

        if it % grad_accum == 0:
            optimizer.zero_grad()

        x, y = micro_batches[it % grad_accum]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, stats = model(x, labels=y, log_norms=True)

        (loss / grad_accum).backward()

        if (it + 1) % grad_accum == 0:
            if cfg.train.grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_norm)

            optimizer.step()
            opt_step += 1
            dt = time.time() - t0

            loss_val = loss.item()
            bar.update(1)
            bar.set_postfix_str(f"loss={loss_val:.6f}")

            if opt_step % log_interval == 0:
                log_dict = {
                    "train/loss": loss_val,
                    "train/step": opt_step,
                    "perf/step_ms": dt * 1000,
                }
                for k, v in stats.items():
                    log_dict[f"stat/{k}"] = v
                wandb.log(log_dict)

    print(f"Final loss: {loss.item():.6f}")
    wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
