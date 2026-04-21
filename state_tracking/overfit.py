"""One-batch overfitting sanity check for the M2RNN baseline.

Grabs one real training batch, freezes it, and repeatedly updates the
model on that exact batch. A well-wired model should drive loss to ~0
and token accuracy to ~1.0 within a few hundred steps; if it doesn't,
something is miswired (autograd path broken, grads not flowing, etc.).

Invocation mirrors `state_tracking.train`:

    python -m state_tracking.overfit \\
        model=rnn data=s3_128 batch_size=32 \\
        overfit.steps=500 overfit.log_every=25

Any `state_tracking.yaml` override works (e.g. `model.backend=torch`).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf

_WS_ROOT = Path(__file__).resolve().parents[1]
if str(_WS_ROOT) not in sys.path:
    sys.path.insert(0, str(_WS_ROOT))

from state_tracking.data_module import build_dataloaders  # noqa: E402
from state_tracking.metrics import token_accuracy  # noqa: E402
from state_tracking.models import get_model  # noqa: E402

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="state_tracking", version_base=None)
def main(cfg: DictConfig) -> None:
    # Extra knobs for this script; fall back to sensible defaults if they
    # aren't in the base config.
    steps: int = int(cfg.get("overfit", {}).get("steps", 500))
    log_every: int = int(cfg.get("overfit", {}).get("log_every", 25))
    lr: float = float(cfg.get("overfit", {}).get("lr", cfg.optimizer.lr))

    set_seed(cfg.seed)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    if accelerator.is_main_process:
        log.setLevel(logging.INFO)
        log.info(
            "overfit sanity check | model=%s backend=%s steps=%d lr=%g",
            cfg.model.name,
            cfg.model.get("backend", "default"),
            steps,
            lr,
        )

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
    pad_id = tokenizer.pad_token_id

    model = get_model(cfg, vocab_size=dm["n_vocab"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
        weight_decay=0.0,  # no wd - we *want* to overfit
    )

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, dm["train_loader"]
    )

    # Grab one batch and pin it on-device. No shuffling, no re-fetching.
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    if accelerator.is_main_process:
        log.info(
            "frozen batch shapes: input_ids=%s labels=%s",
            tuple(input_ids.shape),
            tuple(labels.shape),
        )
        log.info("params: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)

        with accelerator.autocast():
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=pad_id,
            )

        accelerator.backward(loss)

        # Measure grad norm BEFORE clipping so we can tell apart
        # "grad flow is healthy but large" vs. "grads are zero".
        is_log_step = step == 1 or step % log_every == 0 or step == steps
        if is_log_step:
            with torch.no_grad():
                grad_sq = torch.zeros(1, device=accelerator.device)
                for p in model.parameters():
                    if p.grad is not None:
                        grad_sq += p.grad.detach().float().pow(2).sum()
                grad_norm_preclip = float(grad_sq.sqrt().item())

        if cfg.optimizer.grad_clip and cfg.optimizer.grad_clip > 0:
            accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
        optimizer.step()

        if is_log_step:
            with torch.no_grad():
                acc = token_accuracy(logits.detach(), labels, pad_id)
            if accelerator.is_main_process:
                log.info(
                    "step %4d | loss=%.4f | token_acc=%.4f | grad_norm=%.4g",
                    step,
                    loss.item(),
                    float(acc),
                    grad_norm_preclip,
                )

    if accelerator.is_main_process:
        log.info("done. If loss didn't collapse towards 0, something is wrong.")


if __name__ == "__main__":
    main()
