import os
import time
from dataclasses import dataclass
from tqdm import tqdm

import torch
import torch.nn.functional as F
import wandb

from transformers import AutoTokenizer

from hnet import HierarchicalLM, Config
from utils import CycleIterator, validate, get_lr, num_parameters, seed_everything, visualize_boundaries


@dataclass
class TrainConfig:
    # data
    data_dir: str = "data/tinystories-gpt4-clean"
    tokenizer_name: str = "bicycleman15/tinystories-gpt4-clean-tokenizer"

    # model
    block_size: int = 128
    vocab_size: int = 75

    # training  (64 * 512 = 32,768 tokens/step → 3,052 steps ≈ 100M tokens)
    batch_size: int = 64
    train_iters: int = 3_052
    grad_accum: int = 1
    grad_norm: float = 1.0
    seed: int = 42

    # optimizer
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    warmup_iters: int = 100

    # eval
    eval_interval: int = 300
    eval_iters: int = 100

    # saving
    save_dir: str = "checkpoints"
    save_interval: int = 1000

    # wandb
    wandb_project: str = "hnet-tinystories"
    wandb_run_name: str = "hnet"


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
    seed_everything(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # data
    train_dataset = TokenDataset(os.path.join(cfg.data_dir, "train.pt"), cfg.block_size)
    test_dataset = TokenDataset(os.path.join(cfg.data_dir, "test.pt"), cfg.block_size)

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, generator=g,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=True,
    )
    train_iterator = CycleIterator(train_loader)

    # tokenizer (for boundary visualization)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

    # model
    model_config = Config(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
    )
    model = HierarchicalLM(model_config).to(device)
    model.setup_cache(device=device)
    print(model)

    print(model_config)
    print(f"Parameters: {num_parameters(model):,}")

    # optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )

    # wandb
    wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=vars(cfg))

    # training loop
    bar = tqdm(range(cfg.train_iters), desc="Training")
    for step in bar:
        input_ids, targets = next(train_iterator)
        input_ids, targets = input_ids.to(device), targets.to(device)

        if step % cfg.grad_accum == 0:
            optimizer.zero_grad()

        lr = get_lr(cfg.lr, step, cfg.warmup_iters, cfg.train_iters, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        logits, avg_chunk_size = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        (loss / cfg.grad_accum).backward()

        if (step + 1) % cfg.grad_accum == 0:
            if cfg.grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm)
            optimizer.step()

        bar.set_postfix_str(f"loss={loss.item():.4f} lr={lr:.6f} avg_chunk_size={avg_chunk_size:.1f}")
        wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/avg_chunk_size": avg_chunk_size})

        # if step % 25 == 0:
        #     visualize_boundaries(model, input_ids, tokenizer)

        # eval
        if step > 0 and step % cfg.eval_interval == 0:
            val_loss, val_ppl = validate(model, test_loader, device)
            print(f"\nStep {step}: val_loss={val_loss:.4f} ppl={val_ppl:.2f}")
            wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl})

        # save
        if step > 0 and step % cfg.save_interval == 0:
            os.makedirs(cfg.save_dir, exist_ok=True)
            path = os.path.join(cfg.save_dir, f"step_{step:07d}.pt")
            torch.save(model.state_dict(), path)
            print(f"\nSaved checkpoint: {path}")

    # final save & eval
    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "final.pt"))

    val_loss, val_ppl = validate(model, test_loader, device)
    print(f"Final: val_loss={val_loss:.4f} ppl={val_ppl:.2f}")
    wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl})
    wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
