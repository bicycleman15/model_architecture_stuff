"""GRPO fine-tuner for the CoT star-graph model.

Loads a checkpoint produced by :mod:`next_token.pretrain` and performs Group
Relative Policy Optimization with sparse 0/1 reward = does the path emitted
after ``</think>`` match the ground-truth path?

Single inner update per rollout, so the standard GRPO objective collapses to
REINFORCE-with-group-baseline:

    L = mean_i mean_t [ -A_i * log pi(o_{i,t} | q, o_{i,<t}) + beta * KL_k3(pi || pi_ref) ]

where
    A_i = (r_i - mean(r_group)) / (std(r_group) + eps)

Beta=0 disables the KL term and skips the reference-model forward.

Run from the workspace root::

    accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 \
        --num_processes=1 -m next_token.grpo \
        init.ckpt_path=Results/.../ckpt/final.pt \
        data.dataset=star_3x5_5M
"""

from __future__ import annotations

import json
import logging
import math
import sys
from contextlib import nullcontext
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
)
from next_token.models import get_model  # noqa: E402
from next_token.pretrain import _eval_cot  # noqa: E402

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _num_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _resolve_dataset_dir(cfg: DictConfig) -> Path:
    data_dir = Path(cfg.data.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.cwd) / data_dir
    return data_dir / str(cfg.data.dataset)


def _build_datasets(cfg: DictConfig):
    """Same dataset setup as ``pretrain._build_datasets`` (folder + meta.json)."""
    sub = _resolve_dataset_dir(cfg)
    meta_path = sub / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing meta.json at {meta_path}. Generate the dataset first:\n"
            f"  python -m next_token.generate_data_pretrain --name={cfg.data.dataset} ..."
        )
    with open(meta_path, "r") as f:
        meta = json.load(f)
    deg = int(meta["deg"])
    path_len = int(meta["path_len"])
    num_nodes = int(meta["num_nodes"])
    cot = meta.get("cot", {})
    min_b = int(cot.get("min_backtracks", 0))
    max_b = int(cot.get("max_backtracks", 0))
    min_d = int(cot.get("min_decoy_depth", path_len - 1))
    max_d = int(cot.get("max_decoy_depth", path_len - 1))

    train_path = sub / "train.txt"
    test_path = sub / "test.txt"
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
    meta["_resolved"] = {
        "deg": deg, "path_len": path_len, "num_nodes": num_nodes,
        "min_b": min_b, "max_b": max_b, "min_d": min_d, "max_d": max_d,
    }
    return tokenizer, train_ds, test_ds, meta


def _load_ckpt(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    needed = {"state_dict", "vocab_size", "block_size"}
    missing = needed - set(payload.keys())
    if missing:
        raise ValueError(f"Checkpoint at {path} missing keys: {missing}")
    return payload


def _build_model_from_ckpt(
    cfg: DictConfig,
    ckpt: dict,
    device: torch.device,
    *,
    freeze: bool = False,
    strict: bool = True,
):
    """Instantiate via ``get_model`` and load the checkpoint state dict."""
    vocab_size = int(ckpt["vocab_size"])
    block_size = int(ckpt["block_size"])
    model = get_model(cfg, vocab_size=vocab_size, block_size=block_size)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=strict)
    if (missing or unexpected) and not strict:
        log.warning(
            f"load_state_dict (non-strict): missing={list(missing)} "
            f"unexpected={list(unexpected)}"
        )
    model.to(device)
    model.setup_cache(device=device)
    if freeze:
        model.requires_grad_(False)
        model.eval()
    return model, vocab_size, block_size


# ---------------------------------------------------------------------------
# Sampling + reward + log-prob helpers
# ---------------------------------------------------------------------------


def _sample_with_temp(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    vocab_size: int | None,
) -> torch.Tensor:
    """Stochastic sampling on (B, V) logits with optional top-k + vocab masking."""
    if vocab_size is not None and vocab_size < logits.size(-1):
        logits = logits.clone()
        logits[..., vocab_size:] = float("-inf")
    scaled = logits / max(1e-6, float(temperature))
    if top_k is not None and top_k > 0:
        k = min(int(top_k), scaled.size(-1))
        topk_vals, _ = scaled.topk(k, dim=-1)
        kth = topk_vals[:, -1].unsqueeze(-1)
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def _ar_sample(
    policy,
    prompt: torch.Tensor,
    *,
    max_new: int,
    vocab_size: int,
    temperature: float,
    top_k: int | None,
    eos_id: int,
    pad_id: int,
    autocast_ctx,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized AR rollout with per-row early-stop on ``<eos>``.

    Returns
    -------
    gen_ids : (B, max_new) long
        Sampled response tokens. After ``<eos>`` the row is filled with ``pad_id``.
    response_mask : (B, max_new) bool
        ``True`` for valid response positions (including the ``<eos>`` token);
        ``False`` once the row has emitted ``<eos>`` and for any padding.
    """
    B, prefix_len = prompt.shape
    device = prompt.device
    seq = torch.cat(
        [prompt, torch.full((B, max_new), pad_id, dtype=torch.long, device=device)],
        dim=1,
    )
    done = torch.zeros(B, dtype=torch.bool, device=device)
    response_mask = torch.zeros(B, max_new, dtype=torch.bool, device=device)
    for t in range(max_new):
        with autocast_ctx():
            logits, _ = policy(seq[:, : prefix_len + t], labels=None)
        nxt = _sample_with_temp(
            logits[:, -1, :].float(), temperature, top_k, vocab_size
        )
        nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
        response_mask[:, t] = ~done
        seq[:, prefix_len + t] = nxt
        done = done | (nxt == eos_id)
        if bool(done.all().item()):
            break
    gen_ids = seq[:, prefix_len:]
    return gen_ids, response_mask


def _compute_rewards(
    gen_ids: torch.Tensor,
    gt_paths: torch.Tensor,
    *,
    think_close_id: int,
    path_len: int,
) -> torch.Tensor:
    """Sparse 0/1 reward per rollout, vectorized across the batch.

    For each row, find the first ``</think>``; the next ``path_len`` tokens are
    the predicted answer. Reward = 1.0 iff the predicted path matches
    ``gt_paths`` exactly. If ``</think>`` is missing or the answer slice would
    overflow, reward = 0.0.
    """
    BG, L = gen_ids.shape
    device = gen_ids.device
    pos = torch.arange(L, device=device).unsqueeze(0).expand(BG, -1)
    big = torch.full_like(pos, L + 1)
    is_close = gen_ids == think_close_id
    first_close = torch.where(is_close, pos, big).min(dim=1).values
    valid = (first_close + path_len) < L
    start = (first_close + 1).clamp_max(max(0, L - path_len))
    ar = torch.arange(path_len, device=device).unsqueeze(0)
    idx = start.unsqueeze(1) + ar
    pred = gen_ids.gather(1, idx)
    correct = (pred == gt_paths).all(dim=1) & valid
    return correct.float()


def _apply_length_penalty(
    rewards: torch.Tensor,
    response_lens: torch.Tensor,
    group_size: int,
    alpha: float,
    eps_std: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Arora & Zanette (2025) length penalty on top of 0/1 correctness reward.

    Eq. 4 + 5 of https://arxiv.org/abs/2502.04463::

        r(x, y) = 1{correct} * (1 - alpha * sigmoid((len(y) - mu) / sigma))

    where ``mu``, ``sigma`` are computed *over correct rollouts* within the
    prompt's group. Incorrect rollouts always stay at reward 0; correct ones
    are squashed into ``[1 - alpha, 1]`` (since sigmoid is in (0, 1)).

    Returns ``(shaped_rewards, length_factor)`` both flattened to (BG,). The
    factor is ``f(len(y)) = sigmoid(...)`` and is logged for diagnostics; it is
    zero everywhere when ``alpha <= 0`` so callers can avoid extra branches.
    """
    if alpha <= 0.0:
        return rewards, torch.zeros_like(rewards)

    BG = rewards.shape[0]
    B = BG // group_size
    r = rewards.view(B, group_size)
    L = response_lens.view(B, group_size).float()

    correct = r > 0.5
    correct_f = correct.float()
    n_corr = correct_f.sum(dim=1, keepdim=True).clamp_min(1.0)        # (B, 1)
    sum_L = (L * correct_f).sum(dim=1, keepdim=True)
    mean = sum_L / n_corr                                              # mean over correct
    var = ((L - mean) ** 2 * correct_f).sum(dim=1, keepdim=True) / n_corr
    std = var.clamp_min(0.0).sqrt()

    f = torch.sigmoid((L - mean) / (std + eps_std))                    # (B, G)
    shaped = r * (1.0 - alpha * f)                                     # incorrect stay 0
    return shaped.view(BG), f.view(BG)


def _group_advantage(
    rewards: torch.Tensor,
    group_size: int,
    eps: float,
    *,
    mode: str = "grpo",
) -> torch.Tensor:
    """Group baseline subtraction.

    - ``mode='grpo'``: ``A_i = (r_i - mean(group)) / (std(group) + eps)``.
    - ``mode='rloo'``: leave-one-out baseline (Kool et al. 2019; Arora &
      Zanette 2025 Appendix J): ``A_i = r_i - mean(r_{j != i})
      = (G * r_i - sum(r)) / (G - 1)``. Undefined for ``G == 1``.
    """
    BG = rewards.shape[0]
    B = BG // group_size
    r = rewards.view(B, group_size)
    if mode == "grpo":
        mean = r.mean(dim=1, keepdim=True)
        std = r.std(dim=1, keepdim=True, unbiased=False)
        adv = (r - mean) / (std + eps)
    elif mode == "rloo":
        if group_size < 2:
            raise ValueError("RLOO advantage requires group_size >= 2")
        denom = group_size - 1
        adv = (group_size * r - r.sum(dim=1, keepdim=True)) / denom
    else:
        raise ValueError(f"Unknown advantage mode: {mode!r} (expected 'grpo' or 'rloo')")
    return adv.view(BG)


def _logprobs_at_response(
    model,
    full_seq: torch.Tensor,
    *,
    prefix_len: int,
    max_new: int,
    vocab_size: int,
    autocast_ctx,
    no_grad: bool = False,
    return_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Per-token log-prob of the actual sampled token at response positions.

    ``full_seq`` is ``(BG, prefix_len + max_new)``. We score the response
    region; for response position ``t`` (0-indexed, mapping to absolute
    position ``prefix_len + t``), the predictive distribution comes from the
    transformer's output at absolute position ``prefix_len + t - 1``.

    Returns
    -------
    log_probs : (BG, max_new) float
        ``log p(full_seq[:, prefix_len + t] | full_seq[:, < prefix_len + t])``.
        Caller is responsible for masking out invalid positions.
    entropy : (BG, max_new) float or None
        Per-position entropy of the policy at the same predictive positions
        (only when ``return_entropy=True``); always detached.
    """
    grad_ctx = torch.no_grad() if no_grad else nullcontext()
    with grad_ctx, autocast_ctx():
        logits, _ = model(full_seq, labels=None)
    if vocab_size is not None and vocab_size < logits.size(-1):
        logits = logits[..., :vocab_size]
    sl = logits[:, prefix_len - 1 : prefix_len - 1 + max_new, :].float()
    logp = torch.log_softmax(sl, dim=-1)
    targets = full_seq[:, prefix_len : prefix_len + max_new]
    log_probs = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    entropy = None
    if return_entropy:
        with torch.no_grad():
            p = logp.detach().exp()
            entropy = -(p * logp.detach()).sum(dim=-1)
    return log_probs, entropy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="config", config_name="grpo", version_base=None)
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

    max_new = cfg.grpo.get("max_new_tokens", None)
    if max_new is None or int(max_new) <= 0:
        max_new = max_target_len
    max_new = min(int(max_new), max_target_len)

    # ------------------------------------------------------------------ ckpt
    ckpt_path = Path(cfg.init.ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.cwd) / ckpt_path
    ckpt = _load_ckpt(ckpt_path)
    if accelerator.is_main_process:
        log.info(f"Loaded policy checkpoint: {ckpt_path}")

    if int(ckpt["vocab_size"]) != tokenizer.vocab_size:
        raise ValueError(
            f"vocab_size mismatch: ckpt={ckpt['vocab_size']} "
            f"tokenizer={tokenizer.vocab_size} (dataset num_nodes={num_nodes})"
        )

    block_size = int(ckpt["block_size"])
    # The log-prob forward operates on the full ``prefix + response`` sequence
    # (length ``prefix_len + max_new``) so we need the *full* length to fit
    # inside ``block_size``.
    if prefix_len + max_new > block_size:
        new_cap = block_size - prefix_len
        if accelerator.is_main_process:
            log.warning(
                f"max_new={max_new} would exceed block_size={block_size} "
                f"(prefix_len={prefix_len}); capping to {new_cap}."
            )
        max_new = new_cap
    if max_new <= 0:
        raise ValueError(
            f"Computed max_new={max_new} is non-positive (prefix_len={prefix_len}, "
            f"block_size={block_size}). Re-pretrain with a larger block_size."
        )

    policy, vocab_size, _ = _build_model_from_ckpt(
        cfg, ckpt, accelerator.device,
        freeze=False, strict=bool(cfg.init.strict),
    )

    use_kl = float(cfg.grpo.beta) > 0.0
    reference = None
    if use_kl:
        ref_path = cfg.init.get("ref_ckpt_path", None)
        ref_ckpt = ckpt
        if ref_path:
            ref_path = Path(ref_path)
            if not ref_path.is_absolute():
                ref_path = (
                    Path(hydra.core.hydra_config.HydraConfig.get().runtime.cwd)
                    / ref_path
                )
            ref_ckpt = _load_ckpt(ref_path)
            if accelerator.is_main_process:
                log.info(f"Loaded reference checkpoint: {ref_path}")
        reference, _, _ = _build_model_from_ckpt(
            cfg, ref_ckpt, accelerator.device,
            freeze=True, strict=bool(cfg.init.strict),
        )

    if accelerator.is_main_process:
        log.info(f"Policy params: {_num_params(policy):,}")
        log.info(
            f"Star-graph CoT: deg={deg} path_len={path_len} num_nodes={num_nodes} "
            f"prefix_len={prefix_len} max_new={max_new} "
            f"vocab={vocab_size} block_size={block_size}"
        )
        _adv_mode_log = str(cfg.grpo.get("advantage", "grpo"))
        _lp_alpha_log = (
            float(cfg.grpo.length_penalty.alpha)
            if cfg.grpo.get("length_penalty", None) is not None
            else 0.0
        )
        log.info(
            f"GRPO: G={cfg.grpo.group_size} beta={cfg.grpo.beta} "
            f"T={cfg.grpo.temperature} top_k={cfg.grpo.top_k} "
            f"adv_eps={cfg.grpo.adv_eps} advantage={_adv_mode_log} "
            f"length_penalty.alpha={_lp_alpha_log}"
        )
        log.info(
            f"Train prompts: {len(train_ds):,} | Test prompts: {len(test_ds):,}"
        )

    # ----------------------------------------------------------------- loaders
    pad_id = tokenizer.DUMMY
    collate = partial(cot_pad_collate, pad_id=pad_id)
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

    # ---------------------------------------------------------- optimizer + sched
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )

    total_steps = int(cfg.schedule.steps)
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

    policy, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        policy, optimizer, train_loader, test_loader, scheduler
    )

    # ---------------------------------------------------------------- wandb
    if cfg.logging.wandb and accelerator.is_main_process:
        _adv_name = str(cfg.grpo.get("advantage", "grpo")).lower()
        _lp_alpha_name = (
            float(cfg.grpo.length_penalty.alpha)
            if cfg.grpo.get("length_penalty", None) is not None
            else 0.0
        )
        run_name = (
            f"grpo_{cfg.model.name}_{cfg.data.dataset}"
            f"_G{cfg.grpo.group_size}_b{cfg.grpo.beta:g}"
            f"_lr{cfg.optimizer.lr:g}_T{cfg.grpo.temperature:g}"
            f"_seed{cfg.seed}"
        )
        if _adv_name == "rloo":
            run_name += "_rloo"
        if _lp_alpha_name > 0.0:
            run_name += f"_lp{_lp_alpha_name:g}"
        custom_name = cfg.logging.get("name", None)
        if custom_name:
            run_name = f"{custom_name} {run_name}"
        wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ---------------------------------------------------------------- helpers
    G = int(cfg.grpo.group_size)
    beta = float(cfg.grpo.beta)
    adv_eps = float(cfg.grpo.adv_eps)
    temperature = float(cfg.grpo.temperature)
    top_k = cfg.grpo.get("top_k", None)
    adv_mode = str(cfg.grpo.get("advantage", "grpo")).lower()
    if adv_mode not in {"grpo", "rloo"}:
        raise ValueError(f"cfg.grpo.advantage must be 'grpo' or 'rloo', got {adv_mode!r}")
    if adv_mode == "rloo" and G < 2:
        raise ValueError("RLOO advantage requires cfg.grpo.group_size >= 2")
    lp_cfg = cfg.grpo.get("length_penalty", None)
    lp_alpha = float(lp_cfg.alpha) if lp_cfg is not None else 0.0
    lp_eps_std = float(lp_cfg.eps_std) if lp_cfg is not None else 1e-6
    if not (0.0 <= lp_alpha < 1.0):
        raise ValueError(
            f"cfg.grpo.length_penalty.alpha must be in [0, 1), got {lp_alpha}"
        )
    if (
        cfg.logging.wandb
        and accelerator.is_main_process
        and wandb.run is not None
    ):
        wandb.run.summary["advantage_mode"] = adv_mode
        wandb.run.summary["length_penalty_alpha"] = lp_alpha
    eos_id = tokenizer.EOS
    think_close_id = tokenizer.THINK_CLOSE
    think_open_id = tokenizer.THINK_OPEN
    backtrack_id = tokenizer.BACKTRACK
    autocast_ctx = accelerator.autocast

    def run_eval(step: int) -> None:
        policy.eval()
        metrics, samples_log = _eval_cot(
            policy,
            test_loader,
            accelerator,
            prefix_len=prefix_len,
            path_len=path_len,
            deg=deg,
            max_target_len=max_target_len,
            pad_id=pad_id,
            think_open_id=think_open_id,
            think_close_id=think_close_id,
            backtrack_id=backtrack_id,
            eos_id=eos_id,
            temperature=float(cfg.eval.temperature),
            top_k=cfg.eval.get("top_k", None),
            max_batches=cfg.eval.get("max_batches", None),
            tokenizer=tokenizer,
            log_samples=int(cfg.eval.get("log_samples", 0)),
            vocab_size=vocab_size,
        )
        if accelerator.is_main_process:
            report_idx = sorted({0, 1, path_len - 2, path_len - 1})
            tok_str = " ".join(
                f"t{j}={metrics.get(f'val/sample_token_{j}', float('nan')):.3f}"
                for j in report_idx
            )
            log.info(
                f"[eval @ step {step}/{total_steps}] "
                f"forced_loss={metrics.get('val/forced_loss', float('nan')):.4f} "
                f"sample_seq={metrics.get('val/sample_seq_acc', float('nan')):.4f} "
                f"no_close={metrics.get('val/no_think_close_rate', float('nan')):.3f} "
                f"eos_pos={metrics.get('val/eos_correct_pos_rate', float('nan')):.3f} "
                f"cot_valid={metrics.get('val/cot_chains_valid_rate', float('nan')):.3f} "
                f"cot_goal={metrics.get('val/cot_ends_at_goal_rate', float('nan')):.3f} "
                f"{tok_str}"
            )
            for k, s in enumerate(samples_log):
                log.info(
                    "  sample[%d] correct=%s eos_ok=%s cot_valid=%s cot_goal=%s\n"
                    "    prefix:    %s\n"
                    "    gt_trace:  %s\n"
                    "    gen:       %s\n"
                    "    gt_path:   %s\n"
                    "    pred_path: %s",
                    k, s["correct"], s["eos_ok"], s["cot_valid"], s["cot_ends_at_goal"],
                    s["prefix"], s["gt_trace"], s["gen"], s["gt_path"], s["pred_path"],
                )
            if cfg.logging.wandb:
                wandb.log(metrics, step=step)
        policy.train()

    # ---------------------------------------------------------------- loop
    eval_every = int(cfg.eval.get("every_steps", 0))
    bar = tqdm(
        total=total_steps,
        desc="grpo",
        disable=not accelerator.is_main_process,
    )
    policy.train()
    train_iter = iter(train_loader)
    global_step = 0
    while global_step < total_steps:
        try:
            full_ids, lengths = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            full_ids, lengths = next(train_iter)
        global_step += 1

        full_ids = full_ids.to(accelerator.device)
        lengths = lengths.to(accelerator.device)
        B = full_ids.size(0)

        # --- prompts + ground-truth answer ---
        prompt = full_ids[:, :prefix_len]
        ar = torch.arange(path_len, device=accelerator.device)
        gt_idx = (lengths.unsqueeze(1) - path_len - 1 + ar).clamp_min(0)
        gt_paths = full_ids.gather(1, gt_idx)                                    # (B, P)

        # --- expand to G rollouts per prompt ---
        prompt_g = prompt.repeat_interleave(G, dim=0)                            # (BG, prefix)
        gt_paths_g = gt_paths.repeat_interleave(G, dim=0)                        # (BG, P)

        # --- rollouts (no grad) ---
        policy.eval()
        gen_ids, response_mask = _ar_sample(
            policy,
            prompt_g,
            max_new=max_new,
            vocab_size=vocab_size,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id,
            pad_id=pad_id,
            autocast_ctx=autocast_ctx,
        )
        policy.train()

        # --- reward + (optional) length-penalty + group advantage ---
        rewards_raw = _compute_rewards(
            gen_ids, gt_paths_g,
            think_close_id=think_close_id, path_len=path_len,
        )                                                                         # (BG,)
        # Response lengths for the length penalty + diagnostics. Includes the
        # trailing <eos>; matches ``counts`` computed below from response_mask.
        response_lens = response_mask.float().sum(dim=1)                          # (BG,)
        rewards, _ = _apply_length_penalty(
            rewards_raw, response_lens, G,
            alpha=lp_alpha, eps_std=lp_eps_std,
        )
        advantages = _group_advantage(rewards, G, adv_eps, mode=adv_mode).detach()  # (BG,)

        # Per-group statistics of correct-rollout lengths. These are the
        # quantities the length penalty actually depends on, and they are NOT
        # tautological: a shrinking ``length_correct_std_mean`` says the model
        # is converging on a single short solution, and ``length_correct_mu_mean``
        # tracks the (per-prompt-averaged) length of correct rollouts.
        with torch.no_grad():
            B_groups = rewards_raw.shape[0] // G
            r_grp = rewards_raw.view(B_groups, G)
            L_grp = response_lens.view(B_groups, G).float()
            corr_grp = (r_grp > 0.5).float()
            n_corr_grp = corr_grp.sum(dim=1)                                      # (B,)
            mu_corr_grp = (
                (L_grp * corr_grp).sum(dim=1) / n_corr_grp.clamp_min(1.0)
            )                                                                     # (B,)
            var_corr_grp = (
                ((L_grp - mu_corr_grp.unsqueeze(1)) ** 2 * corr_grp).sum(dim=1)
                / n_corr_grp.clamp_min(1.0)
            )
            std_corr_grp = var_corr_grp.clamp_min(0.0).sqrt()                     # (B,)

        # --- recompute log-probs under current policy + reference ---
        full_seq = torch.cat([prompt_g, gen_ids], dim=1)                          # (BG, prefix+max_new)
        log_pi, entropy = _logprobs_at_response(
            policy, full_seq,
            prefix_len=prefix_len, max_new=max_new, vocab_size=vocab_size,
            autocast_ctx=autocast_ctx, no_grad=False, return_entropy=True,
        )

        mask_f = response_mask.float()
        counts = mask_f.sum(dim=1).clamp_min(1.0)                                 # (BG,)

        pg_per = -(advantages.unsqueeze(1) * log_pi * mask_f).sum(dim=1) / counts # (BG,)

        if use_kl:
            log_pi_ref, _ = _logprobs_at_response(
                reference, full_seq,
                prefix_len=prefix_len, max_new=max_new, vocab_size=vocab_size,
                autocast_ctx=autocast_ctx, no_grad=True, return_entropy=False,
            )
            log_ratio = log_pi_ref - log_pi
            kl_k3 = torch.exp(log_ratio) - log_ratio - 1.0                        # (BG, max_new)
            kl_per = (kl_k3 * mask_f).sum(dim=1) / counts                         # (BG,)
        else:
            kl_per = torch.zeros_like(pg_per)

        loss = (pg_per + beta * kl_per).mean()

        # --- step ---
        optimizer.zero_grad()
        accelerator.backward(loss)
        clip_val = (
            cfg.optimizer.grad_clip
            if cfg.optimizer.grad_clip and cfg.optimizer.grad_clip > 0
            else float("inf")
        )
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), clip_val)
        grad_norm_val = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
        optimizer.step()
        scheduler.step()

        # --- diagnostics ---
        with torch.no_grad():
            rewards_raw_g = accelerator.gather(rewards_raw.detach())
            rewards_g = accelerator.gather(rewards.detach())
            adv_g = accelerator.gather(advantages.detach())
            counts_g = accelerator.gather(counts.detach())
            n_corr_grp_g = accelerator.gather(n_corr_grp.detach())
            mu_corr_grp_g = accelerator.gather(mu_corr_grp.detach())
            std_corr_grp_g = accelerator.gather(std_corr_grp.detach())
            ent_mean = (entropy * mask_f).sum() / mask_f.sum().clamp_min(1.0)
            ent_g = accelerator.gather(ent_mean.detach().unsqueeze(0)).mean()
            kl_mean = kl_per.detach().mean()
            kl_g = accelerator.gather(kl_mean.unsqueeze(0)).mean()

        # Raw 0/1 correctness rate (= what current 'reward_mean' historically logged).
        reward_raw_mean = rewards_raw_g.mean().item()
        # Reward after length-penalty shaping (== raw when alpha == 0).
        reward_shaped_mean = rewards_g.mean().item()
        reward_shaped_std = rewards_g.std(unbiased=False).item()
        adv_abs_mean = adv_g.abs().mean().item()
        # Response length, split by raw correctness.
        # ``counts`` already includes the trailing ``<eos>`` (response_mask is
        # True up to and including the EOS-emitting position). When no EOS is
        # produced for a row, counts == max_new for that row.
        correct_mask = rewards_raw_g > 0.5
        n_correct = int(correct_mask.sum().item())
        n_incorrect = int((~correct_mask).sum().item())
        resp_len_all = counts_g.float().mean().item()
        resp_len_correct = (
            counts_g[correct_mask].float().mean().item()
            if n_correct > 0 else float("nan")
        )
        resp_len_incorrect = (
            counts_g[~correct_mask].float().mean().item()
            if n_incorrect > 0 else float("nan")
        )
        # Per-group correct-length statistics (the actual variables that the
        # length penalty conditions on). ``mu`` averages per-group mean correct
        # lengths over groups with >= 1 correct rollout; ``std`` averages
        # per-group std over groups with >= 2 correct (std is identically 0
        # with a single correct rollout, so including those would just dilute).
        has_corr_g = n_corr_grp_g >= 1
        has_std_g = n_corr_grp_g >= 2
        if int(has_corr_g.sum().item()) > 0:
            length_correct_mu_mean = (
                mu_corr_grp_g[has_corr_g].float().mean().item()
            )
        else:
            length_correct_mu_mean = float("nan")
        if int(has_std_g.sum().item()) > 0:
            length_correct_std_mean = (
                std_corr_grp_g[has_std_g].float().mean().item()
            )
        else:
            length_correct_std_mean = float("nan")

        bar.update(1)
        if accelerator.is_main_process:
            postfix = {
                "loss": f"{loss.item():.4f}",
                "r_raw": f"{reward_raw_mean:.3f}",
                "kl": f"{kl_g.item():.4f}",
                "len": f"{resp_len_all:.1f}",
                "gn": f"{grad_norm_val:.2f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
            if lp_alpha > 0.0:
                postfix["r_sh"] = f"{reward_shaped_mean:.3f}"
            bar.set_postfix(postfix)

        if cfg.logging.wandb and accelerator.is_main_process:
            payload = {
                "train/loss": loss.item(),
                # Raw 0/1 correctness; matches the value historically logged as
                # train/reward_mean and is the right metric for accuracy curves.
                "train/reward_raw_mean": reward_raw_mean,
                # Shaped reward after the length-penalty multiplier.
                "train/reward_mean": reward_shaped_mean,
                "train/reward_std": reward_shaped_std,
                "train/adv_abs_mean": adv_abs_mean,
                "train/kl": kl_g.item(),
                "train/entropy": ent_g.item(),
                "train/response_len": resp_len_all,
                "train/n_correct": n_correct,
                "train/n_incorrect": n_incorrect,
                "train/grad_norm": grad_norm_val,
                "train/lr": scheduler.get_last_lr()[0],
                "train/step": global_step,
            }
            if n_correct > 0:
                payload["train/response_len_correct"] = resp_len_correct
            if n_incorrect > 0:
                payload["train/response_len_incorrect"] = resp_len_incorrect
            if not math.isnan(length_correct_mu_mean):
                payload["train/length_correct_mu_mean"] = length_correct_mu_mean
            if not math.isnan(length_correct_std_mean):
                payload["train/length_correct_std_mean"] = length_correct_std_mean
            wandb.log(payload, step=global_step)

        if eval_every > 0 and (
            global_step % eval_every == 0 or global_step == total_steps
        ):
            run_eval(step=global_step)

    bar.close()

    if eval_every > 0 and global_step % eval_every != 0:
        run_eval(step=global_step)

    if cfg.logging.wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
