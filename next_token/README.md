# Next-token failures: star-graph experiment

Reproduces the star-graph next-token-prediction experiment from
[Bachmann & Nagarajan 2024](https://arxiv.org/abs/2403.06963)
([reference repo](https://github.com/gregorbachmann/Next-Token-Failures)) inside
this codebase, reusing the workspace `Transformer` from
[`models/transformer.py`](../models/transformer.py).

## Layout

```
next_token/
  data.py             # star_graph(), NumeralTokenizer, StarGraphDataset
  generate_data.py    # fire CLI: write train/test .txt files
  train.py            # Hydra + Accelerate trainer
  models/
    transformer.py    # thin wrapper around workspace Transformer
  config/
    star_graph.yaml   # top-level
    data/{paper,small}.yaml
    model/transformer.yaml
```

## Step 1 - generate data

```bash
# paper defaults (deg=2, path=5, num_nodes=50, 200k train / 20k test)
python -m next_token.generate_data --deg=2 --path_len=5 --num_nodes=50 \
    --n_train=200000 --n_test=20000

# small for fast iteration
python -m next_token.generate_data --deg=2 --path_len=5 --num_nodes=20 \
    --n_train=20000 --n_test=2000

# reverse-encoded targets (Bachmann & Nagarajan section 5)
python -m next_token.generate_data --deg=2 --path_len=5 --num_nodes=50 \
    --n_train=200000 --n_test=20000 --reverse=True
```

Files land under `next_token/data/deg{deg}_path{p}_nodes{n}{_rev?}/{train,test}.txt`.

## Step 2 - train

### Paper-default run

```bash
WANDB_MODE=offline accelerate launch --config-file accelerate.yaml \
    --mixed_precision=bf16 --num_processes=1 \
    -m next_token.train
```

### Smoke test (~minutes on a single GPU)

```bash
WANDB_MODE=offline accelerate launch --config-file accelerate.yaml \
    --mixed_precision=bf16 --num_processes=1 \
    -m next_token.train data=small schedule.epochs=10
```

### Ablations

```bash
# reverse-encoded targets (expected to fix the failure)
... -m next_token.train data.reverse=true

# teacherless / multi-token objective (expected to fix the failure)
... -m next_token.train data.teacherless=true

# bigger model / different lr
... -m next_token.train model.dim=512 model.n_layer=8 optimizer.lr=3e-4
```

## What to look for

With paper defaults the standard transformer is supposed to **fail on the
first target token**: free-generation accuracy on `val/gen_token_0` should
hover near `1/deg = 0.5` even after the *teacher-forced* loss
(`val/forced_loss`) collapses near zero. Setting `data.reverse=true` or
`data.teacherless=true` should fix `val/gen_token_0` and push
`val/gen_seq_acc` toward 1.0.

Per-position accuracies are logged as `val/forced_token_{j}` and
`val/gen_token_{j}` for `j in [0, target_len)` so the failure pattern is
directly visible in wandb.
