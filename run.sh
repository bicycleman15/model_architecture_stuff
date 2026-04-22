#!/bin/bash

WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 overfit.py \
--config-path config \
--config-name bpe.yaml \
wandb.project="fineweb-1b" \
wandb.exp_name="path 12L lr 1e-3 damp 1e-2 sgd momen 0.9" \
\
train.batch_size=32 \
train.global_batch_size=256 \
\
train.train_steps=4000 \
eval.eval_interval=400 \
eval.eval_iters=100 \
\
optimizer.lr=1e-3 \
optimizer.min_lr=1e-4 \
\
model_type=path_transformer \
path_transformer.block_size=1024 \
path_transformer.n_layer=12 \
path_transformer.dim=768 \
path_transformer.n_head=12 \
path_transformer.use_fused_ops=True \
path_transformer.damping=1e-2

# hourglass
WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
--config-path config \
--config-name byte.yaml \
wandb.project="fineweb-byte-2" \
wandb.exp_name="uniform-4 6L lr 8e-4" \
\
train.batch_size=32 \
train.global_batch_size=256 \
\
train.train_steps=8000 \
eval.eval_interval=800 \
eval.eval_iters=100 \
\
optimizer.lr=8e-4 \
optimizer.min_lr=8e-5 \
\
model_type=hourglass \
hourglass.block_size=1024 \
hourglass.chunk_method="uniform" \
hourglass.chunk_size=4 \
\
hourglass.dim=768 \
hourglass.n_head=12 \
hourglass.n_compressor_layers=3 \
hourglass.n_processor_layers=6 \
hourglass.n_decoder_layers=3

WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
--config-path config \
--config-name byte.yaml \
wandb.project="fineweb-byte-2" \
wandb.exp_name="hnet basic 6L lr 8e-4" \
\
train.batch_size=32 \
train.global_batch_size=256 \
\
train.train_steps=8000 \
eval.eval_interval=800 \
eval.eval_iters=100 \
\
optimizer.lr=8e-4 \
optimizer.min_lr=8e-5 \
\
model_type=hnet \
hnet.block_size=1024 \
hnet.chunk_method="router" \
hnet.chunk_size=4 \
\
hnet.dim=768 \
hnet.n_head=12 \
hnet.n_compressor_layers=3 \
hnet.n_processor_layers=6 \
hnet.n_decoder_layers=3


WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
--config-path config \
--config-name byte.yaml \
wandb.project="fineweb-byte-2" \
wandb.exp_name="rl 6L lr 8e-4" \
\
train.batch_size=32 \
train.global_batch_size=256 \
\
train.train_steps=8000 \
eval.eval_interval=800 \
eval.eval_iters=100 \
\
optimizer.lr=8e-4 \
optimizer.min_lr=8e-5 \
\
model_type=reinforce_hourglass \
reinforce_hourglass.block_size=1024 \
reinforce_hourglass.chunk_method="router" \
reinforce_hourglass.chunk_size=4 \
\
reinforce_hourglass.dim=768 \
reinforce_hourglass.n_head=12 \
reinforce_hourglass.n_compressor_layers=3 \
reinforce_hourglass.n_processor_layers=6 \
reinforce_hourglass.n_decoder_layers=3 \
\
reinforce_hourglass.use_router_scaling=False

## residual stuff

WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
--config-path config/residual \
--config-name fineweb.yaml \
wandb.project="fineweb-residual" \
wandb.exp_name="vanilla 36L 256D lr 8e-4" \
\
train.batch_size=128 \
train.global_batch_size=512 \
\
train.train_steps=8000 \
eval.eval_interval=800 \
eval.eval_iters=100 \
\
optimizer.lr=8e-4 \
optimizer.min_lr=8e-5 \
\
model_type=transformer \
transformer.block_size=1024 \
transformer.n_layer=36 \
transformer.dim=256 \
transformer.n_head=4


WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
--config-path config/residual \
--config-name fineweb.yaml \
wandb.project="fineweb-residual" \
wandb.exp_name="mean res 24L 256D lr 8e-4" \
\
train.batch_size=128 \
train.global_batch_size=512 \
\
train.train_steps=8000 \
eval.eval_interval=800 \
eval.eval_iters=500 \
\
optimizer.lr=8e-4 \
optimizer.min_lr=8e-5 \
\
model_type=mean_residual_transformer \
mean_residual_transformer.block_size=1024 \
mean_residual_transformer.n_layer=36 \
mean_residual_transformer.dim=256 \
mean_residual_transformer.n_head=4


# overfit stuff

# vanilla

WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 overfit.py \
--config-path config/residual \
--config-name fineweb.yaml \
wandb.project="overfit-residual" \
wandb.exp_name="vanilla 12L 256D lr 12e-4" \
\
train.batch_size=128 \
train.global_batch_size=512 \
\
train.train_steps=32000 \
\
optimizer.lr=12e-4 \
optimizer.min_lr=12e-5 \
\
model_type=transformer \
transformer.block_size=1024 \
transformer.n_layer=12 \
transformer.dim=256 \
transformer.n_head=4

# res

WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 overfit.py \
--config-path config/residual \
--config-name fineweb.yaml \
wandb.project="overfit-residual" \
wandb.exp_name="mean tp2_rms 36L 256D lr 1e-3" \
\
train.batch_size=512 \
train.global_batch_size=512 \
\
train.train_steps=4000 \
train.warmup_steps=400 \
\
optimizer.lr=1e-3 \
optimizer.min_lr=1e-4 \
\
model_type=mean_residual_transformer \
mean_residual_transformer.block_size=512 \
mean_residual_transformer.n_layer=36 \
mean_residual_transformer.dim=256 \
mean_residual_transformer.n_head=4 \
mean_residual_transformer.mean_power=2



#### state tracking

# Paper: arXiv:2502.10297, Appendix C.1
#   S_3: train at k=128 (2M samples), eval length-extrapolation at k=512 (500k samples)
#   single-layer DeltaProduct, 12 heads, head_dim=32
#   batch=2048, lr=1e-3 cosine, epochs=100, wd=1e-6, AdamW
#   no curriculum over sequence length

# --- generate datasets (once; skips if CSVs already exist) ---
# python -m state_tracking.generate_data --group=S3 --k=128 --samples=2000000
# python -m state_tracking.generate_data --group=S3 --k=512 --samples=500000

# --- train: DeltaProduct (n_h=2) on S_3, k=128 train / k=512 eval ---
WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 \
-m state_tracking.train \
--config-path config \
--config-name state_tracking.yaml \
model=deltaproduct \
data=s3_128 \
batch_size=2048 \
schedule.epochs=100 \
optimizer.lr=1e-3 \
optimizer.weight_decay=1e-6 \
curriculum.enabled=false

# --- DeltaNet (n_h=1) ---
WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 \
-m state_tracking.train \
--config-path config \
--config-name state_tracking.yaml \
model=deltanet \
data=s3_128 \
batch_size=2048 \
schedule.epochs=100 \
optimizer.lr=1e-3 \
optimizer.weight_decay=1e-6 \
curriculum.enabled=false

# --- Transformer baseline (paper doesn't use this; curriculum usually helps here) ---
WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 \
-m state_tracking.train --config-path config --config-name state_tracking.yaml \
model=transformer data=s3_128 \
batch_size=2048 schedule.epochs=30 optimizer.lr=1e-3 \
optimizer.weight_decay=1e-6 \
curriculum.enabled=false

# --- M2RNN baseline (vendored from open-lm-engine/accelerated-model-architectures) ---
# State tensor is [B, S, N, K, V]; large heads + big batch OOMs easily, so
# use a smaller batch than DeltaProduct.
# Default uses the vendored triton kernel (model.backend=triton). Pass
# `model.backend=torch` to fall back to the pure-PyTorch reference (slow;
# useful for CPU debugging only).
WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 \
-m state_tracking.train \
--config-path config \
--config-name state_tracking.yaml \
model=rnn \
data=s3_128 \
batch_size=512 \
schedule.epochs=30 \
optimizer.lr=1e-3 \
optimizer.weight_decay=1e-6 \
optimizer.grad_clip=1.0 \
curriculum.enabled=false

# --- M2RNN one-batch overfit sanity check ---
# Freezes one training batch and hammers the model on it for N steps.
# Loss should collapse towards 0 and token_acc towards 1.0 within a
# few hundred steps; otherwise something in the model / autograd path
# is broken. Small batch so it fits comfortably on a single GPU.
WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 \
-m state_tracking.overfit \
--config-path config --config-name state_tracking.yaml \
model=rnn data=s3_128 \
batch_size=32 \
optimizer.lr=1e-3 optimizer.grad_clip=1.0 \
curriculum.enabled=false \
+overfit.steps=500 +overfit.log_every=10 model.backend=torch model.gradient_clipping=1.0


#### path preserving stuff

accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 overfit.py \
--config-path config \
--config-name bpe.yaml \
wandb.project="overfit-path" \
wandb.exp_name="path 12L lr 1e-3 damping 1e-1" \
\
train.batch_size=32 \
train.global_batch_size=256 \
\
train.train_steps=4000 \
eval.eval_interval=400 \
eval.eval_iters=100 \
\
optimizer.lr=1e-3 \
optimizer.min_lr=1e-4 \
\
model_type=path_transformer \
path_transformer.block_size=1024 \
path_transformer.n_layer=12 \
path_transformer.dim=768 \
path_transformer.n_head=12 \
path_transformer.use_fused_ops=True \
path_transformer.damping=1e-1