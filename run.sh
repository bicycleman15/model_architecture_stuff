#!/bin/bash

WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
--config-path config \
--config-name byte.yaml \
wandb.project="fineweb-byte-2" \
wandb.exp_name="llamabyte 6L lr 8e-4" \
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
model_type=transformer \
transformer.block_size=1024 \
transformer.n_layer=6 \
transformer.dim=768 \
transformer.n_head=12

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
wandb.exp_name="mean res 12L 256D lr 1e-3" \
\
train.batch_size=512 \
train.global_batch_size=512 \
\
train.train_steps=10000 \
train.warmup_steps=0 \
\
optimizer.lr=1e-3 \
optimizer.min_lr=1e-4 \
\
model_type=mean_residual_transformer \
mean_residual_transformer.block_size=512 \
mean_residual_transformer.n_layer=12 \
mean_residual_transformer.dim=256 \
mean_residual_transformer.n_head=4