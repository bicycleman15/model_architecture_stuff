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
