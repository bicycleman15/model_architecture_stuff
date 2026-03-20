#!/bin/bash

WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
--config-path config \
--config-name byte.yaml \
wandb.project="fineweb-byte" \
wandb.exp_name="test" \
\
dataset.path="/Users/jp7467/Desktop/model_architecture_stuff/data/tinystories-gpt4-clean" \
dataset.tokenizer_name="bicycleman15/tinystories-gpt4-clean-tokenizer" \
dataset.vocab_size=256 \
\
train.batch_size=32 \
\
train.train_epochs=1 \
eval.eval_interval=100 \
eval.eval_iters=10 \
train.grad_accum=2 \
\
optimizer.lr=8e-4 \
optimizer.min_lr=8e-5 \
\
model_type=transformer \
transformer.block_size=256 \
transformer.n_layer=3 \
transformer.dim=512 \
transformer.n_head=8

# hourglass
WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
--config-path config \
--config-name byte.yaml \
wandb.project="fineweb-byte" \
wandb.exp_name="test" \
\
train.batch_size=32 \
\
train.train_epochs=1 \
eval.eval_interval=100 \
eval.eval_iters=10 \
train.grad_accum=2 \
\
optimizer.lr=8e-4 \
optimizer.min_lr=8e-5 \
\
model_type=hourglass \
hourglass.block_size=1024 \
hourglass.chunk_method="uniform" \
hourglass.chunk_size=4 \
\
hourglass.dim=512 \
hourglass.n_head=8 \
hourglass.n_compressor_layers=3 \
hourglass.n_processor_layers=6 \
hourglass.n_decoder_layers=3
