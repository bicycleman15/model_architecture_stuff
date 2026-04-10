#!/bin/bash
#SBATCH --job-name=byte
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100_dev,a100_short,a100_long

# first activate the environment and then submit the job from the terminal :)
# use `sbatch submit.slurm`
# note we do not have WANDB_MODE=offline here because we are submitting the job to the cluster


# WANDB_MODE=offline \
# accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 char_train.py \
# --config-path config/dynamic_chunk \
# --config-name char.yaml \
# wandb.project="fineweb-1b-byte-padded-mod15" \
# wandb.exp_name="mask flat 6L lr 8e-4" \
# \
# train.batch_size=32 \
# train.global_batch_size=32 \
# \
# train.train_steps=4000 \
# eval.eval_interval=400 \
# eval.eval_iters=50 \
# \
# optimizer.lr=8e-4 \
# optimizer.min_lr=8e-5 \
# \
# model_type=transformer \
# transformer.block_size=8192 \
# transformer.n_layer=6 \
# transformer.dim=768 \
# transformer.n_head=12 \
# \
# dataset.mask_zero_padding=true


# WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 char_train.py \
--config-path config/dynamic_chunk \
--config-name char.yaml \
wandb.project="fineweb-1b-byte-padded-mod15" \
wandb.exp_name="mask spacebyte 6L lr 8e-4" \
\
train.batch_size=16 \
train.global_batch_size=32 \
\
train.train_steps=4000 \
train.warmup_steps=1000 \
eval.eval_interval=400 \
eval.eval_iters=50 \
\
optimizer.lr=8e-4 \
optimizer.min_lr=8e-5 \
\
model_type=hourglass \
hourglass.block_size=8192 \
hourglass.chunk_method="spacebyte" \
hourglass.chunk_size=5 \
\
hourglass.dim=768 \
hourglass.n_head=12 \
hourglass.n_compressor_layers=2 \
hourglass.n_processor_layers=6 \
hourglass.n_decoder_layers=2 \
\
dataset.mask_zero_padding=true


echo "Run finished at: "
date
exit