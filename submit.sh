#!/bin/bash
#SBATCH --job-name=path
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100_dev,a100_short,a100_long

# first activate the environment and then submit the job from the terminal :)
# use `sbatch submit.slurm`
# note we do not have WANDB_MODE=offline here because we are submitting the job to the cluster

WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 overfit.py \
--config-path config \
--config-name bpe.yaml \
wandb.project="overfit-path" \
wandb.exp_name="vanilla muon" \
\
train.batch_size=64 \
train.global_batch_size=64 \
\
train.train_steps=400 \
\
optimizer.lr=1e-2 \
optimizer.min_lr=1e-2 \
train.grad_norm=1000000 \
train.warmup_steps=0 \
\
model_type=transformer \
transformer.block_size=512 \
transformer.n_layer=6 \
transformer.dim=256 \
transformer.n_head=4 \
transformer.use_fused_ops=True \
optimizer.name=muon \
optimizer.muon_momentum=0.95


WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 overfit.py \
--config-path config \
--config-name bpe.yaml \
wandb.project="overfit-path" \
wandb.exp_name="1 _ p _ 0.01 muon" \
\
train.batch_size=64 \
train.global_batch_size=64 \
\
train.train_steps=400 \
\
optimizer.lr=1e-2 \
optimizer.min_lr=1e-2 \
train.grad_norm=1000000 \
train.warmup_steps=0 \
\
model_type=path_transformer \
path_transformer.block_size=512 \
path_transformer.n_layer=6 \
path_transformer.dim=256 \
path_transformer.n_head=4 \
path_transformer.use_fused_ops=True \
path_transformer.damping=0.01 \
optimizer.name=muon \
optimizer.muon_momentum=0.95

echo "Run finished at: "
date
exit