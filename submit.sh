#!/bin/bash
#SBATCH --job-name=path
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100_dev,a100_short,a100_long

# first activate the environment and then submit the job from the terminal :)
# use `sbatch submit.slurm`
# note we do not have WANDB_MODE=offline here because we are submitting the job to the cluster

accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=2 --multi_gpu train.py \
--config-path config \
--config-name bpe.yaml \
wandb.project="fineweb-1b" \
wandb.exp_name="path 12L lr 1e-3 damping 1e-1" \
\
train.batch_size=32 \
train.global_batch_size=256 \
\
train.train_steps=4000 \
eval.eval_interval=400 \
eval.eval_iters=100 \
\
optimizer.lr=4 \
optimizer.min_lr=1e-4 \
\
model_type=path_transformer \
path_transformer.block_size=1024 \
path_transformer.n_layer=12 \
path_transformer.dim=768 \
path_transformer.n_head=12 \
path_transformer.use_fused_ops=True \
path_transformer.damping=1e-1


echo "Run finished at: "
date
exit