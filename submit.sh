#!/bin/bash
#SBATCH --job-name=byte
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100_dev,a100_short,a100_long

# first activate the environment and then submit the job from the terminal :)
# use `sbatch submit.slurm`
# note we do not have WANDB_MODE=offline here because we are submitting the job to the cluster


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



echo "Run finished at: "
date
exit