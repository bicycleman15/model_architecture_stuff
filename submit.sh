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


accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
--config-path config/dynamic_chunk \
--config-name char.yaml \
wandb.project="tinystories-padded-mod15" \
wandb.exp_name="flat 6L lr 8e-4" \
\
train.batch_size=32 \
train.global_batch_size=32 \
\
train.train_steps=4000 \
eval.eval_interval=400 \
eval.eval_iters=50 \
\
optimizer.lr=8e-4 \
optimizer.min_lr=8e-5 \
\
model_type=transformer \
transformer.block_size=8192 \
transformer.n_layer=6 \
transformer.dim=768 \
transformer.n_head=12



echo "Run finished at: "
date
exit