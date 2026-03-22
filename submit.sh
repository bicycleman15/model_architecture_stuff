#!/bin/bash
#SBATCH --job-name=byte
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100_dev,a100_short,a100_long

# first activate the environment and then submit the job from the terminal :)
# use `sbatch submit.slurm`
# note we do not have WANDB_MODE=offline here because we are submitting the job to the cluster

# WANDB_MODE=offline \
# accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
# --config-path config \
# --config-name byte.yaml \
# wandb.project="fineweb-byte" \
# wandb.exp_name="vanilla l12 init lr 4e-4" \
# \
# train.batch_size=32 \
# \
# train.train_epochs=1 \
# eval.eval_interval=2500 \
# eval.eval_iters=100 \
# train.grad_accum=8 \
# \
# optimizer.lr=4e-4 \
# optimizer.min_lr=4e-5 \
# \
# model_type=transformer \
# transformer.block_size=1024 \
# \
# transformer.dim=768 \
# transformer.n_head=12 \
# transformer.n_layer=12



# WANDB_MODE=offline \
# accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
# --config-path config \
# --config-name byte.yaml \
# wandb.project="fineweb-byte" \
# wandb.exp_name="uniform-6 res init lr 4e-4" \
# \
# train.batch_size=32 \
# \
# train.train_epochs=1 \
# eval.eval_interval=2500 \
# eval.eval_iters=100 \
# train.grad_accum=8 \
# \
# optimizer.lr=4e-4 \
# optimizer.min_lr=4e-5 \
# \
# model_type=hourglass \
# hourglass.block_size=1024 \
# hourglass.chunk_method="uniform" \
# hourglass.chunk_size=6 \
# \
# hourglass.dim=768 \
# hourglass.n_head=12 \
# hourglass.n_compressor_layers=3 \
# hourglass.n_processor_layers=6 \
# hourglass.n_decoder_layers=3



# WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
--config-path config \
--config-name byte.yaml \
wandb.project="fineweb-byte" \
wandb.exp_name="reinforce gamma_0.99 lr_4e-4" \
\
train.batch_size=32 \
\
train.train_epochs=1 \
eval.eval_interval=2500 \
eval.eval_iters=100 \
train.grad_accum=8 \
\
optimizer.lr=4e-4 \
optimizer.min_lr=4e-5 \
\
model_type=reinforce_hourglass \
reinforce_hourglass.block_size=1024 \
reinforce_hourglass.reinforce_gamma=0.99 \
\
reinforce_hourglass.dim=768 \
reinforce_hourglass.n_head=12 \
reinforce_hourglass.n_compressor_layers=3 \
reinforce_hourglass.n_processor_layers=6 \
reinforce_hourglass.n_decoder_layers=3



echo "Run finished at: "
date
exit