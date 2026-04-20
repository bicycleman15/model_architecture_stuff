#!/bin/bash
#SBATCH --job-name=state
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

accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 \
-m state_tracking.train --config-path config --config-name state_tracking.yaml \
model=transformer data=s3_128 \
batch_size=2048 schedule.epochs=30 optimizer.lr=1e-3 \
optimizer.weight_decay=1e-6 \
curriculum.enabled=false


echo "Run finished at: "
date
exit