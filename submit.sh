#!/bin/bash
#SBATCH --job-name=hnet
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100_dev,a100_short,a100_long

# first activate the environment and then submit the job from the terminal :)
# use `sbatch submit.slurm`
# note we do not have WANDB_MODE=offline here because we are submitting the job to the cluster

accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py

echo "Run finished at: "
date
exit