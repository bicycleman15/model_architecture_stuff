#!/bin/bash
#SBATCH --job-name=res
#SBATCH --open-mode=append
#SBATCH --output=/scratch/jp7467/slurm_logs/%j_%x.out
#SBATCH --error=/scratch/jp7467/slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_235_cds
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8

START_TIME=$(date +%s)
echo "Job started at: $(date)"
################################################

source ~/.bashrc # so that we can read HF_AUTH_TOKEN :)

conda activate
conda activate t2

################################################

accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 overfit.py \
--config-path config/residual \
--config-name fineweb.yaml \
wandb.project="overfit-residual" \
wandb.exp_name="sanity check mean tp1_rms 36L 256D lr 8e-4" \
\
train.batch_size=512 \
train.global_batch_size=512 \
\
train.train_steps=4000 \
train.warmup_steps=400 \
\
optimizer.lr=8e-4 \
optimizer.min_lr=8e-5 \
\
model_type=mean_residual_transformer \
mean_residual_transformer.block_size=512 \
mean_residual_transformer.n_layer=36 \
mean_residual_transformer.dim=256 \
mean_residual_transformer.n_head=4 \
mean_residual_transformer.mean_power=1


################################################

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Job finished at: $(date)"
echo "Time taken: $((ELAPSED_TIME / 3600))h $((ELAPSED_TIME % 3600 / 60))m $((ELAPSED_TIME % 60))s"