#!/bin/bash
#SBATCH --job-name=overfit-sweep
#SBATCH --open-mode=append
#SBATCH --output=/scratch/jp7467/slurm_logs/%j_%x.out
#SBATCH --error=/scratch/jp7467/slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_235_cds
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8

START_TIME=$(date +%s)
echo "Job started at: $(date)"
################################################

source ~/.bashrc
conda activate
conda activate t2

################################################

# for MODEL in transformer mean_residual_transformer; do
for MODEL in transformer; do
  for LR in 5e-4 1e-3 5e-3 1e-2; do
    for WARMUP in 0 400; do

      MIN_LR=$(python3 -c "print(f'{float(\"$LR\") / 10:.1e}')")
      NAME="${MODEL} 12L 256D lr${LR} wu${WARMUP}"

      echo "=========================================="
      echo "Running: $NAME"
      echo "  LR=$LR  MIN_LR=$MIN_LR  WARMUP=$WARMUP"
      echo "=========================================="

      accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 overfit.py \
      --config-path config/residual \
      --config-name fineweb.yaml \
      wandb.project="overfit-residual" \
      wandb.exp_name="$NAME" \
      \
      train.batch_size=512 \
      train.global_batch_size=512 \
      \
      train.train_steps=4000 \
      train.warmup_steps=$WARMUP \
      \
      optimizer.lr=$LR \
      optimizer.min_lr=$MIN_LR \
      \
      model_type=$MODEL \
      ${MODEL}.block_size=512 \
      ${MODEL}.n_layer=12 \
      ${MODEL}.dim=256 \
      ${MODEL}.n_head=4

    done
  done
done

################################################

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Job finished at: $(date)"
echo "Time taken: $((ELAPSED_TIME / 3600))h $((ELAPSED_TIME % 3600 / 60))m $((ELAPSED_TIME % 60))s"
