#!/bin/bash
#SBATCH --job-name=path
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
accelerate launch \
--config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 \
-m next_token.train \
logging.project="learnability-star-graph" \
logging.name="3 probe" \
data=paper \
data.num_nodes=100 \
data.n_train=200000 \
\
model.n_layer=12 \
model=transformer \
\
batch_size=512 \
schedule.warmup_steps=100 \
optimizer.lr=5e-4 \
optimizer.min_lr=5e-5 \
optimizer.weight_decay=0.1 \
schedule.epochs=50 \
eval.every_pct=0.01 \
\
nextlat.enabled=true \
nextlat.lambda_h=1.0 \
nextlat.lambda_kl=0.0 \
nextlat.n_hidden_layers=3 \
nextlat.hidden_mult=1

# \
# nextlat.enabled=true \
# nextlat.lambda_h=0 \
# nextlat.lambda_kl=1.0 \
# nextlat.n_hidden_layers=3 \
# nextlat.hidden_mult=1

# data.teacherless=true

# model=hybrid \
# 'model.pattern=[attention,rnn]' \
# model.rnn.gradient_clipping=1.0 \

# WANDB_MODE=offline \
# accelerate launch \
# --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 \
# -m next_token.train \
# logging.project=learnability-star-graph logging.name=test \
# data=paper data.num_nodes=100 \
# model=hybrid \
# 'model.pattern=[rnn, attention]' \
# model.n_layer=12 batch_size=512 \
# optimizer.lr=5e-4 optimizer.min_lr=5e-5 optimizer.weight_decay=0.1 \
# schedule.epochs=50


# WANDB_MODE=offline \
# accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 overfit.py \
# --config-path config \
# --config-name bpe.yaml \
# wandb.project="overfit-path" \
# wandb.exp_name="vanilla muon" \
# \
# train.batch_size=64 \
# train.global_batch_size=64 \
# \
# train.train_steps=400 \
# \
# optimizer.lr=1e-2 \
# optimizer.min_lr=1e-2 \
# train.grad_norm=1000000 \
# train.warmup_steps=0 \
# \
# model_type=transformer \
# transformer.block_size=512 \
# transformer.n_layer=6 \
# transformer.dim=256 \
# transformer.n_head=4 \
# transformer.use_fused_ops=True \
# optimizer.name=muon \
# optimizer.muon_momentum=0.95


# WANDB_MODE=offline \
# accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 overfit.py \
# --config-path config \
# --config-name bpe.yaml \
# wandb.project="overfit-path" \
# wandb.exp_name="1 _ p _ 0.01 muon" \
# \
# train.batch_size=64 \
# train.global_batch_size=64 \
# \
# train.train_steps=400 \
# \
# optimizer.lr=1e-2 \
# optimizer.min_lr=1e-2 \
# train.grad_norm=1000000 \
# train.warmup_steps=0 \
# \
# model_type=path_transformer \
# path_transformer.block_size=512 \
# path_transformer.n_layer=6 \
# path_transformer.dim=256 \
# path_transformer.n_head=4 \
# path_transformer.use_fused_ops=True \
# path_transformer.damping=0.01 \
# optimizer.name=muon \
# optimizer.muon_momentum=0.95

# WANDB_MODE=offline \
# accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
# --config-path config \
# --config-name bpe.yaml \
# wandb.project="fineweb-1b" \
# wandb.exp_name="no_clip vanilla 12L lr 2e-2 muon" \
# \
# train.batch_size=32 \
# train.global_batch_size=256 \
# \
# train.train_steps=4000 \
# train.warmup_steps=1000 \
# eval.eval_interval=400 \
# eval.eval_iters=100 \
# \
# train.grad_norm=10000 \
# optimizer.name=muon \
# optimizer.lr=0.02 \
# optimizer.min_lr=0.002 \
# optimizer.adamw_lr_mul=0.1 \
# optimizer.betas=[0.9,0.95] \
# optimizer.weight_decay=0.0 \
# optimizer.muon_weight_decay=0.0 \
# optimizer.muon_momentum=0.95 \
# optimizer.muon_nesterov=true \
# optimizer.muon_ns_steps=5 \
# \
# model_type=transformer \
# transformer.block_size=1024 \
# transformer.n_layer=12 \
# transformer.dim=768 \
# transformer.n_head=12 \
# transformer.use_fused_ops=True


# WANDB_MODE=offline \
# accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
# --config-path config \
# --config-name bpe.yaml \
# wandb.project="fineweb-1b" \
# wandb.exp_name="no_clip zero path 1_p_g_2 12L lr 2e-2 damp 1e-2 muon" \
# \
# train.batch_size=32 \
# train.global_batch_size=256 \
# \
# train.train_steps=4000 \
# train.warmup_steps=1000 \
# eval.eval_interval=400 \
# eval.eval_iters=100 \
# \
# train.grad_norm=10000 \
# optimizer.name=muon \
# optimizer.lr=0.02 \
# optimizer.min_lr=0.002 \
# optimizer.adamw_lr_mul=0.1 \
# optimizer.betas=[0.9,0.95] \
# optimizer.weight_decay=0.0 \
# optimizer.muon_weight_decay=0.0 \
# optimizer.muon_momentum=0.95 \
# optimizer.muon_nesterov=true \
# optimizer.muon_ns_steps=5 \
# \
# model_type=path_transformer \
# path_transformer.block_size=1024 \
# path_transformer.n_layer=12 \
# path_transformer.dim=768 \
# path_transformer.n_head=12 \
# path_transformer.use_fused_ops=True \
# path_transformer.damping=1e-2

# WANDB_MODE=offline \
# accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
# --config-path config \
# --config-name bpe.yaml \
# wandb.project="fineweb-1b" \
# wandb.exp_name="no_clip path 1_p_g_2 12L lr 1e-3 damp 1e-2 adam" \
# \
# train.batch_size=32 \
# train.global_batch_size=256 \
# \
# train.train_steps=4000 \
# train.warmup_steps=1000 \
# eval.eval_interval=400 \
# eval.eval_iters=100 \
# \
# train.grad_norm=10000 \
# optimizer.name=adamw \
# optimizer.lr=1e-3 \
# optimizer.min_lr=1e-4 \
# \
# model_type=path_transformer \
# path_transformer.block_size=1024 \
# path_transformer.n_layer=12 \
# path_transformer.dim=768 \
# path_transformer.n_head=12 \
# path_transformer.use_fused_ops=True \
# path_transformer.damping=1e-2

# accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
# --config-path config \
# --config-name bpe.yaml \
# wandb.project="fineweb-1b" \
# wandb.exp_name="no_clip vanilla 12L lr 1e-3 adam" \
# \
# train.batch_size=32 \
# train.global_batch_size=256 \
# \
# train.train_steps=4000 \
# train.warmup_steps=1000 \
# eval.eval_interval=400 \
# eval.eval_iters=100 \
# \
# train.grad_norm=10000 \
# optimizer.name=adamw \
# optimizer.lr=1e-3 \
# optimizer.min_lr=1e-4 \
# \
# model_type=transformer \
# transformer.block_size=1024 \
# transformer.n_layer=12 \
# transformer.dim=768 \
# transformer.n_head=12 \
# transformer.use_fused_ops=True


echo "Run finished at: "
date
exit