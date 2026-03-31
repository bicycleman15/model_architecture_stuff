#!/bin/bash
#SBATCH --job-name=byte
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

accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
--config-path config \
--config-name byte.yaml \
wandb.project="fineweb-byte-2" \
wandb.exp_name="uniform-8 6L lr 8e-4" \
\
train.batch_size=32 \
train.global_batch_size=256 \
\
train.train_steps=8000 \
eval.eval_interval=800 \
eval.eval_iters=100 \
\
optimizer.lr=8e-4 \
optimizer.min_lr=8e-5 \
\
model_type=hourglass \
hourglass.block_size=1024 \
hourglass.chunk_method="uniform" \
hourglass.chunk_size=8 \
\
hourglass.dim=768 \
hourglass.n_head=12 \
hourglass.n_compressor_layers=3 \
hourglass.n_processor_layers=6 \
hourglass.n_decoder_layers=3

# BPE baseline
# WANDB_MODE=offline \
# accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
# --config-path config \
# --config-name bpe.yaml \
# wandb.project="fineweb-nospace-byte" \
# wandb.exp_name="bpe 1152D 6L lr 8e-4" \
# dataset.path="/gpfs/data/ranganathlab/Jatin/model_architecture_stuff/data/fineweb-1b-nospace-gpt2" \
# dataset.tokenizer_name="gpt2" \
# \
# train.batch_size=32 \
# \
# train.train_epochs=1 \
# eval.eval_interval=300 \
# eval.eval_iters=100 \
# train.grad_accum=8 \
# \
# optimizer.lr=8e-4 \
# optimizer.min_lr=8e-5 \
# \
# model_type=transformer \
# transformer.block_size=256 \
# \
# transformer.dim=1152 \
# transformer.n_head=18 \
# transformer.n_layer=6



# WANDB_MODE=offline \
# accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
# --config-path config \
# --config-name byte.yaml \
# wandb.project="fineweb-nospace-byte" \
# wandb.exp_name="space lr 4e-4" \
# dataset.path="/gpfs/data/ranganathlab/Jatin/model_architecture_stuff/data/fineweb-1b-nospace-byte" \
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
# hourglass.chunk_method="spacebyte" \
# hourglass.chunk_size=8 \
# \
# hourglass.dim=768 \
# hourglass.n_head=12 \
# hourglass.n_compressor_layers=3 \
# hourglass.n_processor_layers=6 \
# hourglass.n_decoder_layers=3



# WANDB_MODE=offline \
# accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
# --config-path config \
# --config-name byte.yaml \
# wandb.project="fineweb-byte" \
# wandb.exp_name="reinforce vanilla_0.01 gamma_0.99 aux_0.1 z_loss_0.01 trial_1 lr_4e-4" \
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
# model_type=reinforce_hourglass \
# reinforce_hourglass.block_size=1024 \
# \
# reinforce_hourglass.reinforce_gamma=0.99 \
# reinforce_hourglass.reinforce_weight=0.01 \
# \
# reinforce_hourglass.aux_weight=0.1 \
# reinforce_hourglass.use_auxiliary_vocab=true \
# \
# reinforce_hourglass.target_rate_weight=0.01 \
# reinforce_hourglass.use_router_scaling=false \
# \
# reinforce_hourglass.dim=768 \
# reinforce_hourglass.n_head=12 \
# reinforce_hourglass.n_compressor_layers=3 \
# reinforce_hourglass.n_processor_layers=6 \
# reinforce_hourglass.n_decoder_layers=3

# WANDB_MODE=offline \
# accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
# --config-path config \
# --config-name byte.yaml \
# wandb.project="fineweb-nospace-byte" \
# wandb.exp_name="hnet lr 4e-4" \
# dataset.path="/gpfs/data/ranganathlab/Jatin/model_architecture_stuff/data/fineweb-1b-nospace-byte" \
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
# model_type=hnet \
# hnet.block_size=1024 \
# \
# hnet.dim=768 \
# hnet.n_head=12 \
# hnet.n_compressor_layers=3 \
# hnet.n_processor_layers=6 \
# hnet.n_decoder_layers=3


# WANDB_MODE=offline \
# accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
# --config-path config \
# --config-name byte.yaml \
# wandb.project="fineweb-byte" \
# wandb.exp_name="hnet lr 4e-4" \
# dataset.path="/gpfs/data/ranganathlab/Jatin/model_architecture_stuff/data/fineweb-1b-byte" \
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
# model_type=hnet \
# hnet.block_size=1024 \
# \
# hnet.dim=768 \
# hnet.n_head=12 \
# hnet.n_compressor_layers=3 \
# hnet.n_processor_layers=6 \
# hnet.n_decoder_layers=3



echo "Run finished at: "
date
exit