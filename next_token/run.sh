# generate (paper or small)
python -m next_token.generate_data --deg=2 --path_len=5 --num_nodes=50 \
    --n_train=200000 --n_test=20000

# paper-default training
WANDB_MODE=offline accelerate launch --config-file accelerate.yaml \
    --mixed_precision=bf16 --num_processes=1 -m next_token.train

# smoke test (just confirmed working: ~12s for 5 epochs)
WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 \
-m next_token.train \
data=paper \
optimizer.lr=1e-4 \
optimizer.min_lr=1e-5 \
schedule.epochs=10 \
eval.every_pct=0.2 \
data.reverse=false


WANDB_MODE=offline \
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 \
-m next_token.train \
data=paper \
model=hybrid \
optimizer.lr=1e-4 \
optimizer.min_lr=1e-5 \
schedule.epochs=10 \
eval.every_pct=0.1 \
data.reverse=false \
data.teacherless=true

# ablations
... data.reverse=true
... data.teacherless=true
