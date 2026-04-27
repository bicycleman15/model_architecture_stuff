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
model=transformer \
optimizer.lr=1e-4 \
optimizer.min_lr=1e-5 \
schedule.epochs=10 \
eval.every_pct=0.1 \
data.reverse=false \
data.teacherless=false \
nextlat.enabled=true


# match nextLat paper
WANDB_MODE=offline \
accelerate launch \
--config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 \
-m next_token.train \
logging.project="learnability-star-graph" \
logging.name="test vanilla" \
data=paper \
data.num_nodes=100 \
\
model=transformer \
model.n_layer=12 \
model.n_head=6 \
model.dim=384 \
\
batch_size=512 \
optimizer.lr=5e-4 \
optimizer.min_lr=5e-5 \
optimizer.weight_decay=0.1 \
schedule.epochs=50 \
\
nextlat.enabled=true \
nextlat.lambda_h=1.0 \
nextlat.lambda_kl=1.0 \
nextlat.n_hidden_layers=3 \
nextlat.hidden_mult=1

# ablations
... data.reverse=true
... data.teacherless=true
