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
schedule.epochs=5 \
data.reverse=false

# ablations
... data.reverse=true
... data.teacherless=true
