#!/bin/bash
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=1
export WANDB_DISABLED=true
export LOCAL_RANK=0
export WANDB_PROJECT="fabric_test"
export WANDB_ENTITY="tadev"
export WANDB_NAME="[bart]test"

python ./train.py \
    --output_dir="model_outputs/" \
    --train_datasets_path="./raw_data/data-02-stock_daily.csv" \
    --train_data_ratio=0.7 \
    --seed=42 \
    --num_workers=12 \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --accumulate_grad_batches=1 \
    --max_epochs=3 \
    --log_every_n_steps=1 \
    --strategy=auto \
    --learning_rate=0.00005 \
    --max_lr=0.0001 \
    --weight_decay=0.0001 \
    --warmup_ratio=0.2 \
    --div_factor=10 \
    --final_div_factor=10