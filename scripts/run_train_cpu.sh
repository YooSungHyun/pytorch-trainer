#!/bin/bash
export WANDB_DISABLED=false
export WANDB_PROJECT="torch-trainer"
export WANDB_ENTITY=""
export WANDB_NAME="cpu(fp32;512batch)-lstm"

python ./cpu_train.py \
    --output_dir=cpu_outputs/ \
    --train_datasets_path=./raw_data/LSTM-Multivariate_pollution.csv \
    --eval_datasets_path=./raw_data/pollution_test_data1.csv \
    --seed=42 \
    --num_workers=12 \
    --per_device_train_batch_size=512 \
    --per_device_eval_batch_size=1 \
    --accumulate_grad_batches=1 \
    --max_epochs=50 \
    --learning_rate=0.001 \
    --weight_decay=0.0001 \
    --warmup_ratio=0.01 \
    --div_factor=25 \
    --final_div_factor=10 \
    --log_every_n=100 \
    --model_dtype="fp32"