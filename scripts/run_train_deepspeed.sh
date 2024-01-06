#!/bin/bash
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export WANDB_PROJECT="torch-trainer"
export WANDB_ENTITY=""
export WANDB_NAME="ds-lstm"
export HUGGINGFACE_HUB_CACHE="./.cache"
export HF_DATASETS_CACHE="./.cache"

deepspeed --include localhost:0,1,2,3 --master_port 61000 ./ds_train.py \
    --output_dir=ds_outputs/ \
    --train_datasets_path=./raw_data/LSTM-Multivariate_pollution.csv \
    --eval_datasets_path=./raw_data/pollution_test_data1.csv \
    --seed=42 \
    --num_workers=12 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=1 \
    --accumulate_grad_batches=1 \
    --max_epochs=50 \
    --learning_rate=0.001 \
    --weight_decay=0.0001 \
    --warmup_ratio=0.01 \
    --div_factor=10 \
    --final_div_factor=10 \
    --dataloader_drop_last=False \
    --sampler_shuffle=True \
    --log_every_n=100 \
    --model_dtype=fp16 \
    --deepspeed_config=ds_config/zero2.json