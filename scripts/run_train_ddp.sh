#!/bin/bash
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=1
export WANDB_DISABLED=false
export WANDB_PROJECT="torch-trainer"
export WANDB_ENTITY=""
export WANDB_NAME="ddp(4,fp16,512batch)-lstm"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

torchrun --master_port=25111 --nproc_per_node=4 ./ddp_train.py \
    --output_dir="ddp_outputs/" \
    --train_datasets_path="./raw_data/LSTM-Multivariate_pollution.csv" \
    --eval_datasets_path="./raw_data/pollution_test_data1.csv" \
    --seed=42 \
    --num_workers=12 \
    --per_device_train_batch_size=128 \
    --per_device_eval_batch_size=1 \
    --accumulate_grad_batches=1 \
    --max_epochs=50 \
    --learning_rate=0.001 \
    --weight_decay=0.0001 \
    --warmup_ratio=0.01 \
    --div_factor=25 \
    --final_div_factor=10 \
    --dataloader_drop_last=False \
    --sampler_shuffle=True \
    --log_every_n=100 \
    --model_dtype=bf16