#!/bin/bash
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES="0,1,2,3"

torchrun --master_port=25111 --nproc_per_node=4 ./ddp_infer.py \
    --model_path=ddp_outputs/epoch-0050.ckpt \
    --data_path=./raw_data/pollution_test_data1.csv \
    --seed=42 \
    --num_workers=12 \
    --per_device_test_batch_size=2 \
    --dataloader_drop_last=False \
    --sampler_shuffle=True \
    --model_dtype=bf16 \
    --metric_on_cpu=true