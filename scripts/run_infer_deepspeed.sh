#!/bin/bash
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export HUGGINGFACE_HUB_CACHE="./.cache"
export HF_DATASETS_CACHE="./.cache"

deepspeed --include localhost:0,1,2,3 --master_port 61000 ./ds_infer.py \
    --model_path=ds_outputs/ \
    --data_path=./raw_data/pollution_test_data1.csv \
    --seed=42 \
    --num_workers=12 \
    --per_device_test_batch_size=2 \
    --dataloader_drop_last=False \
    --sampler_shuffle=True \
    --model_dtype=fp16 \
    --metric_on_cpu=false