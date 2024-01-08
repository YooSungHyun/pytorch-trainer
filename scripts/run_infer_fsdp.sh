#!/bin/bash
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES="0,1,2,3"

torchrun --master_port=25111 --nproc_per_node=4 ./fsdp_hf_infer.py \
    --transformers_model_name=roberta-base \
    --model_path=fsdp_outputs/epoch-0005/total_model.pt \
    --data_path=imdb \
    --seed=42 \
    --num_workers=12 \
    --per_device_test_batch_size=32 \
    --dataloader_drop_last=False \
    --model_dtype=bf16 \
    --metric_on_cpu=true