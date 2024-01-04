#!/bin/bash
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=1
export WANDB_DISABLED=false
export WANDB_PROJECT="torch-trainer"
export WANDB_ENTITY=""
export WANDB_NAME="fsdp(4,fp32)-hf_T5"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

torchrun --master_port=25111 --nproc_per_node=4 ./fsdp_train.py \
    --transformers_model_name="t5-base" \
    --output_dir="fsdp_outputs/" \
    --seed=42 \
    --num_workers=12 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --accumulate_grad_batches=1 \
    --max_epochs=10 \
    --learning_rate=0.002 \
    --weight_decay=0.0 \
    --warmup_ratio=0.01 \
    --div_factor=25 \
    --final_div_factor=10 \
    --dataloader_drop_last=False \
    --sampler_shuffle=True \
    --log_every_n=10 \
    --model_dtype=fp32 \
    --group_by_length=true