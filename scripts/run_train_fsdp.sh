#!/bin/bash
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=false
export WANDB_PROJECT="torch-trainer"
export WANDB_ENTITY=""
export WANDB_NAME="fsdp(4,fp32)-hf-imdb"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

torchrun --master_port=25111 --nproc_per_node=4 ./fsdp_train.py \
    --transformers_model_name="roberta-base" \
    --output_dir="fsdp_outputs/" \
    --seed=42 \
    --num_workers=12 \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --accumulate_grad_batches=1 \
    --max_epochs=5 \
    --learning_rate=2e-5 \
    --weight_decay=1e-5 \
    --warmup_ratio=0.01 \
    --div_factor=10 \
    --final_div_factor=10 \
    --dataloader_drop_last=False \
    --sampler_shuffle=True \
    --log_every_n=1 \
    --model_dtype=fp32 \
    --group_by_length=true