#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

# ======================
# Path Configuration
# ======================
MODEL_PATH="Qwen/Qwen3-VL-48-Instruct"
OUTPUT_DIR="./checkpoints"
CACHE_DIR="./cache"

# ======================
# Model Configuration
# ======================
DATASETS="traffic_signs"

# ======================
# Training Launch
# ======================
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_qwen.py \
         --model_name_or_path $MODEL_PATH \
         --tune_mm_llm True \
         --tune_mm_vision False \
         --tune_mm_mlp False \
         --dataset_use $DATASETS \
         --output_dir $OUTPUT_DIR \
         --cache_dir $CACHE_DIR \
         --bf16 \
         --per_device_train_batch_size 1\
         --gradient_accumulation_steps 4 \
         --learning_rate 2e-7 \
         --mm_projector_lr 1e-5 \
         --vision_tower_lr 1e-6 \
         --optim adamw_torch \
         --model_max_length 4096 \
         --data_flatten True \
         --data_packing True \
         --max_pixels $((576*28*28)) \
         --min_pixels $((16*28*28)) \
         --video_fps 2 \
         --video_max_frames 8 \
         --video_min_frames 4 \
         --video_max_pixels $((1664*28*28)) \
         --video_min_pixels $((256*28*28)) \
         --num_train_epochs 3 \
         --warmup_ratio 0.03 \
         --lr_scheduler_type cosine \
         --weight_decay 0.01 \
         --logging_steps 10 \
         --save_steps 500 \
         --save_total_limit 3 \
         --lora_enable True \
         --lora_r 8 \
         --lora_alpha 16 \
         --lora_dropout 0.0 \

