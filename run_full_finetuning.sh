#!/bin/bash
# Full Fine-Tuning: MedAlpaca-7b with FSDP + CPU Offload
# 환경: L4 GPU x 4 (24GB each)
#
# 사용법:
#   chmod +x run_full_finetuning.sh
#   ./run_full_finetuning.sh
#
# S3 버킷 설정 (본인 버킷으로 변경)
S3_BUCKET="s3://khlee-healthllm-checkpoints/healthalpaca-full-finetune"

set -e

cd "$(dirname "$0")/medalpaca"

echo "============================================"
echo "Full Fine-Tuning: FSDP + CPU Offload"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l) detected"
echo "S3:   ${S3_BUCKET}"
echo "============================================"

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_full_finetuning.py \
    --model medalpaca/medalpaca-7b \
    --data_path ../data/finetune_data.json \
    --output_dir ../outputs/healthalpaca-7b-full \
    --prompt_template prompts/medalpaca.json \
    --model_max_length 2048 \
    --per_device_batch_size 1 \
    --global_batch_size 32 \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --bf16 True \
    --warmup_steps 100 \
    --save_steps 50 \
    --save_total_limit 2 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --s3_path "${S3_BUCKET}"
