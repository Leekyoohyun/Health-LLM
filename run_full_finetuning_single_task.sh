#!/bin/bash
# Full Fine-Tuning: 단일 task로 먼저 테스트
# stress(1421), readiness(1300), sleep_quality(1300), fatigue(1300) 중 택 1
#
# 사용법:
#   ./run_full_finetuning_single_task.sh stress
#   ./run_full_finetuning_single_task.sh readiness

TASK=${1:-stress}
S3_BUCKET="s3://khlee-healthllm-checkpoints/healthalpaca-full-${TASK}"

set -e

# DeepSpeed에 필요한 CUDA_HOME 설정
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

cd "$(dirname "$0")/medalpaca"

echo "============================================"
echo "Full Fine-Tuning: SINGLE TASK (${TASK})"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l) detected"
echo "S3:   ${S3_BUCKET}"
echo "============================================"

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_full_finetuning.py \
    --model medalpaca/medalpaca-7b \
    --data_path ../data/finetune_data.json \
    --output_dir ../outputs/healthalpaca-7b-full-${TASK} \
    --prompt_template prompt_templates/medalpaca.json \
    --model_max_length 2048 \
    --per_device_batch_size 1 \
    --global_batch_size 32 \
    --num_epochs 1 \
    --learning_rate 2e-5 \
    --bf16 True \
    --warmup_steps 50 \
    --save_steps 30 \
    --save_total_limit 2 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --task_filter "${TASK}" \
    --s3_path "${S3_BUCKET}"
