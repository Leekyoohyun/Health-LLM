#!/bin/bash
# Full Fine-Tuning: 13B 모델용 (A100 p4d/p4de)
#
# ===== 사용법 =====
# [T4 테스트] 그대로 실행:
#   bash run_full_finetuning_13b.sh
#
# [A100 본 학습] 아래 두 줄만 변경:
#   MODEL="medalpaca/medalpaca-13b"
#   BF16=True
#   그리고 NUM_GPUS, EPOCHS 조정
# ==================

# ── 여기만 바꾸면 됨 ──
MODEL="gpt2"                    # 테스트: gpt2 → 본학습: medalpaca/medalpaca-13b
BF16=False                      # 테스트: False (T4) → 본학습: True (A100)
NUM_GPUS=1                      # 테스트: 1 (T4) → 본학습: 8 (p4d/p4de)
EPOCHS=1                        # 테스트: 1 → 본학습: 5
TASK=${1:-stress}
# ──────────────────

S3_BUCKET="s3://khlee-healthllm-checkpoints/healthalpaca-full-13b-${TASK}"
GLOBAL_BATCH=32
PER_DEVICE_BATCH=1

set -e

export CUDA_HOME=${CUDA_HOME:-$(python -c "import sys; print(sys.prefix)")}

cd "$(dirname "$0")/medalpaca"

echo "============================================"
echo "Full Fine-Tuning: ${MODEL}"
echo "Task:   ${TASK}"
echo "GPUs:   ${NUM_GPUS}"
echo "BF16:   ${BF16}"
echo "Epochs: ${EPOCHS}"
echo "S3:     ${S3_BUCKET}"
echo "============================================"

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    train_full_finetuning.py \
    --model ${MODEL} \
    --data_path ../data/finetune_data.json \
    --output_dir ../outputs/healthalpaca-13b-full-${TASK} \
    --prompt_template prompt_templates/medalpaca.json \
    --model_max_length 2048 \
    --per_device_batch_size ${PER_DEVICE_BATCH} \
    --global_batch_size ${GLOBAL_BATCH} \
    --num_epochs ${EPOCHS} \
    --learning_rate 2e-5 \
    --bf16 ${BF16} \
    --warmup_steps 50 \
    --save_steps 100 \
    --save_total_limit 3 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --task_filter "${TASK}" \
    --s3_path "${S3_BUCKET}" \
    --ds_config ds_config_zero3.json
