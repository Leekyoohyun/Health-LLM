#!/bin/bash
# Full Fine-Tuning: 13B 모델용 (A100 p4d/p4de)
#
# ===== 사용법 =====
# [T4 테스트] 그대로 실행:
#   bash run_full_finetuning_13b.sh
#
# [A100 본 학습] 아래 변수만 변경:
#   MODEL="medalpaca/medalpaca-13b"
#   BF16=True
#   NUM_GPUS=8, EPOCHS=5, PER_DEVICE_BATCH=4
# ==================

# ── 여기만 바꾸면 됨 ──
MODEL="gpt2"                    # 테스트: gpt2 → 본학습: medalpaca/medalpaca-13b
BF16=False                      # 테스트: False (T4) → 본학습: True (A100)
NUM_GPUS=1                      # 테스트: 1 (T4) → 본학습: 8 (p4d/p4de)
EPOCHS=1                        # 테스트: 1 → 본학습: 5
PER_DEVICE_BATCH=1              # 테스트: 1 (T4) → 본학습: 4 (A100, 논문 설정)
# ──────────────────

GLOBAL_BATCH=128                # 논문 설정: 128

set -e

export CUDA_HOME=${CUDA_HOME:-$(python -c "import sys; print(sys.prefix)")}

cd "$(dirname "$0")/medalpaca"

OUTPUT_DIR="../outputs/test_gpt2"          # 본학습: ../outputs/healthalpaca-13b-full
LOG_FILE="${OUTPUT_DIR}/train.log"
mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "Full Fine-Tuning: ${MODEL}"
echo "GPUs:       ${NUM_GPUS}"
echo "BF16:       ${BF16}"
echo "Epochs:     ${EPOCHS}"
echo "Batch:      per_device=${PER_DEVICE_BATCH}, global=${GLOBAL_BATCH}"
echo "Output:     ${OUTPUT_DIR}"
echo "Log:        ${LOG_FILE}"
echo "============================================"

# tee로 stdout+stderr를 파일에도 저장
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    train_full_finetuning.py \
    --model ${MODEL} \
    --data_path ../data/finetune_data.json \
    --output_dir ${OUTPUT_DIR} \
    --prompt_template prompt_templates/medalpaca.json \
    --model_max_length 2048 \
    --per_device_batch_size ${PER_DEVICE_BATCH} \
    --global_batch_size ${GLOBAL_BATCH} \
    --num_epochs ${EPOCHS} \
    --learning_rate 2e-5 \
    --bf16 ${BF16} \
    --warmup_steps 50 \
    --save_steps 0 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --ds_config ds_config_zero3.json \
    2>&1 | tee -a "${LOG_FILE}"
