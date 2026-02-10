#!/bin/bash
set -e

echo "=========================================="
echo "  Resume LoRA Training from Checkpoint"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_WEIGHTS_ONLY=0  # PyTorch 2.6 compatibility
S3_BUCKET="${S3_BUCKET:-khlee-healthllm-checkpoints}"

cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"

OUTPUT_DIR="./outputs/healthalpaca-7b-lora"
LOG_DIR="./logs"
mkdir -p $LOG_DIR $OUTPUT_DIR

# S3에서 최신 checkpoint 다운로드 (선택사항)
echo "Checking S3 for latest checkpoint..."
LATEST_S3=$(aws s3 ls s3://$S3_BUCKET/checkpoints/ | grep "PRE checkpoint-" | tail -1 | awk '{print $2}' | sed 's#/##')

if [ -n "$LATEST_S3" ]; then
    echo "Found checkpoint in S3: $LATEST_S3"
    echo "Downloading from S3..."
    aws s3 sync "s3://$S3_BUCKET/checkpoints/$LATEST_S3/" "$OUTPUT_DIR/$LATEST_S3/"
    RESUME_FROM="$OUTPUT_DIR/$LATEST_S3"
else
    # 로컬에서 최신 checkpoint 찾기
    RESUME_FROM=$(ls -td $OUTPUT_DIR/checkpoint-* 2>/dev/null | head -1)
fi

if [ -z "$RESUME_FROM" ]; then
    echo "ERROR: No checkpoint found to resume from!"
    exit 1
fi

echo "Resuming from: $RESUME_FROM"
echo ""

GPU_LOG="$LOG_DIR/gpu_memory_resume_$(date +%Y%m%d_%H%M%S).csv"
echo "timestamp,gpu0_used,gpu1_used,gpu2_used,gpu3_used" > $GPU_LOG

# GPU monitoring
while true; do
    TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
    GPU_INFO=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ',' | sed 's/,$//')
    echo "$TIMESTAMP,$GPU_INFO" >> $GPU_LOG
    sleep 10
done &
MONITOR_PID=$!

# Checkpoint backup
CHECKPOINT_LOG="$LOG_DIR/checkpoint_backup_resume_$(date +%Y%m%d_%H%M%S).log"
LAST_UPLOADED=""
while true; do
    sleep 10
    LATEST=$(ls -td $OUTPUT_DIR/checkpoint-* 2>/dev/null | head -1)
    if [ -n "$LATEST" ] && [ "$LATEST" != "$LAST_UPLOADED" ]; then
        CHECKPOINT_NAME=$(basename "$LATEST")
        echo "[$(date +%H:%M:%S)] Uploading $CHECKPOINT_NAME to S3..." | tee -a $CHECKPOINT_LOG
        aws s3 sync "$LATEST" "s3://$S3_BUCKET/checkpoints/$CHECKPOINT_NAME/" 2>&1 | tee -a $CHECKPOINT_LOG || continue
        echo "[$(date +%H:%M:%S)] ✅ Uploaded: $CHECKPOINT_NAME" | tee -a $CHECKPOINT_LOG
        LAST_UPLOADED="$LATEST"
    fi
done &
BACKUP_PID=$!

TRAIN_LOG="$LOG_DIR/lora_resume_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date +%H:%M:%S)] Starting LoRA training (RESUME)..."

torchrun --nproc_per_node=4 medalpaca/train.py \
    medalpaca/medalpaca-7b \
    --prompt_template medalpaca/prompt_templates/medalpaca.json \
    --data_path data/finetune_data.json \
    --output_dir "$OUTPUT_DIR" \
    --resume_from_checkpoint "$RESUME_FROM" \
    --val_set_size 0 \
    --num_epochs 3 \
    --per_device_batch_size 4 \
    --global_batch_size 128 \
    --learning_rate 2e-5 \
    --warmup_steps 50 \
    --eval_steps 25 \
    --save_steps 25 \
    --save_total_limit 5 \
    --fp16 False \
    --bf16 True \
    --train_in_8bit True \
    --use_lora True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --model_max_length 2048 \
    2>&1 | tee $TRAIN_LOG

EXIT_CODE=${PIPESTATUS[0]}

kill $MONITOR_PID 2>/dev/null || true
kill $BACKUP_PID 2>/dev/null || true

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Resume training SUCCESS"

    echo "Uploading final model to S3..."
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    aws s3 sync "$OUTPUT_DIR" "s3://$S3_BUCKET/final_models/healthalpaca-7b-lora_$TIMESTAMP/"
    aws s3 sync "$OUTPUT_DIR" "s3://$S3_BUCKET/final_models/healthalpaca-7b-lora_latest/" --delete
else
    echo "❌ Resume training FAILED"
fi

echo "Log: $TRAIN_LOG"
