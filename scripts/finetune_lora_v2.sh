#!/bin/bash
set -e

echo "=========================================="
echo "  LoRA Fine-tuning (8bit + bf16 + A10G)"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0,1,2,3
S3_BUCKET="${S3_BUCKET:-khlee-healthllm-checkpoints}"

cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"

OUTPUT_DIR="./outputs/healthalpaca-7b-lora"
LOG_DIR="./logs"
mkdir -p $LOG_DIR $OUTPUT_DIR

GPU_LOG="$LOG_DIR/gpu_memory_lora_$(date +%Y%m%d_%H%M%S).csv"
echo "timestamp,gpu0_used,gpu1_used,gpu2_used,gpu3_used" > $GPU_LOG

# GPU monitoring in background
while true; do
    TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
    GPU_INFO=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ',' | sed 's/,$//')
    echo "$TIMESTAMP,$GPU_INFO" >> $GPU_LOG
    sleep 10
done &
MONITOR_PID=$!

echo "[$(date +%H:%M:%S)] GPU monitoring started (PID: $MONITOR_PID)"

# Checkpoint backup to S3 in background (실시간 감지)
CHECKPOINT_LOG="$LOG_DIR/checkpoint_backup_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date +%H:%M:%S)] Starting S3 checkpoint backup (real-time)" | tee -a $CHECKPOINT_LOG

LAST_UPLOADED=""
while true; do
    sleep 10  # 10초마다 체크 (빠른 감지)

    # 가장 최근 checkpoint 찾기
    LATEST=$(ls -td $OUTPUT_DIR/checkpoint-* 2>/dev/null | head -1)

    # 새로운 checkpoint가 생성되었으면 즉시 업로드
    if [ -n "$LATEST" ] && [ "$LATEST" != "$LAST_UPLOADED" ]; then
        CHECKPOINT_NAME=$(basename "$LATEST")
        echo "[$(date +%H:%M:%S)] 🔍 New checkpoint detected: $CHECKPOINT_NAME" | tee -a $CHECKPOINT_LOG

        # 즉시 S3 업로드 (진행상황 표시)
        echo "[$(date +%H:%M:%S)] ⬆️  Uploading to S3..." | tee -a $CHECKPOINT_LOG
        aws s3 sync "$LATEST" "s3://$S3_BUCKET/checkpoints/$CHECKPOINT_NAME/" 2>&1 | tee -a $CHECKPOINT_LOG || {
            echo "[$(date +%H:%M:%S)] ⚠️  Upload failed, will retry" | tee -a $CHECKPOINT_LOG
            continue
        }

        echo "[$(date +%H:%M:%S)] ✅ Uploaded: $CHECKPOINT_NAME" | tee -a $CHECKPOINT_LOG
        LAST_UPLOADED="$LATEST"
    fi
done &
BACKUP_PID=$!

echo "[$(date +%H:%M:%S)] Checkpoint backup started (PID: $BACKUP_PID)"

TRAIN_LOG="$LOG_DIR/lora_finetune_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date +%H:%M:%S)] Starting LoRA training..."

torchrun --nproc_per_node=4 medalpaca/train.py \
    medalpaca/medalpaca-7b \
    --prompt_template medalpaca/prompt_templates/medalpaca.json \
    --data_path data/finetune_data.json \
    --output_dir "$OUTPUT_DIR" \
    --val_set_size 0 \
    --num_epochs 3 \
    --per_device_batch_size 6 \
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

echo ""
echo "=========================================="
echo "  Results"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "LoRA Fine-tuning SUCCESS"
    echo ""

    FINAL_LOSS=$(grep -oP "loss['\"]?:\s*\K[0-9.]+" $TRAIN_LOG | tail -1)
    if [ -n "$FINAL_LOSS" ]; then
        echo "Final Loss: $FINAL_LOSS"
    fi

    echo ""
    echo "Peak GPU Memory:"
    awk -F',' 'NR>1 {
        for(i=2;i<=NF;i++) if($i>max[i]) max[i]=$i
    } END {
        for(i=2;i<=NF;i++) printf "  GPU %d: %d MB\n", i-2, max[i]
    }' $GPU_LOG

    echo ""
    echo "Average GPU Memory:"
    awk -F',' 'NR>1 {
        for(i=2;i<=NF;i++) {sum[i]+=$i; count[i]++}
    } END {
        for(i=2;i<=NF;i++) printf "  GPU %d: %.0f MB\n", i-2, sum[i]/count[i]
    }' $GPU_LOG

    echo ""
    echo "Uploading final model to S3..."

    # 최종 모델 전체 업로드
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    aws s3 sync "$OUTPUT_DIR" "s3://$S3_BUCKET/final_models/healthalpaca-7b-lora_$TIMESTAMP/" || {
        echo "WARNING: S3 upload failed, but model is saved locally"
    }

    # 최신 버전으로도 복사
    aws s3 sync "$OUTPUT_DIR" "s3://$S3_BUCKET/final_models/healthalpaca-7b-lora_latest/" --delete || true

    echo "✓ Model uploaded to S3: s3://$S3_BUCKET/final_models/healthalpaca-7b-lora_$TIMESTAMP/"
    echo "✓ Latest version: s3://$S3_BUCKET/final_models/healthalpaca-7b-lora_latest/"
else
    echo "LoRA Fine-tuning FAILED (Exit code: $EXIT_CODE)"
fi

aws s3 cp $TRAIN_LOG "s3://$S3_BUCKET/logs/$(basename $TRAIN_LOG)" --quiet 2>/dev/null || true
aws s3 cp $GPU_LOG "s3://$S3_BUCKET/logs/$(basename $GPU_LOG)" --quiet 2>/dev/null || true
aws s3 cp $CHECKPOINT_LOG "s3://$S3_BUCKET/logs/$(basename $CHECKPOINT_LOG)" --quiet 2>/dev/null || true

echo ""
echo "=========================================="
echo "  Experiment 2 COMPLETED"
echo "=========================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $TRAIN_LOG"
echo "GPU memory log: $GPU_LOG"
echo "Checkpoint backup log: $CHECKPOINT_LOG"
