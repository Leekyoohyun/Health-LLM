#!/bin/bash
# finetune_lora.sh - LoRA Fine-tuning 실험 (주요 실험)
# 목적: 실제 학습 완료 및 모델 성능 검증
# 예상 결과: 성공 (~20분)
#
# 실행: bash scripts/finetune_lora.sh

set -e

echo "=========================================="
echo "  실험 2: LoRA Fine-tuning (주요 실험)"
echo "=========================================="
echo ""

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3
S3_BUCKET="${S3_BUCKET:-healthllm-checkpoints-CHANGE-ME}"

# 작업 디렉토리 확인
cd "$(dirname "$0")/.."
echo "작업 디렉토리: $(pwd)"

OUTPUT_DIR="./outputs/healthalpaca-7b-lora"
LOG_DIR="./logs"
mkdir -p $LOG_DIR $OUTPUT_DIR

# S3에서 기존 체크포인트 복구 (있는 경우)
echo "[$(date +%H:%M:%S)] S3에서 기존 체크포인트 확인 중..."
aws s3 sync "s3://$S3_BUCKET/lora_checkpoints/" "$OUTPUT_DIR/" --quiet 2>/dev/null || echo "  → 기존 체크포인트 없음 (새로 시작)"

# 마지막 체크포인트 경로 확인
RESUME_PATH=""
if [ -d "$OUTPUT_DIR" ]; then
    LATEST_CKPT=$(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST_CKPT" ]; then
        RESUME_PATH="--resume_from_checkpoint $LATEST_CKPT"
        echo "[$(date +%H:%M:%S)] 체크포인트에서 재개: $LATEST_CKPT"
    fi
fi

# GPU 메모리 모니터링 시작 (백그라운드)
GPU_LOG="$LOG_DIR/gpu_memory_lora_$(date +%Y%m%d_%H%M%S).csv"
echo "timestamp,gpu0_used,gpu1_used,gpu2_used,gpu3_used" > $GPU_LOG

(
    while true; do
        TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
        GPU_INFO=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ',' | sed 's/,$//')
        echo "$TIMESTAMP,$GPU_INFO" >> $GPU_LOG
        sleep 10
    done
) &
MONITOR_PID=$!
echo "[$(date +%H:%M:%S)] GPU 모니터링 시작 (PID: $MONITOR_PID)"

# 학습 로그 파일
TRAIN_LOG="$LOG_DIR/lora_finetune_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date +%H:%M:%S)] LoRA Fine-tuning 시작..."
echo ""

# LoRA Fine-tuning 실행 (4 GPU 분산 학습)
torchrun --nproc_per_node=4 medalpaca/train.py \
    --model_name "medalpaca/medalpaca-7b" \
    --data_path "data/finetune_data.json" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --warmup_steps 50 \
    --logging_steps 10 \
    --save_steps 50 \
    --save_total_limit 3 \
    --bf16 True \
    --load_in_8bit True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    $RESUME_PATH \
    2>&1 | tee $TRAIN_LOG

EXIT_CODE=${PIPESTATUS[0]}

# 모니터링 종료
kill $MONITOR_PID 2>/dev/null || true
echo ""
echo "[$(date +%H:%M:%S)] GPU 모니터링 종료"

# 결과 분석
echo ""
echo "=========================================="
echo "  LoRA Fine-tuning 결과 분석"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ LoRA Fine-tuning 성공!"
    echo ""

    # 최종 Loss 출력
    FINAL_LOSS=$(grep -oP "loss['\"]?:\s*\K[0-9.]+" $TRAIN_LOG | tail -1)
    if [ -n "$FINAL_LOSS" ]; then
        echo "📊 Final Loss: $FINAL_LOSS"
    fi

    # Peak Memory 출력
    echo ""
    echo "Peak GPU Memory Usage:"
    awk -F',' 'NR>1 {
        for(i=2;i<=NF;i++) if($i>max[i]) max[i]=$i
    } END {
        for(i=2;i<=NF;i++) printf "  GPU %d: %d MB\n", i-2, max[i]
    }' $GPU_LOG

    # Average Memory 출력
    echo ""
    echo "Average GPU Memory Usage:"
    awk -F',' 'NR>1 {
        for(i=2;i<=NF;i++) {sum[i]+=$i; count[i]++}
    } END {
        for(i=2;i<=NF;i++) printf "  GPU %d: %.0f MB\n", i-2, sum[i]/count[i]
    }' $GPU_LOG

    # S3 최종 동기화
    echo ""
    echo "[$(date +%H:%M:%S)] 최종 모델을 S3에 업로드 중..."
    aws s3 sync "$OUTPUT_DIR" "s3://$S3_BUCKET/lora_checkpoints/" --quiet
    aws s3 sync "$OUTPUT_DIR" "s3://$S3_BUCKET/final_model_lora/" --quiet
    echo "✅ 모델 저장 완료: s3://$S3_BUCKET/final_model_lora/"
else
    echo "❌ LoRA Fine-tuning 실패 (Exit code: $EXIT_CODE)"
    echo ""
    echo "에러 로그 확인: tail -50 $TRAIN_LOG"
fi

# S3에 로그 업로드
echo ""
echo "[$(date +%H:%M:%S)] 로그를 S3에 업로드 중..."
aws s3 cp $TRAIN_LOG "s3://$S3_BUCKET/logs/$(basename $TRAIN_LOG)" --quiet || echo "S3 업로드 실패 (무시)"
aws s3 cp $GPU_LOG "s3://$S3_BUCKET/logs/$(basename $GPU_LOG)" --quiet || echo "S3 업로드 실패 (무시)"

echo ""
echo "=========================================="
echo "  실험 2 완료"
echo "=========================================="
echo ""
echo "출력 디렉토리: $OUTPUT_DIR"
echo "로그 파일: $TRAIN_LOG"
echo "GPU 메모리 로그: $GPU_LOG"
echo ""
echo "모델 파일 확인:"
ls -lh $OUTPUT_DIR/ 2>/dev/null || echo "  (출력 디렉토리 없음)"
