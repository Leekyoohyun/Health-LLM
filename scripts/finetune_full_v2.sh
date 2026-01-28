#!/bin/bash
# finetune_full_v2.sh - Full Fine-tuning 실험 (OOM 테스트)
# 실행: bash scripts/finetune_full_v2.sh

set -e

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3
S3_BUCKET="${S3_BUCKET:-khlee-healthllm-checkpoints}"

# 작업 디렉토리 확인
cd "$(dirname "$0")/.."
echo "작업 디렉토리: $(pwd)"

# GPU 메모리 모니터링 시작 (백그라운드)
LOG_DIR="./logs"
mkdir -p $LOG_DIR

GPU_LOG="$LOG_DIR/gpu_memory_full_$(date +%Y%m%d_%H%M%S).csv"
echo "timestamp,gpu0_used,gpu1_used,gpu2_used,gpu3_used" > $GPU_LOG

(
    while true; do
        TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
        GPU_INFO=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ',' | sed 's/,$//')
        echo "$TIMESTAMP,$GPU_INFO" >> $GPU_LOG
        sleep 2
    done
) &
MONITOR_PID=$!
echo "[$(date +%H:%M:%S)] GPU 모니터링 시작 (PID: $MONITOR_PID)"

# 학습 로그 파일
TRAIN_LOG="$LOG_DIR/full_finetune_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date +%H:%M:%S)] Full Fine-tuning 시작..."
echo ""

# Full Fine-tuning 실행 (4 GPU 분산 학습)
torchrun --nproc_per_node=4 medalpaca/train.py \
    medalpaca/medalpaca-7b \
    --prompt_template medalpaca/prompt_templates/medalpaca.json \
    --data_path data/finetune_data.json \
    --output_dir ./outputs/healthalpaca-7b-full \
    --num_epochs 3 \
    --per_device_batch_size 1 \
    --global_batch_size 128 \
    --learning_rate 2e-5 \
    --warmup_steps 50 \
    --eval_steps 50 \
    --save_total_limit 1 \
    --fp16 False \
    --bf16 True \
    --gradient_checkpointing True \
    --use_lora False \
    --train_in_8bit False \
    2>&1 | tee $TRAIN_LOG

EXIT_CODE=${PIPESTATUS[0]}

# 모니터링 종료
kill $MONITOR_PID 2>/dev/null || true
echo ""
echo "[$(date +%H:%M:%S)] GPU 모니터링 종료"

# 결과 분석
echo ""
echo "=========================================="
echo "  Full Fine-tuning 결과 분석"
echo "=========================================="

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Full Fine-tuning 실패 (Exit code: $EXIT_CODE)"
    echo ""

    # OOM 확인
    if grep -q "CUDA out of memory\|OutOfMemoryError" $TRAIN_LOG; then
        echo "📊 OOM 발생 확인!"
        echo ""
        echo "에러 메시지:"
        grep -A 2 "CUDA out of memory\|OutOfMemoryError" $TRAIN_LOG | head -10
        echo ""
    fi

    # Peak Memory 출력
    echo "Peak GPU Memory Usage:"
    awk -F',' 'NR>1 {
        for(i=2;i<=NF;i++) if($i>max[i]) max[i]=$i
    } END {
        for(i=2;i<=NF;i++) printf "  GPU %d: %d MB\n", i-2, max[i]
    }' $GPU_LOG
else
    echo "✅ Full Fine-tuning 성공"
    echo "   7B 모델 전체 학습이 24GB GPU에서 가능"
fi

# S3에 로그 업로드
echo ""
echo "[$(date +%H:%M:%S)] 로그를 S3에 업로드 중..."
aws s3 cp $TRAIN_LOG "s3://$S3_BUCKET/logs/$(basename $TRAIN_LOG)" --quiet || echo "S3 업로드 실패 (무시)"
aws s3 cp $GPU_LOG "s3://$S3_BUCKET/logs/$(basename $GPU_LOG)" --quiet || echo "S3 업로드 실패 (무시)"

echo ""
echo "=========================================="
echo "  실험 1 완료"
echo "=========================================="
echo ""
echo "로그 파일: $TRAIN_LOG"
echo "GPU 메모리 로그: $GPU_LOG"
echo ""
echo "다음 단계: bash scripts/finetune_lora_v2.sh"
