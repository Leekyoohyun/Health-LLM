#!/bin/bash
# 13B 모델 3가지 평가 일괄 실행
#
# 사용법:
#   bash run_eval_13b.sh                  # 전체 샘플
#   bash run_eval_13b.sh --half           # 반으로 자르기 (PMData 72, AW_FB 156)
#
# 필요 인스턴스: g6.12xlarge (L4 × 4, 96GB)
# 예상 시간: 전체 ~6h, 반 ~3h
#
# 모델 로딩:
#   [프로세스 1] medalpaca-13b → zeroshot 평가 → 종료 (GPU 해제)
#   [프로세스 2] medalpaca-13b → fewshot 평가 → 종료 (GPU 해제)
#   [프로세스 3] healthalpaca-13b-full → finetuned 평가 → 종료
#   ※ 각 Python 프로세스 종료 시 CUDA 메모리 완전 해제됨
#   ※ HF 캐시 덕분에 2번째 로딩은 다운로드 없이 빠름

set -e

cd "$(dirname "$0")"

# ── 옵션 파싱 ──
MAX_SAMPLES=""
SUFFIX=""
if [[ "$1" == "--half" ]]; then
    MAX_SAMPLES="--max_samples 0.5"
    SUFFIX="_half"
    echo "Mode: HALF samples (PMData 72, AW_FB 156)"
else
    echo "Mode: FULL samples (PMData 144, AW_FB 313)"
fi

BASE_MODEL="medalpaca/medalpaca-13b"
FT_MODEL="outputs/healthalpaca-13b-full"
TRAIN_DATA="data/finetune_data.json"
RESULT_DIR="results_13b"
mkdir -p $RESULT_DIR

echo "============================================"
echo "  13B Model Evaluation (3 modes)"
echo "============================================"
echo "Base model:     $BASE_MODEL"
echo "Finetuned:      $FT_MODEL"
echo "Train data:     $TRAIN_DATA (fewshot용)"
echo "Results dir:    $RESULT_DIR"
echo "============================================"

# ── 1. ZeroShot baseline ──
echo ""
echo "[1/3] ZeroShot Baseline..."
python evaluate_13b.py \
    --mode zeroshot \
    --model $BASE_MODEL \
    --output "$RESULT_DIR/results_13b_zeroshot${SUFFIX}.json" \
    $MAX_SAMPLES \
    2>&1 | tee "$RESULT_DIR/log_zeroshot${SUFFIX}.txt"

# ── 2. FewShot baseline (같은 모델, HF 캐시에서 빠르게 로드) ──
echo ""
echo "[2/3] FewShot Baseline..."
python evaluate_13b.py \
    --mode fewshot \
    --model $BASE_MODEL \
    --train_data $TRAIN_DATA \
    --output "$RESULT_DIR/results_13b_fewshot${SUFFIX}.json" \
    $MAX_SAMPLES \
    2>&1 | tee "$RESULT_DIR/log_fewshot${SUFFIX}.txt"

# ── 3. Finetuned (다른 모델) ──
echo ""
echo "[3/3] Finetuned..."
python evaluate_13b.py \
    --mode finetuned \
    --model $FT_MODEL \
    --output "$RESULT_DIR/results_13b_finetuned${SUFFIX}.json" \
    $MAX_SAMPLES \
    2>&1 | tee "$RESULT_DIR/log_finetuned${SUFFIX}.txt"

# ── 결과 요약 ──
echo ""
echo "============================================"
echo "  ALL DONE"
echo "============================================"
echo "Results:"
ls -lh $RESULT_DIR/results_13b_*${SUFFIX}.json
echo ""
echo "Next: S3로 결과 업로드"
echo "  aws s3 sync $RESULT_DIR/ s3://khlee-healthllm-checkpoints/results_13b/"
echo ""
echo "Next: MAE/Accuracy 계산"
echo "  python calculate_baseline_mae.py --input $RESULT_DIR/results_13b_zeroshot${SUFFIX}.json --mode zeroshot"
echo "  python calculate_baseline_mae.py --input $RESULT_DIR/results_13b_fewshot${SUFFIX}.json --mode fewshot"
echo "  python calculate_baseline_mae.py --input $RESULT_DIR/results_13b_finetuned${SUFFIX}.json --mode finetuned"
echo "============================================"
