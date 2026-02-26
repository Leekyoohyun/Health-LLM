#!/bin/bash
# 13B 모델 평가
#
# 사용법:
#   bash run_eval_13b.sh baseline              # zeroshot + fewshot (medalpaca-13b)
#   bash run_eval_13b.sh finetuned             # finetuned (healthalpaca-13b-full)
#   bash run_eval_13b.sh baseline --half       # baseline, 반으로 자르기
#   bash run_eval_13b.sh finetuned --half      # finetuned, 반으로 자르기
#
# 필요 인스턴스: g6.12xlarge (L4 × 4, 96GB)
# 예상 시간 (--half): baseline ~2h, finetuned ~1h

set -e

cd "$(dirname "$0")"

# ── 첫 번째 인자: baseline / finetuned ──
RUN_MODE="${1:-}"
if [[ "$RUN_MODE" != "baseline" && "$RUN_MODE" != "finetuned" ]]; then
    echo "Usage: bash run_eval_13b.sh <baseline|finetuned> [--half]"
    exit 1
fi

# ── 두 번째 인자: --half ──
MAX_SAMPLES=""
SUFFIX=""
if [[ "$2" == "--half" ]]; then
    MAX_SAMPLES="--max_samples 0.5"
    SUFFIX="_half"
    echo "Samples: HALF"
else
    echo "Samples: FULL"
fi

BASE_MODEL="medalpaca/medalpaca-13b"
FT_MODEL="outputs/healthalpaca-13b-full"
TRAIN_DATA="data/finetune_data.json"
RESULT_DIR="results_13b"
mkdir -p $RESULT_DIR

echo "============================================"
echo "  13B Evaluation: $RUN_MODE"
echo "============================================"

if [[ "$RUN_MODE" == "baseline" ]]; then
    # ── ZeroShot ──
    echo ""
    echo "[1/2] ZeroShot (medalpaca-13b)..."
    python evaluate_13b.py \
        --mode zeroshot \
        --model $BASE_MODEL \
        --output "$RESULT_DIR/results_13b_zeroshot${SUFFIX}.json" \
        $MAX_SAMPLES \
        2>&1 | tee "$RESULT_DIR/log_zeroshot${SUFFIX}.txt"

    # ── FewShot (같은 모델, HF 캐시 재사용) ──
    echo ""
    echo "[2/2] FewShot (medalpaca-13b)..."
    python evaluate_13b.py \
        --mode fewshot \
        --model $BASE_MODEL \
        --train_data $TRAIN_DATA \
        --output "$RESULT_DIR/results_13b_fewshot${SUFFIX}.json" \
        $MAX_SAMPLES \
        2>&1 | tee "$RESULT_DIR/log_fewshot${SUFFIX}.txt"

elif [[ "$RUN_MODE" == "finetuned" ]]; then
    # ── Finetuned ──
    echo ""
    echo "[1/1] Finetuned (healthalpaca-13b-full)..."
    python evaluate_13b.py \
        --mode finetuned \
        --model $FT_MODEL \
        --output "$RESULT_DIR/results_13b_finetuned${SUFFIX}.json" \
        $MAX_SAMPLES \
        2>&1 | tee "$RESULT_DIR/log_finetuned${SUFFIX}.txt"
fi

# ── 결과 요약 ──
echo ""
echo "============================================"
echo "  DONE: $RUN_MODE"
echo "============================================"
echo "Results:"
ls -lh $RESULT_DIR/results_13b_*${SUFFIX}.json 2>/dev/null || echo "  (no results)"
echo ""
echo "Next: aws s3 sync $RESULT_DIR/ s3://khlee-healthllm-checkpoints/results_13b/"
echo "============================================"
