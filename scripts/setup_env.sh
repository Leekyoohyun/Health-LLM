#!/bin/bash
# setup_env.sh - AWS EC2 환경 설정 스크립트
# 실행: bash scripts/setup_env.sh

set -e  # 에러 발생 시 중단

echo "=========================================="
echo "  Health-LLM 환경 설정 시작"
echo "=========================================="

# 1. S3 버킷 이름 설정 (사용자가 수정해야 함)
S3_BUCKET="${S3_BUCKET:-healthllm-checkpoints-CHANGE-ME}"

if [[ "$S3_BUCKET" == *"CHANGE-ME"* ]]; then
    echo "⚠️  S3_BUCKET 환경변수를 설정해주세요!"
    echo "   export S3_BUCKET=\"healthllm-checkpoints-<your-id>\""
    echo "   또는 이 스크립트의 S3_BUCKET 값을 직접 수정하세요."
    exit 1
fi

echo "[1/6] S3 버킷: $S3_BUCKET"

# 2. Conda 환경 생성
echo "[2/6] Conda 환경 생성 중..."
if conda info --envs | grep -q "healthllm"; then
    echo "  → healthllm 환경이 이미 존재합니다. 스킵."
else
    conda create -n healthllm python=3.10 -y
    echo "  → healthllm 환경 생성 완료"
fi

# 3. Conda 환경 활성화 및 패키지 설치
echo "[3/6] 패키지 설치 중..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate healthllm

pip install --upgrade pip
pip install -r requirements.txt

echo "  → 패키지 설치 완료"

# 4. 환경 변수 설정
echo "[4/6] 환경 변수 설정 중..."
if ! grep -q "S3_BUCKET" ~/.bashrc; then
    echo "export S3_BUCKET=\"$S3_BUCKET\"" >> ~/.bashrc
    echo "  → S3_BUCKET 환경변수 추가됨"
else
    echo "  → S3_BUCKET 환경변수가 이미 존재합니다"
fi

export S3_BUCKET="$S3_BUCKET"

# 5. S3 연결 테스트
echo "[5/6] S3 연결 테스트 중..."
if aws s3 ls "s3://$S3_BUCKET/" > /dev/null 2>&1; then
    echo "  → S3 버킷 접근 성공 ✅"
else
    echo "  → S3 버킷 접근 실패 ❌"
    echo "    IAM Role이 EC2에 연결되었는지 확인하세요."
    exit 1
fi

# 6. GPU 확인
echo "[6/6] GPU 확인 중..."
nvidia-smi --query-gpu=name,memory.total --format=csv
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "  → GPU 개수: $GPU_COUNT"

if [ "$GPU_COUNT" -ne 4 ]; then
    echo "  ⚠️  경고: 4개의 GPU가 예상되지만 ${GPU_COUNT}개가 감지되었습니다."
fi

echo ""
echo "=========================================="
echo "  환경 설정 완료! ✅"
echo "=========================================="
echo ""
echo "다음 단계:"
echo "  1. conda activate healthllm"
echo "  2. 데이터 파일 확인: ls -la data/finetune_data.json"
echo "  3. 실험 1 실행: bash scripts/finetune_full.sh"
echo "  4. 실험 2 실행: bash scripts/finetune_lora.sh"
