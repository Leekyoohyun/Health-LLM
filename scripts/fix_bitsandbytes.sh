#!/bin/bash
# fix_bitsandbytes.sh - Fix bitsandbytes CUDA detection issue
set -e

echo "=========================================="
echo "  bitsandbytes CUDA 경로 수정"
echo "=========================================="

# 1. CUDA 경로 찾기
echo "[1/4] CUDA 라이브러리 경로 찾기..."

# nvidia-smi로 CUDA 버전 확인
CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
echo "  감지된 CUDA 버전: $CUDA_VERSION"

# 가능한 경로들 시도
POSSIBLE_PATHS=(
    "/usr/local/cuda-${CUDA_VERSION}/lib64"
    "/usr/local/cuda/lib64"
    "/usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib"
    "/usr/lib/x86_64-linux-gnu"
    "/usr/local/cuda/targets/x86_64-linux/lib"
)

CUDA_LIB_PATH=""
for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -f "$path/libcudart.so" ] || ls "$path"/libcudart.so* &>/dev/null; then
        CUDA_LIB_PATH="$path"
        echo "  ✅ CUDA 라이브러리 발견: $CUDA_LIB_PATH"
        break
    fi
done

if [ -z "$CUDA_LIB_PATH" ]; then
    echo ""
    echo "❌ libcudart.so를 찾을 수 없습니다!"
    echo ""
    echo "수동으로 찾기:"
    echo "  find /usr -name 'libcudart.so*' 2>/dev/null"
    exit 1
fi

# 2. LD_LIBRARY_PATH 설정
echo ""
echo "[2/4] LD_LIBRARY_PATH 환경변수 설정..."
export LD_LIBRARY_PATH="$CUDA_LIB_PATH:$LD_LIBRARY_PATH"
echo "  export LD_LIBRARY_PATH=\"$CUDA_LIB_PATH:\$LD_LIBRARY_PATH\""

# 3. ~/.bashrc에 영구 추가
echo ""
echo "[3/4] ~/.bashrc에 환경변수 추가..."
BASHRC_LINE="export LD_LIBRARY_PATH=\"$CUDA_LIB_PATH:\$LD_LIBRARY_PATH\""

if ! grep -qF "$CUDA_LIB_PATH" ~/.bashrc; then
    echo "$BASHRC_LINE" >> ~/.bashrc
    echo "  ✅ ~/.bashrc에 추가됨"
else
    echo "  이미 설정되어 있음"
fi

# 4. bitsandbytes 재설치
echo ""
echo "[4/4] bitsandbytes 재설치 중..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate healthllm

pip uninstall -y bitsandbytes
pip install bitsandbytes==0.39.0

echo ""
echo "=========================================="
echo "  bitsandbytes 수정 완료!"
echo "=========================================="
echo ""
echo "환경변수 확인:"
echo "  echo \$LD_LIBRARY_PATH"
echo ""
echo "다음 단계:"
echo "  1. source ~/.bashrc  (또는 새 터미널)"
echo "  2. conda activate healthllm"
echo "  3. cd ~/Health-LLM/scripts"
echo "  4. bash finetune_lora_v2.sh"
