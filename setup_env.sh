#!/bin/bash
# EC2 인스턴스 환경 설정 스크립트
# g4dn (T4 테스트) / p4d (A100 학습) 모두 사용 가능
#
# 사용법:
#   bash setup_env.sh
#
# 완료 후:
#   conda activate healthllm
#   bash run_full_finetuning_13b.sh

set -e

ENV_NAME="healthllm"
PYTHON_VER="3.10"

echo "============================================"
echo "Health-LLM 환경 설정 시작"
echo "============================================"

# ── 1. Conda 환경 생성 ──
if conda info --envs | grep -q "${ENV_NAME}"; then
    echo "[SKIP] conda env '${ENV_NAME}' 이미 존재"
else
    echo "[1/4] conda 환경 생성 (Python ${PYTHON_VER})..."
    conda create -n ${ENV_NAME} python=${PYTHON_VER} -y
fi

# conda activate는 스크립트 내에서 직접 안 되므로 경로 직접 사용
CONDA_PREFIX=$(conda info --envs | grep ${ENV_NAME} | awk '{print $NF}')
PIP="${CONDA_PREFIX}/bin/pip"
PYTHON="${CONDA_PREFIX}/bin/python"

echo "  Conda prefix: ${CONDA_PREFIX}"

# ── 2. CUDA Toolkit 설치 (conda로, DeepSpeed 빌드에 필요) ──
echo "[2/4] CUDA Toolkit 설치 (conda)..."
conda install -n ${ENV_NAME} -c conda-forge cudatoolkit-dev -y 2>/dev/null || \
conda install -n ${ENV_NAME} -c nvidia cuda-toolkit -y 2>/dev/null || \
conda install -n ${ENV_NAME} cuda-toolkit -c nvidia/label/cuda-11.8.0 -y

# nvcc 확인
if [ -f "${CONDA_PREFIX}/bin/nvcc" ]; then
    echo "  nvcc 확인: $(${CONDA_PREFIX}/bin/nvcc --version | grep release)"
else
    echo "  [WARNING] nvcc 미설치. 수동 확인 필요."
fi

# ── 3. PyTorch + pip 패키지 설치 ──
echo "[3/4] PyTorch + pip 패키지 설치..."
${PIP} install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# requirements.txt 설치
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
${PIP} install -r "${SCRIPT_DIR}/requirements.txt"

# ── 4. CUDA_HOME 설정 확인 ──
echo "[4/4] CUDA_HOME 설정..."
export CUDA_HOME="${CONDA_PREFIX}"

# DeepSpeed 설치 확인
${PYTHON} -c "import deepspeed; print(f'  DeepSpeed {deepspeed.__version__} OK')"
${PYTHON} -c "import transformers; print(f'  Transformers {transformers.__version__} OK')"
${PYTHON} -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo ""
echo "============================================"
echo "환경 설정 완료!"
echo "============================================"
echo ""
echo "사용법:"
echo "  conda activate ${ENV_NAME}"
echo "  export CUDA_HOME=${CONDA_PREFIX}"
echo "  bash run_full_finetuning_13b.sh"
echo ""
echo "또는 run 스크립트가 CUDA_HOME을 자동 설정합니다."
