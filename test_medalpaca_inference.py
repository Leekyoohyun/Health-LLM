#!/usr/bin/env python3
"""
MedAlpaca inference 테스트 스크립트
- pipeline으로 medalpaca_pl 정의
- 샘플 몇 개만 실행해서 정상 동작 확인
"""

import json
from transformers import pipeline

# ========================================
# medalpaca_pl 정의 (HuggingFace pipeline)
# ========================================
print("Loading MedAlpaca-7b with pipeline...")
medalpaca_pl = pipeline(
    "text-generation",
    model="medalpaca/medalpaca-7b",
    tokenizer="medalpaca/medalpaca-7b",
    device=0,  # GPU 0
    max_new_tokens=128,
)
print("✓ Model loaded\n")


# ========================================
# 테스트 데이터 로드
# ========================================
print("Loading test data...")
data_file = "evaluation-json-data/PMData_stress_train_all.json"
with open(data_file) as f:
    data = json.load(f)

print(f"✓ Loaded {len(data)} samples from {data_file}\n")


# ========================================
# 샘플 몇 개만 테스트
# ========================================
NUM_TEST_SAMPLES = 3

print(f"Testing {NUM_TEST_SAMPLES} samples...\n")
print("=" * 80)

for i in range(NUM_TEST_SAMPLES):
    sample = data[i]
    question = sample['input']
    ground_truth = sample['output']

    print(f"\n[Sample {i+1}]")
    print(f"Question (first 200 chars): {question[:200]}...")
    print(f"Ground truth: {ground_truth}")

    try:
        # inference 실행
        result = medalpaca_pl(question)
        ans = result[0]['generated_text']

        print(f"✓ SUCCESS")
        print(f"Generated (first 500 chars): {ans[:500]}...")

    except Exception as e:
        ans = "N/A"
        print(f"❌ EXCEPTION: {type(e).__name__}: {str(e)[:100]}")
        print(f"Answer set to: N/A")

    print("-" * 80)

print("\n" + "=" * 80)
print("Test complete!")
