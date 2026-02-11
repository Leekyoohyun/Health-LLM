#!/usr/bin/env python3
"""
repetition_penalty=1.1 효과 확인
여러 task에서 2개씩 샘플 테스트
입력/출력 전부 출력
"""

import json
import torch
from medalpaca.inferer import Inferer

# ========================================
# Inferer 로드
# ========================================
print("=" * 80)
print("Loading MedAlpaca-7b...")
print("=" * 80)

inferer = Inferer(
    model_name="medalpaca/medalpaca-7b",
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,
    torch_dtype=torch.float16,
)

print("✓ Model loaded\n")

# ========================================
# 테스트할 task들
# ========================================
TASKS = [
    {"name": "PMData_stress", "file": "evaluation-json-data/PMData_stress_train_all.json"},
    {"name": "PMData_readiness", "file": "evaluation-json-data/PMData_readiness_train_all.json"},
    {"name": "PMData_sleep_quality", "file": "evaluation-json-data/PMData_sleep_quality_train_all.json"},
    {"name": "LifeSnaps_stress_resilience", "file": "evaluation-json-data/LifeSnaps_stress_resilience_train_all.json"},
]

print("=" * 80)
print("TEST: repetition_penalty=1.1 (2 samples per task)")
print("=" * 80)

import re

for task in TASKS:
    print(f"\n{'=' * 80}")
    print(f"TASK: {task['name']}")
    print("=" * 80)

    # 데이터 로드
    with open(task['file']) as f:
        data = json.load(f)

    # 처음 2개 샘플 테스트
    for sample_idx in range(min(2, len(data))):
        sample = data[sample_idx]
        instruction = sample['instruction']
        input_text = sample['input']
        ground_truth = sample['output']

        print(f"\n{'─' * 80}")
        print(f"[Sample {sample_idx}]")
        print("─" * 80)

        print(f"\n📋 Instruction:")
        print(f"{instruction[:200]}...")

        print(f"\n📊 Input ({len(input_text)} chars):")
        print(f"{input_text[:400]}...")

        print(f"\n🎯 Ground Truth:")
        print(f"{ground_truth}")

        try:
            answer = inferer(
                instruction=instruction,
                input=input_text,
                max_new_tokens=256,
                repetition_penalty=1.1,
                verbose=False
            )

            # 분석
            is_echo = answer.startswith("The recent")
            is_ai_refusal = "AI language model" in answer or "As an AI" in answer
            numbers = re.findall(r'\b\d+\b', answer)
            has_number = len(numbers) > 0

            print(f"\n💬 Predicted Output ({len(answer)} chars):")
            print(f"{answer}")

            print(f"\n📈 Analysis:")
            print(f"  Echo: {is_echo}")
            print(f"  AI Refusal: {is_ai_refusal}")
            print(f"  Has Number: {has_number}")
            if has_number:
                print(f"  Numbers found: {numbers}")

        except Exception as e:
            print(f"\n❌ ERROR: {e}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
