#!/usr/bin/env python3
"""
Inferer 클래스로 repetition_penalty 효과 테스트
PMData_stress 첫 3개 샘플로 비교:
1. 기본값 (repetition_penalty 없음)
2. repetition_penalty=1.5
"""

import json
import torch
from medalpaca.inferer import Inferer

# ========================================
# Inferer 로드
# ========================================
print("=" * 80)
print("Loading MedAlpaca-7b with Inferer...")
print("=" * 80)

inferer = Inferer(
    model_name="medalpaca/medalpaca-7b",
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,
    torch_dtype=torch.float16,
)

print("✓ Model loaded\n")

# ========================================
# 데이터 로드 (PMData_stress 처음 3개)
# ========================================
with open("evaluation-json-data/PMData_stress_train_all.json") as f:
    data = json.load(f)

print("=" * 80)
print("TEST: PMData_stress (first 3 samples)")
print("=" * 80)

for sample_idx in range(3):
    sample = data[sample_idx]
    instruction = sample['instruction']
    input_text = sample['input']
    ground_truth = sample['output']

    print(f"\n{'=' * 80}")
    print(f"Sample {sample_idx}")
    print("=" * 80)
    print(f"Ground truth: {ground_truth}")
    print(f"Input length: {len(input_text)} chars")

    # ========================================
    # Test 1: 기본값 (repetition_penalty 없음)
    # ========================================
    print("\n[Test 1] 기본값 (no repetition_penalty):")
    try:
        answer_default = inferer(
            instruction=instruction,
            input=input_text,
            max_new_tokens=256,
            verbose=False
        )
        print(f"✅ Output: {answer_default[:200]}{'...' if len(answer_default) > 200 else ''}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        answer_default = None

    # ========================================
    # Test 2: repetition_penalty=1.5
    # ========================================
    print("\n[Test 2] repetition_penalty=1.5:")
    try:
        answer_with_penalty = inferer(
            instruction=instruction,
            input=input_text,
            max_new_tokens=256,
            repetition_penalty=1.5,
            verbose=False
        )
        print(f"✅ Output: {answer_with_penalty[:200]}{'...' if len(answer_with_penalty) > 200 else ''}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        answer_with_penalty = None

    # ========================================
    # 비교
    # ========================================
    if answer_default and answer_with_penalty:
        is_echo_default = answer_default.startswith("The recent")
        is_echo_penalty = answer_with_penalty.startswith("The recent")

        print(f"\n📊 Comparison:")
        print(f"  Default:  Echo={is_echo_default}, Length={len(answer_default)}")
        print(f"  Penalty:  Echo={is_echo_penalty}, Length={len(answer_with_penalty)}")

        if is_echo_default and not is_echo_penalty:
            print(f"  ✅ repetition_penalty=1.5 solved echo problem!")
        elif not is_echo_default and not is_echo_penalty:
            print(f"  ✅ Both work (no echo)")
        elif is_echo_default and is_echo_penalty:
            print(f"  ❌ Both have echo problem")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print("\n💡 If repetition_penalty=1.5 consistently solves echo:")
print("   → Add it to evaluate_baseline_full.py")
print("\n💡 If both have echo:")
print("   → Need different approach (model limitation)")
print("=" * 80)
