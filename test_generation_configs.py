#!/usr/bin/env python3
"""
여러 generation config 조합 테스트
repetition_penalty, do_sample, eos_token_id 등
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

print(f"✓ Model loaded")
print(f"EOS token ID: {inferer.data_handler.tokenizer.eos_token_id}")
print(f"PAD token ID: {inferer.data_handler.tokenizer.pad_token_id}")
print()

# ========================================
# 데이터 로드 (PMData_stress 첫 2개)
# ========================================
with open("evaluation-json-data/PMData_stress_train_all.json") as f:
    data = json.load(f)

# ========================================
# 테스트할 설정들
# ========================================
configs = [
    {"name": "기본값", "params": {}},
    {"name": "penalty=1.1", "params": {"repetition_penalty": 1.1}},
    {"name": "penalty=1.2", "params": {"repetition_penalty": 1.2}},
    {"name": "penalty=1.3", "params": {"repetition_penalty": 1.3}},
    {"name": "penalty=1.5", "params": {"repetition_penalty": 1.5}},
    {"name": "penalty=1.2+do_sample", "params": {"repetition_penalty": 1.2, "do_sample": True, "temperature": 0.7}},
]

print("=" * 80)
print("TEST: PMData_stress (first 2 samples)")
print("=" * 80)

for sample_idx in range(2):
    sample = data[sample_idx]
    instruction = sample['instruction']
    input_text = sample['input']
    ground_truth = sample['output']

    print(f"\n{'=' * 80}")
    print(f"Sample {sample_idx}")
    print("=" * 80)
    print(f"Ground truth: {ground_truth}")
    print(f"Input length: {len(input_text)} chars")

    results = []

    for config in configs:
        print(f"\n[{config['name']}]:")
        try:
            answer = inferer(
                instruction=instruction,
                input=input_text,
                max_new_tokens=256,
                verbose=False,
                **config['params']
            )

            # Echo 여부
            is_echo = answer.startswith("The recent")
            is_ai_refusal = "AI language model" in answer or "As an AI" in answer

            # 숫자 추출 시도
            import re
            numbers = re.findall(r'\b\d+\b', answer)
            has_number = len(numbers) > 0

            print(f"  Output: {answer[:150]}...")
            print(f"  Echo: {is_echo} | AI Refusal: {is_ai_refusal} | Has Number: {has_number}")

            results.append({
                'config': config['name'],
                'echo': is_echo,
                'ai_refusal': is_ai_refusal,
                'has_number': has_number,
                'length': len(answer),
                'output': answer
            })

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append({
                'config': config['name'],
                'error': str(e)
            })

    # 요약
    print(f"\n📊 Summary for Sample {sample_idx}:")
    print(f"  {'Config':<25} {'Echo':<8} {'AI Refusal':<12} {'Has Number':<12} {'Length':<8}")
    print("  " + "-" * 70)
    for r in results:
        if 'error' not in r:
            echo_mark = "❌" if r['echo'] else "✅"
            refusal_mark = "❌" if r['ai_refusal'] else "✅"
            number_mark = "✅" if r['has_number'] else "❌"
            print(f"  {r['config']:<25} {echo_mark:<8} {refusal_mark:<12} {number_mark:<12} {r['length']:<8}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print("\n💡 Best config:")
print("   - No echo (✅)")
print("   - No AI refusal (✅)")
print("   - Has number (✅)")
print("=" * 80)
