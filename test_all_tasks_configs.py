#!/usr/bin/env python3
"""
여러 task에서 generation config 테스트
PMData (stress, readiness, sleep_quality) + LifeSnaps
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

# ========================================
# 테스트할 설정들
# ========================================
configs = [
    {"name": "기본값", "params": {}},
    {"name": "penalty=1.1", "params": {"repetition_penalty": 1.1}},
    {"name": "penalty=1.2", "params": {"repetition_penalty": 1.2}},
    {"name": "penalty=1.5", "params": {"repetition_penalty": 1.5}},
]

print("=" * 80)
print("TEST: Multiple Tasks (2 samples each)")
print("=" * 80)

import re

# 전체 결과 저장
all_task_results = {}

for task in TASKS:
    print(f"\n{'=' * 80}")
    print(f"TASK: {task['name']}")
    print("=" * 80)

    # 데이터 로드
    with open(task['file']) as f:
        data = json.load(f)

    task_summary = {config['name']: {'success': 0, 'total': 0} for config in configs}

    # 처음 2개 샘플 테스트
    for sample_idx in range(min(2, len(data))):
        sample = data[sample_idx]
        instruction = sample['instruction']
        input_text = sample['input']
        ground_truth = sample['output']

        print(f"\n[Sample {sample_idx}]")
        print(f"  GT: {ground_truth}")

        for config in configs:
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
                numbers = re.findall(r'\b\d+\b', answer)
                has_number = len(numbers) > 0

                # 성공 조건: Echo 없고, AI refusal 없고, 숫자 포함
                is_success = not is_echo and not is_ai_refusal and has_number

                task_summary[config['name']]['total'] += 1
                if is_success:
                    task_summary[config['name']]['success'] += 1

                status = "✅" if is_success else "❌"
                print(f"  {config['name']:<15} {status} (Echo:{is_echo}, Refusal:{is_ai_refusal}, Number:{has_number})")

            except Exception as e:
                print(f"  {config['name']:<15} ❌ ERROR: {e}")
                task_summary[config['name']]['total'] += 1

    # Task 요약
    print(f"\n📊 Summary for {task['name']}:")
    for config_name, stats in task_summary.items():
        success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {config_name:<15} {stats['success']}/{stats['total']} ({success_rate:.0f}%)")

    all_task_results[task['name']] = task_summary

# ========================================
# 전체 요약
# ========================================
print("\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)

for config in configs:
    config_name = config['name']
    print(f"\n[{config_name}]")

    total_success = 0
    total_samples = 0

    for task_name, task_summary in all_task_results.items():
        stats = task_summary[config_name]
        total_success += stats['success']
        total_samples += stats['total']
        success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {task_name:<30} {stats['success']}/{stats['total']} ({success_rate:.0f}%)")

    overall_rate = (total_success / total_samples * 100) if total_samples > 0 else 0
    print(f"  {'TOTAL':<30} {total_success}/{total_samples} ({overall_rate:.0f}%)")

print("\n" + "=" * 80)
print("💡 Best config: Highest overall success rate")
print("=" * 80)
