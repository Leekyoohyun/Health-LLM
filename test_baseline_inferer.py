#!/usr/bin/env python3
"""
Baseline (MedAlpaca-7b) 전체 태스크 샘플 테스트 - Inferer 사용
논문 재현을 위해 Inferer 클래스 사용 (원본 저자 방식)
"""

import json
import torch
from medalpaca.inferer import Inferer

# ========================================
# 8개 Task 설정
# ========================================
TASKS = [
    {"name": "PMData_stress", "file": "evaluation-json-data/PMData_stress_train_all.json"},
    {"name": "PMData_readiness", "file": "evaluation-json-data/PMData_readiness_train_all.json"},
    {"name": "PMData_sleep_quality", "file": "evaluation-json-data/PMData_sleep_quality_train_all.json"},
    {"name": "PMData_fatigue", "file": "evaluation-json-data/PMData_fatigue_train_all.json"},
    {"name": "LifeSnaps_stress_resilience", "file": "evaluation-json-data/LifeSnaps_stress_resilience_train_all.json"},
    {"name": "LifeSnaps_sleep_disorder", "file": "evaluation-json-data/LifeSnaps_sleep_disorder_train_all.json"},
    {"name": "AW_FB_activity", "file": "evaluation-json-data/AW_FB_activity_train_all.json"},
    {"name": "AW_FB_calories", "file": "evaluation-json-data/AW_FB_calories_train_all.json"},
]

# ========================================
# Inferer 로드 (논문 방식)
# ========================================
print("=" * 80)
print("Loading MedAlpaca-7b with Inferer (논문 재현)...")
print("=" * 80)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

inferer = Inferer(
    model_name="medalpaca/medalpaca-7b",
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,  # 논문과 동일
    torch_dtype=torch.float16,
)

print("✓ Model loaded\n")

# ========================================
# 전체 Task 테스트 (각 1개 샘플)
# ========================================
print("=" * 80)
print("BASELINE EVALUATION - Inferer (논문 방식)")
print("=" * 80)

all_results = []

for task_idx, task in enumerate(TASKS):
    print(f"\n{'=' * 80}")
    print(f"[Task {task_idx+1}/8] {task['name']}")
    print("=" * 80)

    try:
        # 데이터 로드
        with open(task['file']) as f:
            data = json.load(f)

        print(f"Total samples: {len(data)}")

        # 첫 번째 샘플
        sample = data[0]
        instruction = sample['instruction']
        input_text = sample['input']
        ground_truth = sample['output']

        print(f"\nInstruction: {instruction[:100]}...")
        print(f"Input (first 150 chars): {input_text[:150]}...")
        print(f"Ground truth: {ground_truth}")

        # Inferer로 추론 (논문 방식)
        # max_new_tokens는 기본값 128 사용
        answer = inferer(
            instruction=instruction,
            input=input_text,
            max_new_tokens=128,  # 기본값
            verbose=False  # True로 하면 프롬프트 출력
        )

        print(f"\n✅ Generated answer: {answer}")

        # 결과 저장
        all_results.append({
            'task': task['name'],
            'ground_truth': ground_truth,
            'predicted': answer,
            'status': 'SUCCESS'
        })

    except FileNotFoundError:
        print(f"⚠️  File not found: {task['file']}")
        all_results.append({
            'task': task['name'],
            'status': 'FILE_NOT_FOUND'
        })
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)[:150]}")
        all_results.append({
            'task': task['name'],
            'status': 'ERROR',
            'error': str(e)[:150]
        })

# ========================================
# 결과 요약
# ========================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

for result in all_results:
    if result['status'] == 'SUCCESS':
        print(f"\n[{result['task']}]")
        print(f"  Ground Truth: {result['ground_truth']}")
        print(f"  Predicted:    {result['predicted']}")
    else:
        print(f"\n[{result['task']}] ❌ {result['status']}")

# 결과 저장
output_file = "baseline_inferer_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Results saved to: {output_file}")
print("=" * 80)
