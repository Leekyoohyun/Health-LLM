#!/usr/bin/env python3
"""
Baseline (MedAlpaca-7b) Validation Set 평가 - Inferer 사용
학습 시 남겨둔 10% validation set만 평가
각 task 끝날 때마다 저장 (중간 체크 가능)
"""

import json
import torch
from tqdm import tqdm
from medalpaca.inferer import Inferer
from datasets import load_dataset
import os

# ========================================
# 설정
# ========================================
VAL_SET_SIZE = 0.1  # 학습 시 설정과 동일
OUTPUT_FILE = "baseline_full_results.json"

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
# 기존 결과 로드 (있으면)
# ========================================
if os.path.exists(OUTPUT_FILE):
    print(f"⚠️  {OUTPUT_FILE} already exists!")
    print("Loading existing results...")
    with open(OUTPUT_FILE) as f:
        all_results = json.load(f)

    # 이미 완료된 task 확인
    completed_tasks = set(r['task'] for r in all_results)
    print(f"Already completed tasks: {completed_tasks}")
else:
    all_results = []
    completed_tasks = set()

# ========================================
# Inferer 로드
# ========================================
print("=" * 80)
print("Loading MedAlpaca-7b with Inferer...")
print("=" * 80)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

inferer = Inferer(
    model_name="medalpaca/medalpaca-7b",
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,
    torch_dtype=torch.float16,
)

print("✓ Model loaded\n")

# ========================================
# Validation Set 평가 (각 task별 10%)
# ========================================
print("=" * 80)
print("BASELINE VALIDATION SET EVALUATION")
print(f"Validation set size: {VAL_SET_SIZE * 100:.0f}%")
print("=" * 80)

total_samples = 0
total_errors = 0

for task_idx, task in enumerate(TASKS):
    task_name = task['name']

    # 이미 완료된 task는 스킵
    if task_name in completed_tasks:
        print(f"\n[Task {task_idx+1}/8] {task_name} - ALREADY COMPLETED ✅")
        continue

    print(f"\n{'=' * 80}")
    print(f"[Task {task_idx+1}/8] {task_name}")
    print("=" * 80)

    try:
        # 데이터 로드 (학습 시와 동일한 방식)
        dataset = load_dataset("json", data_files=task['file'])

        # train_test_split with same seed as training (seed=42, shuffle=True)
        split_data = dataset["train"].train_test_split(
            test_size=VAL_SET_SIZE,
            shuffle=True,
            seed=42
        )

        # Validation set (학습 시 "test"로 저장됨)
        val_data = split_data["test"]
        val_size = len(val_data)
        total_len = len(dataset["train"])

        print(f"Total samples: {total_len}")
        print(f"Validation set: {val_size} samples (shuffle=True, seed=42)")

        # Validation set 평가
        task_results = []
        task_errors = 0

        for idx in tqdm(range(val_size), desc=f"{task_name}", unit="sample"):
            sample = val_data[idx]
            instruction = sample['instruction']
            input_text = sample['input']
            ground_truth = sample['output']

            try:
                # Inferer로 추론 (원본 저자 방식 - 기본값 사용)
                answer = inferer(
                    instruction=instruction,
                    input=input_text,
                    max_new_tokens=256,  # 128 → 256 (더 긴 출력 허용)
                    # temperature, repetition_penalty 등: 기본값 사용 (논문 재현)
                    verbose=False
                )

                # 결과 저장
                result = {
                    'task': task_name,
                    'val_idx': idx,  # validation set 내 인덱스
                    'ground_truth': ground_truth,
                    'predicted': answer,
                    'status': 'SUCCESS'
                }
                all_results.append(result)
                task_results.append(answer)

            except Exception as e:
                task_errors += 1
                total_errors += 1
                result = {
                    'task': task_name,
                    'val_idx': idx,  # validation set 내 인덱스
                    'ground_truth': ground_truth,
                    'status': 'ERROR',
                    'error': str(e)[:150]
                }
                all_results.append(result)

                if task_errors <= 3:  # 처음 3개 에러만 출력
                    print(f"\n⚠️  Sample {idx} ERROR: {type(e).__name__}: {str(e)[:100]}")

        total_samples += val_size

        # Task 요약
        success_count = val_size - task_errors
        print(f"\n✅ Success: {success_count}/{val_size}")
        if task_errors > 0:
            print(f"❌ Errors: {task_errors}/{val_size}")

        # 샘플 출력 (처음 2개)
        print(f"\nSample predictions (first 2):")
        for i in range(min(2, len(task_results))):
            print(f"  [{i}] GT: {val_data[i]['output']}")
            print(f"      Pred: {task_results[i][:100]}{'...' if len(task_results[i]) > 100 else ''}")

        # ========================================
        # ✅ Task 끝날 때마다 저장!
        # ========================================
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Saved to {OUTPUT_FILE} ({len(all_results)} samples so far)")

    except FileNotFoundError:
        print(f"⚠️  File not found: {task['file']}")
        all_results.append({
            'task': task_name,
            'status': 'FILE_NOT_FOUND'
        })
    except Exception as e:
        print(f"❌ TASK ERROR: {type(e).__name__}: {str(e)[:150]}")
        all_results.append({
            'task': task_name,
            'status': 'TASK_ERROR',
            'error': str(e)[:150]
        })

# ========================================
# 최종 결과
# ========================================
print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print(f"Total samples evaluated: {total_samples}")
print(f"Total errors: {total_errors}")
if total_samples > 0:
    print(f"Success rate: {(total_samples - total_errors) / total_samples * 100:.2f}%")

print(f"\n✅ Results saved to: {OUTPUT_FILE}")
print("=" * 80)

# ========================================
# 다음 단계
# ========================================
print("\n🎯 Next steps:")
print("  1. Check MAE for completed tasks:")
print(f"     python calculate_baseline_mae.py --input {OUTPUT_FILE}")
print()
print("  2. Continue evaluation if interrupted:")
print("     python evaluate_baseline_full.py")
print("=" * 80)
