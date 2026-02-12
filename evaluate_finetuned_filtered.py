#!/usr/bin/env python3
"""
Fine-tuned 평가 (긴 input 필터링 버전)
Training data 범위 내의 샘플만 평가
"""

import json
import torch
from tqdm import tqdm
from medalpaca.inferer import Inferer
from datasets import load_dataset
import os

VAL_SET_SIZE = 0.1
OUTPUT_FILE = "finetuned_filtered_results.json"
MAX_INPUT_LENGTH = 1200  # Training data max = 1249

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

print("=" * 80)
print("FINE-TUNED EVALUATION (FILTERED)")
print(f"Max input length: {MAX_INPUT_LENGTH} chars (training data range)")
print("=" * 80)

# Load model
inferer = Inferer(
    model_name="outputs/healthalpaca-7b-lora",
    base_model="medalpaca/medalpaca-7b",
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,
    torch_dtype=torch.float16,
    peft=True
)
print("✓ Fine-tuned model loaded\n")

all_results = []
total_samples = 0
total_filtered = 0
total_errors = 0

for task_idx, task in enumerate(TASKS):
    task_name = task['name']

    print(f"\n{'=' * 80}")
    print(f"[Task {task_idx+1}/8] {task_name}")
    print("=" * 80)

    try:
        dataset = load_dataset("json", data_files=task['file'])
        split_data = dataset["train"].train_test_split(
            test_size=VAL_SET_SIZE,
            shuffle=True,
            seed=42
        )
        val_data = split_data["test"]

        print(f"Total validation samples: {len(val_data)}")

        task_results = []
        task_filtered = 0

        for idx in tqdm(range(len(val_data)), desc=f"{task_name}"):
            sample = val_data[idx]
            instruction = sample['instruction']
            input_text = sample['input']
            ground_truth = sample['output']

            # Filter by input length
            if len(input_text) > MAX_INPUT_LENGTH:
                task_filtered += 1
                total_filtered += 1
                continue

            try:
                answer = inferer(
                    instruction=instruction,
                    input=input_text,
                    max_new_tokens=128,
                    temperature=0.1,
                    repetition_penalty=1.1,
                    do_sample=False,
                    verbose=False
                )

                result = {
                    'task': task_name,
                    'val_idx': idx,
                    'input_length': len(input_text),
                    'ground_truth': ground_truth,
                    'predicted': answer,
                    'status': 'SUCCESS'
                }
                all_results.append(result)
                task_results.append(answer)

            except Exception as e:
                total_errors += 1
                result = {
                    'task': task_name,
                    'val_idx': idx,
                    'input_length': len(input_text),
                    'ground_truth': ground_truth,
                    'status': 'ERROR',
                    'error': str(e)[:150]
                }
                all_results.append(result)

        total_samples += (len(val_data) - task_filtered)

        print(f"\n✅ Evaluated: {len(val_data) - task_filtered}")
        print(f"🔍 Filtered (too long): {task_filtered}")

    except Exception as e:
        print(f"❌ TASK ERROR: {str(e)[:150]}")

# Save
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total evaluated: {total_samples}")
print(f"Total filtered: {total_filtered}")
print(f"Total errors: {total_errors}")
print(f"Success rate: {(total_samples - total_errors) / total_samples * 100:.2f}%")
print(f"\n✅ Results saved to: {OUTPUT_FILE}")
print("\nNext: python calculate_baseline_mae.py --input finetuned_filtered_results.json")
print("=" * 80)
