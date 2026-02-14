#!/usr/bin/env python3
"""
[TEST] Fine-tuned 평가 (샘플 3개씩만)
"""

import json
import torch
from tqdm import tqdm
from medalpaca.inferer import Inferer
from datasets import load_dataset

VAL_SET_SIZE = 0.1
SEED = 42
TEST_SAMPLES = 3  # 각 task당 3개만
OUTPUT_FILE = "finetuned_pmdata_test_results.json"

TASKS = [
    {"name": "PMData_stress", "file": "evaluation-json-data/PMData_stress_zeroshot.json"},
    {"name": "PMData_readiness", "file": "evaluation-json-data/PMData_readiness_zeroshot.json"},
    {"name": "PMData_sleep_quality", "file": "evaluation-json-data/PMData_sleep_quality_zeroshot.json"},
    {"name": "PMData_fatigue", "file": "evaluation-json-data/PMData_fatigue_zeroshot.json"},
]

print("=" * 80)
print("🧪 TEST MODE - Fine-tuned (3 samples per task)")
print("=" * 80)

print("\nLoading fine-tuned model...")
inferer = Inferer(
    model_name="outputs/healthalpaca-7b-lora",
    base_model="medalpaca/medalpaca-7b",
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,
    torch_dtype=torch.float16,
    peft=True
)
print("✓ Fine-tuned loaded\n")

all_results = []
total_samples = 0

for task_idx, task in enumerate(TASKS):
    task_name = task['name']
    print(f"[{task_idx+1}/4] {task_name} - Testing {TEST_SAMPLES} samples...")

    dataset = load_dataset("json", data_files=task['file'])
    split_data = dataset["train"].train_test_split(test_size=VAL_SET_SIZE, shuffle=True, seed=SEED)
    val_data = split_data["test"]

    # Only take first 3 samples
    num_samples = min(TEST_SAMPLES, len(val_data))

    for idx in tqdm(range(num_samples), desc=task_name):
        sample = val_data[idx]

        try:
            answer = inferer(
                instruction=sample['instruction'],
                input=sample['input'],
                max_new_tokens=128,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                verbose=False
            )

            all_results.append({
                'task': task_name,
                'val_idx': idx,
                'ground_truth': sample['output'],
                'predicted': answer,
                'status': 'SUCCESS'
            })

        except Exception as e:
            all_results.append({
                'task': task_name,
                'val_idx': idx,
                'ground_truth': sample['output'],
                'status': 'ERROR',
                'error': str(e)[:150]
            })

    total_samples += num_samples
    print(f"  ✓ {num_samples} done\n")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print("=" * 80)
print(f"✅ TEST COMPLETE - {total_samples} samples")
print(f"Results: {OUTPUT_FILE}")
print("=" * 80)
