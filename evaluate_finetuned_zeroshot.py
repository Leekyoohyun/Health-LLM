#!/usr/bin/env python3
"""
Fine-tuned 평가 (Zero-shot 형식 + Sampling Decoding)
유저 정보 포함 데이터셋 사용
"""

import json
import torch
from tqdm import tqdm
from medalpaca.inferer import Inferer
from datasets import load_dataset
import os

VAL_SET_SIZE = 0.1
OUTPUT_FILE = "finetuned_zeroshot_results.json"

TASKS = [
    {"name": "PMData_stress", "file": "evaluation-json-data/PMData_stress_zeroshot.json"},
    {"name": "PMData_readiness", "file": "evaluation-json-data/PMData_readiness_zeroshot.json"},
    {"name": "PMData_sleep_quality", "file": "evaluation-json-data/PMData_sleep_quality_zeroshot.json"},
    {"name": "PMData_fatigue", "file": "evaluation-json-data/PMData_fatigue_zeroshot.json"},
]

print("=" * 80)
print("FINE-TUNED EVALUATION (ZERO-SHOT FORMAT + SAMPLING DECODING)")
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
print("✓ Fine-tuned model loaded (LoRA)\n")

all_results = []
total_samples = 0
total_errors = 0

for task_idx, task in enumerate(TASKS):
    task_name = task['name']

    print(f"\n{'=' * 80}")
    print(f"[Task {task_idx+1}/4] {task_name}")
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

        for idx in tqdm(range(len(val_data)), desc=f"{task_name}"):
            sample = val_data[idx]
            instruction = sample['instruction']
            input_text = sample['input']
            ground_truth = sample['output']

            try:
                answer = inferer(
                    instruction=instruction,
                    input=input_text,
                    max_new_tokens=128,
                    temperature=0.7,      # Sampling temperature (논문)
                    top_k=50,             # Top-K sampling
                    top_p=0.9,            # Nucleus sampling
                    repetition_penalty=1.1,
                    do_sample=True,       # Enable sampling! (diversity)
                    verbose=False
                )

                result = {
                    'task': task_name,
                    'val_idx': idx,
                    'ground_truth': ground_truth,
                    'predicted': answer,
                    'status': 'SUCCESS'
                }
                all_results.append(result)

            except Exception as e:
                total_errors += 1
                result = {
                    'task': task_name,
                    'val_idx': idx,
                    'ground_truth': ground_truth,
                    'status': 'ERROR',
                    'error': str(e)[:150]
                }
                all_results.append(result)

        total_samples += len(val_data)
        print(f"✅ Evaluated: {len(val_data)}")

    except Exception as e:
        print(f"❌ TASK ERROR: {str(e)[:150]}")

# Save
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total evaluated: {total_samples}")
print(f"Total errors: {total_errors}")
print(f"Success rate: {(total_samples - total_errors) / total_samples * 100:.2f}%")
print(f"\n✅ Results saved to: {OUTPUT_FILE}")
print("\nNext: python calculate_baseline_mae.py --input finetuned_zeroshot_results.json")
print("=" * 80)
