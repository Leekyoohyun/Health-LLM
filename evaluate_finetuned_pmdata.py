#!/usr/bin/env python3
"""
Fine-tuned 평가 (MedAlpaca-7b + LoRA)
PMData 4개 task, 10% validation set, Sampling decoding
"""

import json
import torch
from tqdm import tqdm
from medalpaca.inferer import Inferer
from datasets import load_dataset
import os

VAL_SET_SIZE = 0.1
SEED = 42
OUTPUT_FILE = "finetuned_pmdata_results.json"

TASKS = [
    {"name": "PMData_stress", "file": "evaluation-json-data/PMData_stress_zeroshot.json"},
    {"name": "PMData_readiness", "file": "evaluation-json-data/PMData_readiness_zeroshot.json"},
    {"name": "PMData_sleep_quality", "file": "evaluation-json-data/PMData_sleep_quality_zeroshot.json"},
    {"name": "PMData_fatigue", "file": "evaluation-json-data/PMData_fatigue_zeroshot.json"},
]

print("=" * 80)
print("FINE-TUNED EVALUATION (MedAlpaca-7b + LoRA)")
print("=" * 80)
print(f"Validation set: 10% (seed={SEED})")
print(f"Sampling: temperature=0.7, do_sample=True")
print("=" * 80)

# Load fine-tuned model (WITH LoRA)
print("\nLoading fine-tuned model...")
inferer = Inferer(
    model_name="outputs/healthalpaca-7b-lora",
    base_model="medalpaca/medalpaca-7b",
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,  # 입력 길이 제한 해제
    torch_dtype=torch.float16,
    peft=True  # WITH LoRA
)
print("✓ Fine-tuned model loaded (MedAlpaca-7b + LoRA)\n")

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

        # Train/Val split (train.py와 동일)
        split_data = dataset["train"].train_test_split(
            test_size=VAL_SET_SIZE,
            shuffle=True,
            seed=SEED
        )
        val_data = split_data["test"]

        print(f"Total samples: {len(dataset['train'])}")
        print(f"Validation samples: {len(val_data)}")

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
                    temperature=0.7,      # Sampling temperature
                    top_k=50,             # Top-K sampling
                    top_p=0.9,            # Nucleus sampling
                    repetition_penalty=1.1,
                    do_sample=True,       # Enable sampling (diversity!)
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

# Save results
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total evaluated: {total_samples}")
print(f"Total errors: {total_errors}")
print(f"Success rate: {(total_samples - total_errors) / total_samples * 100:.2f}%")
print(f"\n✅ Results saved to: {OUTPUT_FILE}")
print("\nNext: python calculate_baseline_mae.py --input finetuned_pmdata_results.json")
print("=" * 80)
