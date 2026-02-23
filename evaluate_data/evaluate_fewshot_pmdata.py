#!/usr/bin/env python3
"""
Few-shot 평가 (MedAlpaca-7b, LoRA 없음)
PMData 4개 task, 10% validation set, Sampling decoding

Few-shot 방식:
- Train set (90%)에서 3개 예시를 랜덤 추출
- 프롬프트에 "Here are some examples: ..." 형식으로 포함
- Validation set (10%)으로 평가
"""

import json
import torch
import random
import numpy as np
from tqdm import tqdm
from medalpaca.inferer import Inferer
from datasets import load_dataset

VAL_SET_SIZE = 0.1
SEED = 42
N_SHOTS = 3         # few-shot 예시 개수
OUTPUT_FILE = "fewshot_pmdata_results.json"

# few-shot 예시로 쓸 input을 적절한 길이로 자르기 (토큰 초과 방지)
EXAMPLE_INPUT_MAX_CHARS = 600

TASKS = [
    {"name": "PMData_stress",        "file": "evaluation-json-data/PMData_stress_zeroshot.json"},
    {"name": "PMData_readiness",     "file": "evaluation-json-data/PMData_readiness_zeroshot.json"},
    {"name": "PMData_sleep_quality", "file": "evaluation-json-data/PMData_sleep_quality_zeroshot.json"},
    {"name": "PMData_fatigue",       "file": "evaluation-json-data/PMData_fatigue_zeroshot.json"},
]

print("=" * 80)
print("FEW-SHOT EVALUATION (MedAlpaca-7b, NO LoRA, {}-shot)".format(N_SHOTS))
print("=" * 80)
print(f"Validation set: 10% (seed={SEED})")
print(f"Few-shot examples: {N_SHOTS} from train set")
print(f"Sampling: temperature=0.7, do_sample=True")
print("=" * 80)

# Load baseline model (NO LoRA)
print("\nLoading baseline model...")
inferer = Inferer(
    model_name="medalpaca/medalpaca-7b",
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,
    torch_dtype=torch.float16,
    peft=False  # NO LoRA
)
print("✓ Baseline model loaded (MedAlpaca-7b)\n")

rng = random.Random(SEED)


def build_fewshot_input(examples: list, actual_input: str) -> str:
    """
    Train set 예시들을 few-shot 프롬프트로 구성.
    예시 input은 너무 길면 앞부분만 사용.
    """
    parts = []
    for i, ex in enumerate(examples, 1):
        ex_input = ex['input']
        if len(ex_input) > EXAMPLE_INPUT_MAX_CHARS:
            ex_input = ex_input[:EXAMPLE_INPUT_MAX_CHARS] + "..."
        parts.append(f"[Example {i}]\n{ex_input}\nAnswer: {ex['output']}")

    few_shot_block = "\n\n".join(parts)
    return (
        "Here are some example predictions:\n\n"
        + few_shot_block
        + "\n\nNow predict for the following case:\n"
        + actual_input
    )


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

        # Train / Val split (train.py + baseline와 동일한 split)
        split_data = dataset["train"].train_test_split(
            test_size=VAL_SET_SIZE,
            shuffle=True,
            seed=SEED
        )
        train_data = split_data["train"]
        val_data   = split_data["test"]

        print(f"Total samples: {len(dataset['train'])}")
        print(f"Train samples: {len(train_data)}  |  Validation samples: {len(val_data)}")

        # Train set에서 few-shot 예시 후보 수집
        train_list = [train_data[i] for i in range(len(train_data))]

        for idx in tqdm(range(len(val_data)), desc=f"{task_name}"):
            sample = val_data[idx]
            instruction   = sample['instruction']
            actual_input  = sample['input']
            ground_truth  = sample['output']

            # val 샘플과 다른 participant의 예시를 우선 선택 (없으면 무관하게 선택)
            val_pid = sample.get('participant_id', '')
            diff_pid = [s for s in train_list if s.get('participant_id', '') != val_pid]
            pool = diff_pid if len(diff_pid) >= N_SHOTS else train_list

            shot_examples = rng.sample(pool, N_SHOTS)

            fewshot_input = build_fewshot_input(shot_examples, actual_input)

            try:
                answer = inferer(
                    instruction=instruction,
                    input=fewshot_input,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                    verbose=False
                )

                result = {
                    'task': task_name,
                    'val_idx': idx,
                    'ground_truth': ground_truth,
                    'predicted': answer,
                    'status': 'SUCCESS',
                    'n_shots': N_SHOTS,
                }
                all_results.append(result)

            except Exception as e:
                total_errors += 1
                result = {
                    'task': task_name,
                    'val_idx': idx,
                    'ground_truth': ground_truth,
                    'status': 'ERROR',
                    'error': str(e)[:150],
                    'n_shots': N_SHOTS,
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
if total_samples > 0:
    print(f"Success rate: {(total_samples - total_errors) / total_samples * 100:.2f}%")
print(f"\n✅ Results saved to: {OUTPUT_FILE}")
print("\nNext: python calculate_baseline_mae.py --input fewshot_pmdata_results.json --mode fewshot")
print("=" * 80)
