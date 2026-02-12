#!/usr/bin/env python3
"""
Random Baseline 생성 (MedAlpaca baseline이 작동하지 않으므로)
각 task의 범위에서 random 예측
"""

import json
import random
from datasets import load_dataset
from tqdm import tqdm

VAL_SET_SIZE = 0.1

# Task별 예측 범위
TASK_CONFIG = {
    "PMData_stress": {"type": "regression", "range": (1, 5), "format": "The predicted stress level is {}."},
    "PMData_readiness": {"type": "regression", "range": (0, 10), "format": "The predicted readiness level is {}."},
    "PMData_sleep_quality": {"type": "regression", "range": (1, 5), "format": "The predicted sleep_quality level is {}."},
    "PMData_fatigue": {"type": "classification", "classes": [1, 2, 3, 4, 5], "format": "The predicted fatigue level is {}."},
    "LifeSnaps_stress_resilience": {"type": "regression", "range": (0.2, 5.0), "format": "{:.2f}"},
    "LifeSnaps_sleep_disorder": {"type": "classification", "classes": [0, 1], "format": "{}"},
    "AW_FB_activity": {"type": "classification", "classes": ['Self Pace Walk', 'Sitting', 'Lying', 'Running 7 METs', 'Running 5 METs', 'Running 3 METs'], "format": "The predicted activity type is {}"},
    "AW_FB_calories": {"type": "regression", "range": (0, 100), "format": "The predicted calorie burn is {:.2f}."},
}

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

random.seed(42)  # Reproducible

all_results = []

print("=" * 80)
print("RANDOM BASELINE GENERATION")
print("=" * 80)

for task_idx, task in enumerate(TASKS):
    task_name = task['name']
    config = TASK_CONFIG[task_name]

    print(f"\n[Task {task_idx+1}/8] {task_name}")

    # Load validation set
    dataset = load_dataset("json", data_files=task['file'])
    split_data = dataset["train"].train_test_split(
        test_size=VAL_SET_SIZE,
        shuffle=True,
        seed=42
    )
    val_data = split_data["test"]

    print(f"  Validation samples: {len(val_data)}")

    for idx in tqdm(range(len(val_data)), desc=f"  {task_name}"):
        sample = val_data[idx]
        ground_truth = sample['output']

        # Generate random prediction
        if config['type'] == 'regression':
            min_val, max_val = config['range']
            if isinstance(min_val, float):
                pred_value = random.uniform(min_val, max_val)
            else:
                pred_value = random.randint(min_val, max_val)
            prediction = config['format'].format(pred_value)
        else:  # classification
            pred_value = random.choice(config['classes'])
            prediction = config['format'].format(pred_value)

        result = {
            'task': task_name,
            'val_idx': idx,
            'ground_truth': ground_truth,
            'predicted': prediction,
            'status': 'SUCCESS'
        }
        all_results.append(result)

    print(f"  ✅ {len(val_data)} predictions generated")

# Save
output_file = "random_baseline_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved to: {output_file}")
print(f"Total predictions: {len(all_results)}")
print("\nNext step:")
print(f"  python calculate_baseline_mae.py --input {output_file}")
