#!/usr/bin/env python3
"""
PMData 4개 JSON을 하나로 합쳐서 finetune_data_zeroshot.json 생성
"""

import json
import os

# Input files (서버 경로)
INPUT_DIR = "/home/khlee/fine-tuning/Health-LLM-datas/output"
OUTPUT_FILE = "/home/khlee/fine-tuning/Health-LLM-datas/output/finetune_data_zeroshot.json"

TASKS = [
    "PMData_stress_zeroshot.json",
    "PMData_readiness_zeroshot.json",
    "PMData_sleep_quality_zeroshot.json",
    "PMData_fatigue_zeroshot.json"
]

print("=" * 80)
print("MERGING PMData TASKS FOR FINE-TUNING")
print("=" * 80)

all_data = []

for task_file in TASKS:
    file_path = os.path.join(INPUT_DIR, task_file)

    print(f"\nReading: {task_file}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Remove participant_id field (training doesn't need it)
    for item in data:
        if 'participant_id' in item:
            del item['participant_id']

    print(f"  Samples: {len(data)}")
    all_data.extend(data)

print(f"\n{'=' * 80}")
print(f"TOTAL SAMPLES: {len(all_data)}")
print("=" * 80)

# Save
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ Merged data saved to:")
print(f"   {OUTPUT_FILE}")
print(f"\n📊 Sample breakdown:")
for task_file in TASKS:
    task_name = task_file.replace('_zeroshot.json', '')
    print(f"   - {task_name}")

print("\n" + "=" * 80)
print("READY FOR FINE-TUNING!")
print("=" * 80)
print("\nNext steps:")
print("1. Run training script:")
print("   cd ~/fine-tuning/Health-LLM")
print("   bash finetune_zeroshot.sh")
print("\n" + "=" * 80)
