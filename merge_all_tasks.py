#!/usr/bin/env python3
"""
8개 task 데이터를 병합하여 finetune_data.json 생성
PMData (4개) + LifeSnaps (2개) + AW_FB (2개)
"""
import json
import os

# 8개 task 파일 (evaluation-json-data 디렉토리에 있음)
files = [
    "evaluation-json-data/PMData_stress_train_all.json",
    "evaluation-json-data/PMData_readiness_train_all.json",
    "evaluation-json-data/PMData_sleep_quality_train_all.json",
    "evaluation-json-data/PMData_fatigue_train_all.json",
    "evaluation-json-data/LifeSnaps_stress_resilience_train_all.json",
    "evaluation-json-data/LifeSnaps_sleep_disorder_train_all.json",
    "evaluation-json-data/AW_FB_activity_train_all.json",
    "evaluation-json-data/AW_FB_calories_train_all.json",
]

merged = []
total_samples = 0

print("=" * 80)
print("Merging 8 tasks into finetune_data.json")
print("=" * 80)

for f in files:
    if not os.path.exists(f):
        print(f"[WARNING] File not found: {f}")
        continue

    with open(f) as fp:
        data = json.load(fp)
        task_name = os.path.basename(f).replace("_train_all.json", "")
        num_samples = len(data)
        total_samples += num_samples

        print(f"[{task_name:35s}] {num_samples:6,d} samples")
        merged.extend(data)

print("-" * 80)
print(f"Total: {total_samples:,d} samples")
print()

# 저장
output_file = "Health-LLM/data/finetune_data.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w") as fp:
    json.dump(merged, fp, indent=2, ensure_ascii=False)

print(f"Saved to: {output_file}")
print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
print()

# 검증: instruction 유형 분포
print("=" * 80)
print("Validation: Instruction Distribution")
print("=" * 80)

instruction_counts = {}
for item in merged:
    inst = item.get("instruction", "")
    # 첫 50자만 key로 사용
    inst_key = inst[:50]
    instruction_counts[inst_key] = instruction_counts.get(inst_key, 0) + 1

for inst_key, count in sorted(instruction_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"{count:6,d} samples: {inst_key}...")

print()
print("✓ Merge completed successfully!")
