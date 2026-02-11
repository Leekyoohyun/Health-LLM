#!/usr/bin/env python3
"""
세 PMData task의 입력 데이터 비교
왜 sleep_quality는 echo가 적고 stress/readiness는 많은가?
"""

import json
from datasets import load_dataset

TASKS = [
    {"name": "PMData_stress", "file": "evaluation-json-data/PMData_stress_train_all.json"},
    {"name": "PMData_readiness", "file": "evaluation-json-data/PMData_readiness_train_all.json"},
    {"name": "PMData_sleep_quality", "file": "evaluation-json-data/PMData_sleep_quality_train_all.json"},
]

VAL_SET_SIZE = 0.1

for task in TASKS:
    print(f"\n{'=' * 80}")
    print(f"{task['name']}")
    print("=" * 80)

    # 데이터 로드 (평가와 동일한 방식)
    dataset = load_dataset("json", data_files=task['file'])
    split_data = dataset["train"].train_test_split(
        test_size=VAL_SET_SIZE,
        shuffle=True,
        seed=42
    )
    val_data = split_data["test"]

    # 첫 번째 샘플 분석
    sample = val_data[0]
    instruction = sample['instruction']
    input_text = sample['input']
    output = sample['output']

    print(f"\n📋 Instruction:")
    print(f"  {instruction[:200]}...")
    print(f"  Length: {len(instruction)} chars")

    print(f"\n📊 Input:")
    print(f"  {input_text[:300]}...")
    print(f"  Length: {len(input_text)} chars")
    print(f"  Starts with: {input_text[:50]}")

    print(f"\n🎯 Output:")
    print(f"  {output}")
    print(f"  Length: {len(output)} chars")

    # 센서 종류 확인
    sensors = []
    if "[Steps]" in input_text:
        sensors.append("Steps")
    if "[Calories" in input_text or "[Burned" in input_text:
        sensors.append("Calories")
    if "[Heart Rate]" in input_text or "[Resting Heart Rate]" in input_text:
        sensors.append("Heart Rate")
    if "[Sleep]" in input_text:
        sensors.append("Sleep")

    print(f"\n🔍 Sensors used: {', '.join(sensors)}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\n💡 Key questions:")
print("   1. Is sleep_quality input shorter?")
print("   2. Does sleep_quality use different sensors?")
print("   3. Is the instruction different?")
print("=" * 80)
