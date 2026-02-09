#!/usr/bin/env python3
"""
전체 데이터셋의 토큰 길이 분석
- 8개 task 모든 샘플의 토큰 길이 측정
- 최대값, 평균, 95/99 percentile 계산
- 1024로 충분한지 확인
"""

import json
import numpy as np
from transformers import AutoTokenizer
from pathlib import Path

# ========================================
# Tokenizer 로드
# ========================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-7b")
print("✓ Tokenizer loaded\n")

# ========================================
# Prompt Template 로드
# ========================================
with open("medalpaca/prompt_templates/medalpaca.json") as f:
    prompt_template = json.load(f)

def format_prompt(instruction, input_text):
    """MedAlpaca prompt template 적용"""
    prompt = prompt_template["primer"]
    prompt += prompt_template["instruction"] + instruction
    prompt += prompt_template["input"] + input_text
    prompt += prompt_template["output"]
    return prompt

# ========================================
# 8개 Task 데이터 파일
# ========================================
data_files = [
    "evaluation-json-data/PMData_stress_train_all.json",
    "evaluation-json-data/PMData_readiness_train_all.json",
    "evaluation-json-data/PMData_sleep_quality_train_all.json",
    "evaluation-json-data/PMData_fatigue_train_all.json",
    "evaluation-json-data/LifeSnaps_stress_resilience_train_all.json",
    "evaluation-json-data/LifeSnaps_sleep_disorder_train_all.json",
    "evaluation-json-data/AW_FB_activity_train_all.json",
    "evaluation-json-data/AW_FB_calories_train_all.json",
]

# ========================================
# 각 Task별 토큰 길이 분석
# ========================================
print("=" * 80)
print("Analyzing token lengths for all tasks...")
print("=" * 80)
print()

all_token_lengths = []
task_stats = []

for data_file in data_files:
    task_name = Path(data_file).stem.replace("_train_all", "")

    print(f"[{task_name}]")

    # 데이터 로드
    with open(data_file) as f:
        data = json.load(f)

    # 토큰 길이 측정
    token_lengths = []
    for sample in data:
        instruction = sample['instruction']
        input_text = sample['input']

        # Prompt 생성
        question = format_prompt(instruction, input_text)

        # 토큰화
        tokens = tokenizer(question, return_tensors="pt")
        token_count = tokens.input_ids.shape[1]
        token_lengths.append(token_count)

    all_token_lengths.extend(token_lengths)

    # 통계 계산
    token_lengths = np.array(token_lengths)
    stats = {
        "task": task_name,
        "num_samples": len(token_lengths),
        "min": int(token_lengths.min()),
        "max": int(token_lengths.max()),
        "mean": float(token_lengths.mean()),
        "median": float(np.median(token_lengths)),
        "p95": float(np.percentile(token_lengths, 95)),
        "p99": float(np.percentile(token_lengths, 99)),
    }
    task_stats.append(stats)

    # 출력
    print(f"  Samples: {stats['num_samples']}")
    print(f"  Min: {stats['min']} tokens")
    print(f"  Max: {stats['max']} tokens")
    print(f"  Mean: {stats['mean']:.1f} tokens")
    print(f"  Median: {stats['median']:.1f} tokens")
    print(f"  95th percentile: {stats['p95']:.1f} tokens")
    print(f"  99th percentile: {stats['p99']:.1f} tokens")
    print()

# ========================================
# 전체 통계
# ========================================
print("=" * 80)
print("OVERALL STATISTICS (All 8 tasks combined)")
print("=" * 80)
print()

all_token_lengths = np.array(all_token_lengths)

print(f"Total samples: {len(all_token_lengths)}")
print(f"Min: {all_token_lengths.min()} tokens")
print(f"Max: {all_token_lengths.max()} tokens")
print(f"Mean: {all_token_lengths.mean():.1f} tokens")
print(f"Median: {np.median(all_token_lengths):.1f} tokens")
print(f"95th percentile: {np.percentile(all_token_lengths, 95):.1f} tokens")
print(f"99th percentile: {np.percentile(all_token_lengths, 99):.1f} tokens")
print()

# ========================================
# max_length 권장사항
# ========================================
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()

max_tokens = all_token_lengths.max()
p99_tokens = np.percentile(all_token_lengths, 99)

if max_tokens <= 256:
    print("✓ max_length=256 is sufficient for ALL samples")
elif max_tokens <= 512:
    print("✓ max_length=512 is sufficient for ALL samples")
    print(f"  (Current max: {max_tokens} tokens)")
elif max_tokens <= 1024:
    print("✓ max_length=1024 is sufficient for ALL samples")
    print(f"  (Current max: {max_tokens} tokens)")
elif max_tokens <= 2048:
    print("⚠️  max_length=2048 is needed for ALL samples")
    print(f"  (Current max: {max_tokens} tokens)")
else:
    print("❌ max_length > 2048 is needed!")
    print(f"  (Current max: {max_tokens} tokens)")
    print("  WARNING: LLaMA was trained with max_length=2048")

print()
print(f"If you want to cover 99% of samples:")
if p99_tokens <= 256:
    print("  → Use max_length=256")
elif p99_tokens <= 512:
    print("  → Use max_length=512")
elif p99_tokens <= 1024:
    print("  → Use max_length=1024")
else:
    print(f"  → Use max_length=2048 (99th percentile: {p99_tokens:.0f} tokens)")

print()
print("=" * 80)

# ========================================
# JSON 저장
# ========================================
output = {
    "overall": {
        "total_samples": int(len(all_token_lengths)),
        "min": int(all_token_lengths.min()),
        "max": int(all_token_lengths.max()),
        "mean": float(all_token_lengths.mean()),
        "median": float(np.median(all_token_lengths)),
        "p95": float(np.percentile(all_token_lengths, 95)),
        "p99": float(np.percentile(all_token_lengths, 99)),
    },
    "tasks": task_stats
}

with open("token_length_analysis.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n✓ Detailed statistics saved to: token_length_analysis.json")
