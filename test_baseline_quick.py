#!/usr/bin/env python3
"""
Baseline (MedAlpaca-7b) 빠른 테스트
LoRA 없이 원래 프롬프트만 사용
Fine-tuned와 비교용
"""

import json
import torch
from medalpaca.inferer import Inferer
from datasets import load_dataset

# ========================================
# 8개 Task 설정
# ========================================
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

# ========================================
# Inferer 로드 (Baseline only)
# ========================================
print("=" * 80)
print("Loading Baseline MedAlpaca-7b (no LoRA)...")
print("=" * 80)

inferer = Inferer(
    model_name="medalpaca/medalpaca-7b",
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,
    torch_dtype=torch.float16,
)

print("✓ Baseline model loaded (no LoRA)")

# ========================================
# 🔥 CRITICAL FIX: Tokenizer max_length
# ========================================
if inferer.data_handler.tokenizer.model_max_length < 2048:
    print(f"⚠️  Fixing tokenizer max_length: {inferer.data_handler.tokenizer.model_max_length} → 2048")
    inferer.data_handler.tokenizer.model_max_length = 2048
    print(f"✓ Tokenizer max_length fixed to 2048\n")
else:
    print(f"✓ Tokenizer max_length: {inferer.data_handler.tokenizer.model_max_length}\n")

# ========================================
# Quick Test: 원래 프롬프트 사용
# ========================================
print("=" * 80)
print("QUICK TEST: Baseline Model (ORIGINAL PROMPT)")
print("=" * 80)

for task_idx, task in enumerate(TASKS):
    task_name = task['name']

    print(f"\n{'=' * 80}")
    print(f"[Task {task_idx+1}/8] {task_name}")
    print("=" * 80)

    try:
        # 데이터 로드
        dataset = load_dataset("json", data_files=task['file'])

        # validation set split (seed=42로 동일하게)
        split_data = dataset["train"].train_test_split(
            test_size=0.1,
            shuffle=True,
            seed=42
        )

        val_data = split_data["test"]

        # 첫 번째 샘플만 테스트
        sample = val_data[0]
        instruction = sample['instruction']  # 원래 instruction 그대로
        input_text = sample['input']
        ground_truth = sample['output']

        print(f"\n📋 Instruction (first 200 chars):")
        print(f"   {instruction[:200]}...")

        print(f"\n📊 Input ({len(input_text)} chars):")
        print(f"   {input_text[:400]}...")

        print(f"\n🎯 Ground Truth:")
        print(f"   {ground_truth}")

        # Inference (원래 프롬프트만 사용)
        print(f"\n⏳ Generating...")
        answer = inferer(
            instruction=instruction,  # 개선 없이 원래대로
            input=input_text,
            max_new_tokens=256,
            repetition_penalty=1.1,
            verbose=False
        )

        print(f"\n💬 Predicted Output:")
        print(f"   {answer}")

        # 간단한 분석
        is_numeric = False
        try:
            float(answer.strip().rstrip('</s>').strip())
            is_numeric = True
        except:
            pass

        is_short = len(answer) < 50
        is_echo = answer.startswith(input_text[:50])
        has_eos = '</s>' in answer
        is_ai_refusal = 'AI' in answer or "I'm sorry" in answer or "I don't" in answer

        print(f"\n📈 Analysis:")
        print(f"   - Is numeric: {is_numeric}")
        print(f"   - Is short (<50 chars): {is_short}")
        print(f"   - Is echo: {is_echo}")
        print(f"   - Has EOS token: {has_eos}")
        print(f"   - AI refusal: {is_ai_refusal}")
        print(f"   - Length: {len(answer)} chars")

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {str(e)[:200]}")

print("\n" + "=" * 80)
print("QUICK TEST COMPLETE")
print("=" * 80)
print("\n📊 Compare with:")
print("   python test_finetuned_original_prompt.py")
print("=" * 80)
