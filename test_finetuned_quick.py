#!/usr/bin/env python3
"""
Fine-tuned LoRA 모델 빠른 테스트
각 task에서 1개씩만 샘플링해서 출력 확인
전체 평가 전 sanity check 용도
"""

import json
import torch
from medalpaca.inferer import Inferer
from datasets import load_dataset
from peft import PeftModel

# ========================================
# 설정
# ========================================
LORA_ADAPTER_PATH = "outputs/healthalpaca-7b-lora"

# 프롬프트 개선: 숫자만 출력 강제
NUMERIC_OUTPUT_INSTRUCTION = """
IMPORTANT: Output ONLY a single number (e.g., "3" or "0.4").
Do not include any explanations, reasoning, or additional text.
""".strip()

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
# Inferer 로드 + LoRA Adapter
# ========================================
print("=" * 80)
print("Loading Fine-tuned MedAlpaca-7b (+ LoRA Adapter)...")
print("=" * 80)

inferer = Inferer(
    model_name="medalpaca/medalpaca-7b",
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,
    torch_dtype=torch.float16,
)

print("✓ Base model loaded")
print(f"Loading LoRA adapter from: {LORA_ADAPTER_PATH}")

inferer.model = PeftModel.from_pretrained(
    inferer.model,
    LORA_ADAPTER_PATH
)

print("✓ LoRA adapter loaded\n")

# ========================================
# Quick Test: 각 task에서 1개씩
# ========================================
print("=" * 80)
print("QUICK TEST: Fine-tuned Model (1 sample per task)")
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
        original_instruction = sample['instruction']
        input_text = sample['input']
        ground_truth = sample['output']

        # 프롬프트 개선
        improved_instruction = f"{original_instruction}\n\n{NUMERIC_OUTPUT_INSTRUCTION}"

        print(f"\n📋 Instruction (first 200 chars):")
        print(f"   {original_instruction[:200]}...")

        print(f"\n📊 Input ({len(input_text)} chars):")
        print(f"   {input_text[:400]}...")

        print(f"\n🎯 Ground Truth:")
        print(f"   {ground_truth}")

        # Inference
        print(f"\n⏳ Generating...")
        answer = inferer(
            instruction=improved_instruction,
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
            float(answer.strip())
            is_numeric = True
        except:
            pass

        is_short = len(answer) < 50
        is_echo = answer.startswith(input_text[:50])

        print(f"\n📈 Analysis:")
        print(f"   - Is numeric: {is_numeric}")
        print(f"   - Is short (<50 chars): {is_short}")
        print(f"   - Is echo: {is_echo}")
        print(f"   - Length: {len(answer)} chars")

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {str(e)[:200]}")

print("\n" + "=" * 80)
print("QUICK TEST COMPLETE")
print("=" * 80)
print("\n✅ If outputs look good, run full evaluation:")
print("   python evaluate_finetuned.py")
print("=" * 80)
