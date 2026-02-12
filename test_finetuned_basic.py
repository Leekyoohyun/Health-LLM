#!/usr/bin/env python3
"""
Fine-tuned (LoRA) 모델 기본 테스트
AI Refusal이 사라졌는지 확인
"""

import torch
from medalpaca.inferer import Inferer

print("=" * 80)
print("Fine-tuned MedAlpaca-7b + LoRA Test")
print("=" * 80)

# LoRA adapter 로드
print("Loading MedAlpaca-7b + LoRA adapter...")
inferer = Inferer(
    model_name="outputs/healthalpaca-7b-lora",  # LoRA adapter path
    base_model="medalpaca/medalpaca-7b",  # Base model
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,
    torch_dtype=torch.float16,
    peft=True  # LoRA 사용
)
print("✓ Model + LoRA loaded\n")

# ========================================
# Test 1: Stress 예측 (PMData 형식)
# ========================================
print("=" * 80)
print("Test 1: Stress Prediction (PMData format)")
print("=" * 80)

test1_instruction = "You are a personalized healthcare agent trained to predict stress which ranges from 1 to 5 based on physiological data and user information."
test1_input = "The recent 14-days sensor readings show: [Steps]: [120, 150, 200] steps, [Burned Calories]: [50, 60, 70] calories, [Resting Heart Rate]: [65, 68, 70] beats/min, [SleepMinutes]: [450, 480, 420] minutes, [Mood]: 4 out of 5; What would be the predicted stress?"

response1 = inferer(
    instruction=test1_instruction,
    input=test1_input,
    max_new_tokens=100,
    temperature=0.1,
    repetition_penalty=1.1,
    do_sample=False,
    verbose=False
)

print(f"Instruction: {test1_instruction[:80]}...")
print(f"Input: {test1_input[:80]}...")
print(f"Response: '{response1}'")
print(f"Length: {len(response1)} chars")
print()

# AI Refusal 체크
if "do not provide medical advice" in response1.lower():
    print("⚠️  AI Refusal detected!")
else:
    print("✅ No AI refusal")

# Echo 체크
if test1_input[:50] in response1:
    print("⚠️  Echo detected!")
else:
    print("✅ No echo")
print()

# ========================================
# Test 2: Activity 분류 (AW_FB 형식)
# ========================================
print("=" * 80)
print("Test 2: Activity Classification (AW_FB format)")
print("=" * 80)

test2_instruction = "You are a personalized healthcare agent trained to predict the type of activity among ['Self Pace Walk', 'Sitting', 'Lying', 'Running 7 METs', 'Running 5 METs', 'Running 3 METs'] based on physiological data and user information."
test2_input = "The recent sensor readings show: [Steps]: 3.92 steps, [Burned Calorories]: 0.25 calories, [Heart Rate]: 109.33 beats/min; What would be the predicted activity type?"

response2 = inferer(
    instruction=test2_instruction,
    input=test2_input,
    max_new_tokens=100,
    temperature=0.1,
    repetition_penalty=1.1,
    do_sample=False,
    verbose=False
)

print(f"Instruction: {test2_instruction[:80]}...")
print(f"Input: {test2_input}")
print(f"Response: '{response2}'")
print()

# Echo 체크
if test2_input[:50] in response2:
    print("⚠️  Echo detected!")
else:
    print("✅ No echo")
print()

# ========================================
# Test 3: Stress Resilience (LifeSnaps 형식)
# ========================================
print("=" * 80)
print("Test 3: Stress Resilience (LifeSnaps format)")
print("=" * 80)

test3_instruction = "You are a personalized healthcare agent trained to predict stress_resilience which ranges from 0.2 to 5 based on physiological data and user information."
test3_input = "The recent 7-days sensor readings show: [Stress Score]: [60, 65, 70] out of 100, [Positive Affect Score]: 35 out of 50, [Negative Affect Score]: 20 out of 50, [Lightly Active Minutes]: 30 minutes, [Moderately Active Minutes]: 15 minutes, [Very Active Minutes]: 5 minutes, [Sleep Efficiency]: 0.85, [Sleep Deep Ratio]: 0.2, [Sleep Light Ratio]: 0.5, [Sleep REM Ratio]: 0.3; What would be the predicted stress resilience index?"

response3 = inferer(
    instruction=test3_instruction,
    input=test3_input,
    max_new_tokens=100,
    temperature=0.1,
    repetition_penalty=1.1,
    do_sample=False,
    verbose=False
)

print(f"Instruction: {test3_instruction[:80]}...")
print(f"Input: {test3_input[:80]}...")
print(f"Response: '{response3}'")
print()

# ========================================
# Summary
# ========================================
print("=" * 80)
print("SUMMARY: Fine-tuned vs Baseline")
print("=" * 80)
print("Fine-tuning이 효과가 있다면:")
print("  ✅ AI refusal 사라짐")
print("  ✅ Echo 감소")
print("  ✅ 'The predicted X is Y' 형식 출력")
print()
print("Fine-tuning이 효과가 없다면:")
print("  ❌ 여전히 AI refusal")
print("  ❌ Echo 지속")
print("  → Baseline과 동일한 문제")
print()
print("→ 결과에 따라 평가 전략 결정")
