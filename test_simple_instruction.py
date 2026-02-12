#!/usr/bin/env python3
"""
초단순 Instruction 테스트 - AI Refusal 우회
"""

import torch
from medalpaca.inferer import Inferer

print("Loading MedAlpaca-7b...")
inferer = Inferer(
    model_name="medalpaca/medalpaca-7b",
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,
    torch_dtype=torch.float16,
)

# ========================================
# Strategy 1: 의료 맥락 제거 (데이터 분석으로 프레임)
# ========================================
print("\n" + "=" * 80)
print("Strategy 1: Research/Data Analysis Frame")
print("=" * 80)

test1_instruction = "Analyze the following physiological data and predict the stress level (1-5)."
test1_input = "[Total Sleep]: 7.5 hours, [Deep Sleep]: 1.2 hours, [Resting Heart Rate]: 68 bpm"

response1 = inferer(
    instruction=test1_instruction,
    input=test1_input,
    max_new_tokens=50,
    temperature=0.1,
    repetition_penalty=1.2,  # 1.1 → 1.2로 증가
    do_sample=False,  # Greedy decoding
    verbose=False
)

print(f"Instruction: {test1_instruction}")
print(f"Input: {test1_input}")
print(f"Response: '{response1}'")
print()

# ========================================
# Strategy 2: Instruction 완전 제거
# ========================================
print("=" * 80)
print("Strategy 2: No Instruction (Input Only)")
print("=" * 80)

test2_input = "Predict stress level (1-5):\n[Total Sleep]: 7.5h, [Heart Rate]: 68 bpm\nAnswer:"

response2 = inferer(
    instruction="",
    input=test2_input,
    max_new_tokens=20,
    temperature=0.1,
    repetition_penalty=1.2,
    do_sample=False,
    verbose=False
)

print(f"Input: {test2_input}")
print(f"Response: '{response2}'")
print()

# ========================================
# Strategy 3: Old-style (원본 gen_dataset.py의 OLD 버전)
# ========================================
print("=" * 80)
print("Strategy 3: Old-style Truthful Answer")
print("=" * 80)

test3_instruction = "Answer this question truthfully"
test3_input = "Given the following data, predict the stress level (1-5). [Total Sleep]: 7.5 hours, [Deep Sleep]: 1.2 hours, [Resting Heart Rate]: 68 bpm"

response3 = inferer(
    instruction=test3_instruction,
    input=test3_input,
    max_new_tokens=50,
    temperature=0.1,
    repetition_penalty=1.2,
    do_sample=False,
    verbose=False
)

print(f"Instruction: {test3_instruction}")
print(f"Input: {test3_input}")
print(f"Response: '{response3}'")
print()

# ========================================
# Strategy 4: Activity 분류 (Echo 테스트)
# ========================================
print("=" * 80)
print("Strategy 4: Classification (Echo Test)")
print("=" * 80)

test4_instruction = "Predict the activity type among ['Sitting', 'Lying', 'Walking', 'Running']."
test4_input = "[Steps]: 3.92, [Calories]: 0.25, [Heart Rate]: 109.33 beats/min"

response4 = inferer(
    instruction=test4_instruction,
    input=test4_input,
    max_new_tokens=30,
    temperature=0.1,
    repetition_penalty=1.3,  # 더 강하게
    do_sample=False,
    verbose=False
)

print(f"Instruction: {test4_instruction}")
print(f"Input: {test4_input}")
print(f"Response: '{response4}'")
print()

# ========================================
# Strategy 5: 실제 finetune 데이터 형식 그대로
# ========================================
print("=" * 80)
print("Strategy 5: Exact Finetune Data Format")
print("=" * 80)

test5_instruction = "You are a personalized healthcare agent trained to predict stress which ranges from 1 to 5 based on physiological data and user information."
test5_input = "The recent 14-days sensor readings show: [Steps]: [120, 150, 200] steps, [Burned Calories]: [50, 60, 70] calories, [Resting Heart Rate]: [65, 68, 70] beats/min, [SleepMinutes]: [450, 480, 420] minutes, [Mood]: 4 out of 5; What would be the predicted stress?"

response5 = inferer(
    instruction=test5_instruction,
    input=test5_input,
    max_new_tokens=50,
    temperature=0.1,
    repetition_penalty=1.2,
    do_sample=False,
    verbose=False
)

print(f"Instruction: {test5_instruction[:80]}...")
print(f"Input: {test5_input[:80]}...")
print(f"Response: '{response5}'")
print()

# ========================================
# Summary
# ========================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print("Strategy 1 (Research frame): AI refusal 감소 기대")
print("Strategy 2 (No instruction): Baseline 동작 확인")
print("Strategy 3 (Old-style): 원본 저자 OLD 버전")
print("Strategy 4 (Classification): Echo 감소 확인 (rep_penalty=1.3)")
print("Strategy 5 (Finetune format): 실제 학습 데이터와 동일")
print()
print("→ 가장 좋은 결과를 보이는 Strategy를 baseline/finetuned 평가에 적용")
