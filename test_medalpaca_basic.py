#!/usr/bin/env python3
"""
MedAlpaca-7b 기본 동작 테스트
AI refusal, echo 문제 진단
"""

import torch
from medalpaca.inferer import Inferer

print("=" * 80)
print("MedAlpaca-7b Basic Test")
print("=" * 80)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Inferer 로드
print("Loading MedAlpaca-7b...")
inferer = Inferer(
    model_name="medalpaca/medalpaca-7b",
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,
    torch_dtype=torch.float16,
)
print("✓ Model loaded\n")

# ========================================
# Test 1: 기본 의료 질문
# ========================================
print("=" * 80)
print("Test 1: Basic Medical Question")
print("=" * 80)

test1_instruction = "You are a medical expert. Answer the following question concisely."
test1_input = "What is diabetes?"

print(f"Instruction: {test1_instruction}")
print(f"Input: {test1_input}")
print()

response1 = inferer(
    instruction=test1_instruction,
    input=test1_input,
    max_new_tokens=128,
    temperature=0.7,
    repetition_penalty=1.1,
    verbose=False
)

print(f"Response:\n{response1}")
print()

# ========================================
# Test 2: 숫자 예측 (regression)
# ========================================
print("=" * 80)
print("Test 2: Numeric Prediction (Stress Level)")
print("=" * 80)

test2_instruction = """You are a personalized healthcare agent trained to predict stress level (1-5) based on physiological data.
IMPORTANT: Output ONLY a single number between 1 and 5 (e.g., "3" or "4.2").
Do not include any explanations or additional text."""

test2_input = "The recent sensor readings show: [Total Sleep]: 7.5 hours, [Deep Sleep]: 1.2 hours, [Resting Heart Rate]: 68 bpm; What would be the predicted stress level?"

print(f"Instruction: {test2_instruction[:100]}...")
print(f"Input: {test2_input}")
print()

response2 = inferer(
    instruction=test2_instruction,
    input=test2_input,
    max_new_tokens=128,
    temperature=0.1,  # Low temperature for deterministic output
    repetition_penalty=1.1,
    verbose=False
)

print(f"Response: '{response2}'")
print(f"Length: {len(response2)} chars")
print()

# ========================================
# Test 3: 분류 (classification)
# ========================================
print("=" * 80)
print("Test 3: Classification (Activity Type)")
print("=" * 80)

test3_instruction = """You are a personalized healthcare agent trained to predict the type of activity among ['Self Pace Walk', 'Sitting', 'Lying', 'Running 7 METs', 'Running 5 METs', 'Running 3 METs'] based on physiological data.
IMPORTANT: Output ONLY the activity name.
Do not include any explanations or additional text."""

test3_input = "The recent sensor readings show: [Steps]: 3.92 steps, [Burned Calories]: 0.25 calories, [Heart Rate]: 109.33 beats/min; What would be the predicted activity type?"

print(f"Instruction: {test3_instruction[:100]}...")
print(f"Input: {test3_input}")
print()

response3 = inferer(
    instruction=test3_instruction,
    input=test3_input,
    max_new_tokens=128,
    temperature=0.1,
    repetition_penalty=1.1,
    verbose=False
)

print(f"Response: '{response3}'")
print()

# ========================================
# Test 4: Format 예시 추가 (원본 저자 스타일)
# ========================================
print("=" * 80)
print("Test 4: With Format Example (Original Author Style)")
print("=" * 80)

test4_instruction = """You are a personalized healthcare agent trained to predict stress level (1-5) based on physiological data."""

test4_input = """The recent sensor readings show: [Total Sleep]: 7.5 hours, [Deep Sleep]: 1.2 hours, [Resting Heart Rate]: 68 bpm; What would be the predicted stress level?

For example, the answer should be in the following format:
Answer: 3"""

print(f"Instruction: {test4_instruction}")
print(f"Input (with format example): {test4_input[:150]}...")
print()

response4 = inferer(
    instruction=test4_instruction,
    input=test4_input,
    max_new_tokens=128,
    temperature=0.1,
    repetition_penalty=1.1,
    verbose=False
)

print(f"Response: '{response4}'")
print()

# ========================================
# Test 5: Verbose로 전체 프롬프트 확인
# ========================================
print("=" * 80)
print("Test 5: Check Full Prompt (Verbose)")
print("=" * 80)

response5 = inferer(
    instruction=test2_instruction,
    input=test2_input,
    max_new_tokens=128,
    temperature=0.1,
    repetition_penalty=1.1,
    verbose=True  # Print full prompt
)

print(f"\nResponse: '{response5}'")
print()

print("=" * 80)
print("Test Complete")
print("=" * 80)
print("\n분석:")
print("1. Test 1이 정상 답변 → 모델은 기본적으로 작동함")
print("2. Test 1이 echo/refusal → 모델 로딩 문제")
print("3. Test 2-3이 echo/refusal → instruction-following 능력 부족")
print("4. Test 4가 개선됨 → format 예시가 효과적")
print("5. Verbose 출력으로 프롬프트 구조 확인")
