#!/usr/bin/env python3
"""
짧은 입력으로 모델 테스트 - 문제 진단용
"""
import torch
from medalpaca.inferer import Inferer

print("Loading model...")
model = Inferer(
    model_name="outputs/healthalpaca-7b-lora",
    base_model="medalpaca/medalpaca-7b",
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,
    torch_dtype=torch.float16,
    peft=True,
)

# 테스트 1: 매우 짧은 입력
print("\n" + "=" * 80)
print("TEST 1: Very short input")
print("=" * 80)
response = model(
    instruction="You are a personalized healthcare agent trained to predict stress which ranges from 1 to 5.",
    input="Steps: 2000, Calories: 150, Heart Rate: 65 bpm, Sleep: 7 hours. Predict stress.",
    max_new_tokens=50,
    verbose=True
)
print(f"\nResponse: {response}")
print(f"Length: {len(response)} chars")

# 테스트 2: 중간 길이 입력
print("\n" + "=" * 80)
print("TEST 2: Medium input")
print("=" * 80)
response2 = model(
    instruction="You are a personalized healthcare agent trained to predict stress which ranges from 1 to 5.",
    input="Recent 7-day readings: Steps=[2571, 2067, 1800, 2200, 2500, 1900, 2100], Calories=[199, 173, 180, 200, 210, 170, 190], Heart Rate=[66, 67, 68, 65, 66, 67, 66] bpm. What is the predicted stress?",
    max_new_tokens=50,
    verbose=True
)
print(f"\nResponse: {response2}")
print(f"Length: {len(response2)} chars")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)
if "predicted stress" in response.lower():
    print("✓ Short input works correctly")
else:
    print("✗ Short input also echoes - Model is broken")

if "predicted stress" in response2.lower():
    print("✓ Medium input works correctly")
else:
    print("✗ Medium input also echoes - Model issue confirmed")
