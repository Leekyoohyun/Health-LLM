#!/usr/bin/env python3
"""
실제 긴 입력 (14일 데이터)으로 모델 테스트
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

# 실제 14일 데이터 (evaluation에서 사용하는 것과 동일)
long_input = "The recent 14-days sensor readings show: [Steps]: [2571.0, 2067.0, 1848.0, 2058.0, 1893.0, 2240.0, 3363.0, 3239.0, 3042.0, 2806.0, 4755.0, 2369.0, 2227.0, 1654.0] steps, [Burned Calorories]: [199.0, 173.0, 134.0, 181.0, 152.0, 158.0, 253.0, 216.0, 217.0, 208.0, 333.0, 166.0, 155.0, 125.0] calories, [Resting Heart Rate]: [66.37638664245605, 67.78313732147217, 67.10895824432373, 66.16707134246826, 67.27083015441895, 67.46240520477295, 68.03761386871338, 66.51346015930176, 66.74281787872314, 66.69368553161621, 65.64037036895752, 65.86407661437988, 67.73857021331787, 67.28505611419678] beats/min, [SleepMinutes]: [495.0, 436.0, 479.0, 314.0, 564.0, 381.0, 439.0, 460.0, 540.0, 511.0, 468.0, 602.0] minutes, [Mood]: 3 out of 5; What would be the predicted stress?"

instruction = "You are a personalized healthcare agent trained to predict stress which ranges from 1 to 5 based on physiological data and user information."

print("\n" + "=" * 80)
print("TEST: 14-day sensor data (709 tokens)")
print("=" * 80)
print(f"Expected ground truth: The predicted stress level is 2.")
print()

response = model(
    instruction=instruction,
    input=long_input,
    max_new_tokens=128,
    verbose=True  # Will show prompt and token count
)

print("\n" + "=" * 80)
print(f"Response: {response}")
print(f"Response length: {len(response)} chars")
print("=" * 80)

# 분석
if "predicted stress" in response.lower() and len(response) < 100:
    print("✓ Model works correctly!")
elif long_input[:50] in response:
    print("✗ Model is echoing input")
else:
    print("⚠ Model response is unclear")
