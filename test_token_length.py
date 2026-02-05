#!/usr/bin/env python3
"""
토큰 길이 확인 - Truncation 문제 진단
"""
import torch
from transformers import LlamaTokenizer
from medalpaca.handler import DataHandler

print("Loading tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained("medalpaca/medalpaca-7b")
tokenizer.pad_token_id = 0

# DataHandler 생성 (model_max_length=2048)
handler = DataHandler(
    tokenizer=tokenizer,
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,
    train_on_inputs=False
)

# 테스트 1: 짧은 입력
print("\n" + "=" * 80)
print("TEST 1: Short input")
print("=" * 80)
short_prompt = handler.generate_prompt(
    instruction="You are a personalized healthcare agent trained to predict stress which ranges from 1 to 5.",
    input="Steps: 2000, Calories: 150, Heart Rate: 65 bpm, Sleep: 7 hours. Predict stress.",
    output=""
)
short_tokens = handler.tokenizer(short_prompt, return_tensors=None, add_special_tokens=False)
print(f"Prompt length: {len(short_prompt)} chars")
print(f"Token count: {len(short_tokens['input_ids'])} tokens")

# 테스트 2: 긴 입력 (14일 데이터)
print("\n" + "=" * 80)
print("TEST 2: Long input (14-day data)")
print("=" * 80)
long_input = "The recent 14-days sensor readings show: [Steps]: [2571.0, 2067.0, 1848.0, 2058.0, 1893.0, 2240.0, 3363.0, 3239.0, 3042.0, 2806.0, 4755.0, 2369.0, 2227.0, 1654.0] steps, [Burned Calorories]: [199.0, 173.0, 134.0, 181.0, 152.0, 158.0, 253.0, 216.0, 217.0, 208.0, 333.0, 166.0, 155.0, 125.0] calories, [Resting Heart Rate]: [66.37638664245605, 67.78313732147217, 67.10895824432373, 66.16707134246826, 67.27083015441895, 67.46240520477295, 68.03761386871338, 66.51346015930176, 66.74281787872314, 66.69368553161621, 65.64037036895752, 65.86407661437988, 67.73857021331787, 67.28505611419678] beats/min, [SleepMinutes]: [495.0, 436.0, 479.0, 314.0, 564.0, 381.0, 439.0, 460.0, 540.0, 511.0, 468.0, 602.0] minutes, [Mood]: 3 out of 5; What would be the predicted stress?"

long_prompt = handler.generate_prompt(
    instruction="You are a personalized healthcare agent trained to predict stress which ranges from 1 to 5 based on physiological data and user information.",
    input=long_input,
    output=""
)
long_tokens = handler.tokenizer(long_prompt, return_tensors=None, add_special_tokens=False)
print(f"Prompt length: {len(long_prompt)} chars")
print(f"Token count: {len(long_tokens['input_ids'])} tokens")

# 테스트 3: Tokenizer에서 truncation 확인
print("\n" + "=" * 80)
print("TEST 3: Tokenization with truncation")
print("=" * 80)

# handler.tokenize() 사용 (우리가 수정한 메서드)
tokenized = handler.tokenize(long_prompt, add_eos_token=True)
print(f"After tokenize(): {len(tokenized['input_ids'])} tokens")

# 직접 tokenizer 호출 (max_length 없이)
direct_tokens = tokenizer(long_prompt, return_tensors=None, add_special_tokens=False)
print(f"Direct tokenizer (no max_length): {len(direct_tokens['input_ids'])} tokens")

# 직접 tokenizer 호출 (max_length=2048)
truncated_tokens = tokenizer(
    long_prompt,
    return_tensors=None,
    add_special_tokens=False,
    max_length=2048,
    truncation=True
)
print(f"Direct tokenizer (max_length=2048): {len(truncated_tokens['input_ids'])} tokens")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
if len(long_tokens['input_ids']) > 512:
    print(f"⚠️  Input has {len(long_tokens['input_ids'])} tokens (> 512)")
    if len(tokenized['input_ids']) <= 512:
        print("❌ handler.tokenize() is TRUNCATING to 512!")
        print("   → This is the problem!")
    elif len(tokenized['input_ids']) >= 2048:
        print("✓ handler.tokenize() NOT truncating")
    else:
        print(f"✓ handler.tokenize() truncating to {len(tokenized['input_ids'])} (< 2048)")
else:
    print("✓ Input is short enough")
