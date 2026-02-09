#!/usr/bin/env python3
"""
MedAlpaca inference 테스트 스크립트
- pipeline으로 medalpaca_pl 정의
- 샘플 몇 개만 실행해서 정상 동작 확인
"""

import json
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

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
# medalpaca_pl 정의 (HuggingFace pipeline)
# ========================================
print("Loading MedAlpaca-7b with pipeline...")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# 모델을 GPU로 명시적으로 로드
model = AutoModelForCausalLM.from_pretrained(
    "medalpaca/medalpaca-7b",
    device_map="auto",  # 자동으로 GPU에 배치
    torch_dtype=torch.float16,  # 메모리 절약
)
tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-7b")

medalpaca_pl = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
)
print("✓ Model loaded to GPU\n")


# ========================================
# 테스트 데이터 로드
# ========================================
print("Loading test data...")
data_file = "evaluation-json-data/PMData_stress_train_all.json"
with open(data_file) as f:
    data = json.load(f)

print(f"✓ Loaded {len(data)} samples from {data_file}\n")


# ========================================
# 샘플 몇 개만 테스트
# ========================================
NUM_TEST_SAMPLES = 3

print(f"Testing {NUM_TEST_SAMPLES} samples...\n")
print("=" * 80)

for i in range(NUM_TEST_SAMPLES):
    sample = data[i]
    instruction = sample['instruction']
    input_text = sample['input']
    ground_truth = sample['output']

    # Prompt template 적용
    question = format_prompt(instruction, input_text)

    print(f"\n[Sample {i+1}]")
    print(f"Input (first 200 chars): {input_text[:200]}...")
    print(f"Ground truth: {ground_truth}")

    try:
        # inference 실행
        result = medalpaca_pl(question)
        full_output = result[0]['generated_text']

        # Response 부분만 추출 (template의 response_split 사용)
        if prompt_template["response_split"] in full_output:
            ans = full_output.split(prompt_template["response_split"])[-1].strip()
        else:
            ans = full_output

        print(f"✓ SUCCESS")
        print(f"Generated answer: {ans}")

    except Exception as e:
        ans = "N/A"
        print(f"❌ EXCEPTION: {type(e).__name__}: {str(e)[:100]}")
        print(f"Answer set to: N/A")

    print("-" * 80)

print("\n" + "=" * 80)
print("Test complete!")
