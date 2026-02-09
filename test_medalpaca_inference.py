#!/usr/bin/env python3
"""
MedAlpaca inference 테스트 스크립트
- pipeline으로 medalpaca_pl 정의
- 샘플 몇 개만 실행해서 정상 동작 확인
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# padding token 설정 (없으면 eos_token 사용)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✓ Model loaded to GPU\n")

# ========================================
# Generation Config 확인
# ========================================
print("=== Generation Config ===")
print(f"max_length: {model.generation_config.max_length}")
print(f"max_new_tokens: {model.generation_config.max_new_tokens}")
print(f"temperature: {model.generation_config.temperature}")
print(f"top_p: {model.generation_config.top_p}")
print(f"do_sample: {model.generation_config.do_sample}")
print(f"repetition_penalty: {model.generation_config.repetition_penalty}")
print()

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
    print(f"Full prompt length: {len(question)} chars")

    try:
        # Tokenize input (학습 시와 동일하게 max_length=256)
        inputs = tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=1024,  # 학습 시와 동일!
            return_token_type_ids=False  # LLaMA 계열 모델은 사용 안 함
        ).to(model.device)

        input_length = inputs.input_ids.shape[1]

        # 실제 입력된 prompt 확인 (디버깅)
        truncated_prompt = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        print(f"Truncated prompt (last 200 chars): ...{truncated_prompt[-200:]}")

        # Generate (입력 제외하고 새로운 토큰만 생성)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,  # greedy decoding
                repetition_penalty=1.5,  # echo 방지!
                pad_token_id=tokenizer.eos_token_id,
            )

        # 전체 output 확인 (디버깅용)
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n=== Full output (last 300 chars) ===")
        print(f"...{full_output[-300:]}")

        # 생성된 부분만 디코딩 (입력 제외)
        generated_tokens = outputs[0][input_length:]
        ans = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        print(f"\n✓ SUCCESS")
        print(f"Input tokens: {input_length}")
        print(f"Output tokens: {len(outputs[0])}")
        print(f"Generated tokens: {len(generated_tokens)}")
        print(f"Generated answer: {ans}")

    except Exception as e:
        ans = "N/A"
        print(f"❌ EXCEPTION: {type(e).__name__}: {str(e)[:100]}")
        print(f"Answer set to: N/A")

    print("-" * 80)

print("\n" + "=" * 80)
print("Test complete!")
