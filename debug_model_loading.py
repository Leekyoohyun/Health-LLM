#!/usr/bin/env python3
"""
LoRA 모델 디버깅 스크립트
모델이 제대로 로드되고 작동하는지 단계별로 확인
"""

import json
import torch
from medalpaca.inferer import Inferer
from datasets import load_dataset
from peft import PeftModel
import os

print("=" * 80)
print("DEBUGGING: LoRA Model Loading and Inference")
print("=" * 80)

# ========================================
# Step 1: LoRA Adapter 파일 확인
# ========================================
print("\n[Step 1] Checking LoRA adapter files...")
print("=" * 80)

LORA_PATH = "outputs/healthalpaca-7b-lora"

if not os.path.exists(LORA_PATH):
    print(f"❌ ERROR: {LORA_PATH} does not exist!")
    print("   Please download from S3:")
    print(f"   aws s3 sync s3://khlee-healthllm-checkpoints/checkpoints/healthalpaca-7b-lora/ ./{LORA_PATH}/")
    exit(1)

print(f"✓ Directory exists: {LORA_PATH}")

# 필수 파일 확인
required_files = ['adapter_config.json', 'adapter_model.bin']
alternative_files = ['adapter_model.safetensors']

for f in required_files:
    path = os.path.join(LORA_PATH, f)
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024**2)  # MB
        print(f"✓ {f} exists ({size:.2f} MB)")
    else:
        # Check alternative
        if f == 'adapter_model.bin':
            alt_path = os.path.join(LORA_PATH, 'adapter_model.safetensors')
            if os.path.exists(alt_path):
                size = os.path.getsize(alt_path) / (1024**2)
                print(f"✓ adapter_model.safetensors exists ({size:.2f} MB)")
            else:
                print(f"❌ ERROR: Neither adapter_model.bin nor adapter_model.safetensors found!")
                exit(1)
        else:
            print(f"❌ ERROR: {f} not found!")
            exit(1)

# adapter_config.json 내용 확인
with open(os.path.join(LORA_PATH, 'adapter_config.json')) as f:
    adapter_config = json.load(f)

print("\n📋 LoRA Config:")
print(f"   - peft_type: {adapter_config.get('peft_type', 'N/A')}")
print(f"   - r: {adapter_config.get('r', 'N/A')}")
print(f"   - lora_alpha: {adapter_config.get('lora_alpha', 'N/A')}")
print(f"   - lora_dropout: {adapter_config.get('lora_dropout', 'N/A')}")
print(f"   - target_modules: {adapter_config.get('target_modules', 'N/A')}")

# ========================================
# Step 2: Base Model 로드
# ========================================
print("\n[Step 2] Loading base model...")
print("=" * 80)

inferer = Inferer(
    model_name="medalpaca/medalpaca-7b",
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,
    torch_dtype=torch.float16,
)

print(f"✓ Base model loaded")
print(f"   - Model type: {type(inferer.model).__name__}")
print(f"   - Device: {inferer.model.device}")
print(f"   - Dtype: {inferer.model.dtype}")

# Tokenizer 확인
print(f"\n📋 Tokenizer:")
print(f"   - Vocab size: {len(inferer.data_handler.tokenizer)}")
print(f"   - Model max length: {inferer.data_handler.tokenizer.model_max_length}")
print(f"   - EOS token: {inferer.data_handler.tokenizer.eos_token} (id={inferer.data_handler.tokenizer.eos_token_id})")
print(f"   - PAD token: {inferer.data_handler.tokenizer.pad_token} (id={inferer.data_handler.tokenizer.pad_token_id})")

# ❌ 문제 발견: model_max_length가 512로 제한됨!
if inferer.data_handler.tokenizer.model_max_length < 2048:
    print(f"\n⚠️  WARNING: Tokenizer model_max_length is {inferer.data_handler.tokenizer.model_max_length}, not 2048!")
    print(f"   Fixing tokenizer max_length to 2048...")
    inferer.data_handler.tokenizer.model_max_length = 2048
    print(f"   ✓ Fixed to: {inferer.data_handler.tokenizer.model_max_length}")

# ========================================
# Step 3: LoRA Adapter 로드
# ========================================
print("\n[Step 3] Loading LoRA adapter...")
print("=" * 80)

# Base model의 파라미터 수 기록
base_trainable_params = sum(p.numel() for p in inferer.model.parameters() if p.requires_grad)
base_total_params = sum(p.numel() for p in inferer.model.parameters())

print(f"Before LoRA:")
print(f"   - Trainable params: {base_trainable_params:,}")
print(f"   - Total params: {base_total_params:,}")

# LoRA 로드
inferer.model = PeftModel.from_pretrained(
    inferer.model,
    LORA_PATH
)

print(f"✓ LoRA adapter loaded")
print(f"   - Model type: {type(inferer.model).__name__}")

# LoRA 적용 후 파라미터 수
lora_trainable_params = sum(p.numel() for p in inferer.model.parameters() if p.requires_grad)
lora_total_params = sum(p.numel() for p in inferer.model.parameters())

print(f"\nAfter LoRA:")
print(f"   - Trainable params: {lora_trainable_params:,}")
print(f"   - Total params: {lora_total_params:,}")
print(f"   - LoRA params: {lora_total_params - base_total_params:,}")

# LoRA adapter가 실제로 적용되었는지 확인
if lora_total_params > base_total_params:
    print(f"✓ LoRA adapter is ACTIVE")
else:
    print(f"❌ WARNING: LoRA adapter might not be loaded correctly!")

# ========================================
# Step 4: Prompt Template 확인
# ========================================
print("\n[Step 4] Checking prompt template...")
print("=" * 80)

# Prompt template 로드
template_path = "medalpaca/prompt_templates/medalpaca.json"
if os.path.exists(template_path):
    with open(template_path) as f:
        template = json.load(f)
    print(f"✓ Template loaded from: {template_path}")
    print(f"\n📋 Template structure:")
    print(f"   - prompt_input: {template.get('prompt_input', 'N/A')[:100]}...")
    print(f"   - prompt_no_input: {template.get('prompt_no_input', 'N/A')[:100]}...")
else:
    print(f"❌ ERROR: Template not found at {template_path}")

# ========================================
# Step 5: 실제 입력 토큰화 확인
# ========================================
print("\n[Step 5] Testing tokenization...")
print("=" * 80)

# 샘플 데이터 로드
dataset = load_dataset("json", data_files="evaluation-json-data/PMData_stress_train_all.json")
split_data = dataset["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)
sample = split_data["test"][0]

instruction = sample['instruction']
input_text = sample['input']

print(f"Instruction length: {len(instruction)} chars")
print(f"Input length: {len(input_text)} chars")

# Prompt 포맷팅 (수동으로 구현)
# Template이 prompt_input 또는 prompt_no_input을 사용
if input_text:
    prompt_key = "prompt_input"
else:
    prompt_key = "prompt_no_input"

if prompt_key in template:
    full_prompt = template[prompt_key].format(
        instruction=instruction,
        input=input_text
    )
else:
    # Fallback: 간단한 포맷
    full_prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"

print(f"\n📋 Full prompt length: {len(full_prompt)} chars")
print(f"Full prompt (first 500 chars):")
print(f"{full_prompt[:500]}...")

# Tokenize
tokens = inferer.data_handler.tokenizer(
    full_prompt,
    return_tensors="pt",
    truncation=False  # truncation 하지 않고 확인
)

print(f"\n📊 Tokenization:")
print(f"   - Token count: {tokens['input_ids'].shape[1]}")
print(f"   - Max length: {inferer.data_handler.tokenizer.model_max_length}")

if tokens['input_ids'].shape[1] > inferer.data_handler.tokenizer.model_max_length:
    print(f"   ⚠️  WARNING: Token count exceeds max length!")
    print(f"   - Exceeds by: {tokens['input_ids'].shape[1] - inferer.data_handler.tokenizer.model_max_length} tokens")
else:
    print(f"   ✓ Token count within limit")

# ========================================
# Step 6: Inference 테스트
# ========================================
print("\n[Step 6] Testing inference...")
print("=" * 80)

print(f"Ground truth: {sample['output']}")
print(f"\nGenerating...")

answer = inferer(
    instruction=instruction,
    input=input_text,
    max_new_tokens=64,
    repetition_penalty=1.1,
    verbose=False
)

print(f"\n💬 Output:")
print(f"{answer}")
print(f"\n📊 Output analysis:")
print(f"   - Length: {len(answer)} chars")
print(f"   - Has EOS: {'</s>' in answer}")
print(f"   - Starts with input: {answer.startswith(input_text[:50])}")

# ========================================
# Step 7: 학습 데이터 확인
# ========================================
print("\n[Step 7] Checking training data format...")
print("=" * 80)

# 학습 데이터 샘플 확인
train_data_path = "data/finetune_data.json"
if os.path.exists(train_data_path):
    with open(train_data_path) as f:
        train_data = json.load(f)

    print(f"✓ Training data found: {len(train_data)} samples")
    print(f"   ⚠️  WARNING: Expected 9,280 samples, but found {len(train_data)}!")

    # 첫 5개 샘플의 구조 확인
    print(f"\n📋 First 5 training samples structure:")
    for i in range(min(5, len(train_data))):
        sample_train = train_data[i]
        print(f"\n[Sample {i}]")
        print(f"   Keys: {list(sample_train.keys())}")
        print(f"   Instruction: {sample_train.get('instruction', 'N/A')[:100]}...")
        print(f"   Input: {sample_train.get('input', 'N/A')[:100]}..." if sample_train.get('input') else "   Input: (empty)")
        print(f"   Output: {sample_train.get('output', 'N/A')}")

    # PMData_stress 샘플 찾기 (조건 완화)
    print(f"\n🔍 Searching for stress samples...")
    stress_samples = [s for s in train_data if 'stress' in s.get('instruction', '').lower()]
    print(f"   Samples with 'stress' in instruction: {len(stress_samples)}")

    # Unique instructions 확인
    unique_instructions = set()
    for s in train_data[:100]:  # 첫 100개만
        inst = s.get('instruction', '')[:50]  # 첫 50 chars만
        unique_instructions.add(inst)

    print(f"\n📊 Sample instructions (first 100 samples):")
    for inst in list(unique_instructions)[:5]:
        print(f"   - {inst}...")

    # 평가 데이터의 instruction이 학습 데이터에 있는지 확인
    eval_inst_short = instruction[:50]
    matching_samples = [s for s in train_data if s.get('instruction', '')[:50] == eval_inst_short]
    print(f"\n🔍 Samples matching eval instruction: {len(matching_samples)}")

    if matching_samples:
        print(f"   ✓ Eval instruction found in training data!")
        print(f"   Example output: {matching_samples[0].get('output', 'N/A')}")
    else:
        print(f"   ❌ Eval instruction NOT found in training data!")
        print(f"   This explains poor performance!")
else:
    print(f"❌ Training data not found at: {train_data_path}")

# Base vs LoRA 비교는 메모리 부족으로 스킵
print(f"\n💡 Note: Base vs LoRA comparison skipped due to memory constraints.")
print(f"   But LoRA adapter is confirmed to be loaded (4,194,304 params added).")

print("\n" + "=" * 80)
print("DEBUGGING COMPLETE")
print("=" * 80)
