#!/usr/bin/env python3
"""
Tokenizer 설정 및 Input 길이 디버깅
"""

import torch
from medalpaca.inferer import Inferer
from datasets import load_dataset

print("=" * 100)
print("TOKENIZER DEBUG")
print("=" * 100)

# Load inferer
inferer = Inferer(
    model_name="outputs/healthalpaca-7b-lora",
    base_model="medalpaca/medalpaca-7b",
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,
    torch_dtype=torch.float16,
    peft=True
)

# Check tokenizer settings
tokenizer = inferer.data_handler.tokenizer
print(f"\n1. Tokenizer Settings:")
print(f"   model_max_length: {tokenizer.model_max_length}")
print(f"   DataHandler model_max_length: {inferer.data_handler.model_max_length}")

# Load a problematic sample
dataset = load_dataset("json", data_files="evaluation-json-data/PMData_stress_train_all.json")
split_data = dataset["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)
val_data = split_data["test"]

# Sample 1 (echo case)
sample = val_data[1]
instruction = sample['instruction']
input_text = sample['input']

print(f"\n2. Sample 1 (Echo Case):")
print(f"   Instruction length: {len(instruction)} chars")
print(f"   Input length: {len(input_text)} chars")

# Generate prompt
from medalpaca.handler import DataHandler
prompt = inferer.data_handler.generate_prompt(instruction=instruction, input=input_text)

print(f"\n3. Full Prompt:")
print(f"   Prompt length: {len(prompt)} chars")
print(f"   First 500 chars:")
print(f"   {prompt[:500]}")

# Tokenize
tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
input_ids = tokens['input_ids']
token_count = input_ids.shape[1]

print(f"\n4. Tokenization:")
print(f"   Total tokens: {token_count}")
print(f"   Will truncate? {token_count > 2048}")
print(f"   Will show warning? {token_count > tokenizer.model_max_length}")

# Tokenize with truncation
tokens_truncated = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=2048,
    add_special_tokens=False
)
truncated_count = tokens_truncated['input_ids'].shape[1]

print(f"\n5. After Truncation:")
print(f"   Tokens after truncation: {truncated_count}")
print(f"   Truncated? {truncated_count < token_count}")

# Decode to see what was truncated
if truncated_count < token_count:
    truncated_text = tokenizer.decode(tokens_truncated['input_ids'][0])
    print(f"\n6. Truncated Text (last 500 chars):")
    print(f"   ...{truncated_text[-500:]}")

# Test inference
print(f"\n7. Testing Inference:")
try:
    answer = inferer(
        instruction=instruction,
        input=input_text,
        max_new_tokens=128,
        temperature=0.1,
        repetition_penalty=1.1,
        do_sample=False,
        verbose=False
    )
    print(f"   Output: {answer[:200]}...")

    # Check for echo
    if input_text[:50] in answer:
        print(f"   ⚠️  ECHO DETECTED!")
    else:
        print(f"   ✅ No echo")

except Exception as e:
    print(f"   ERROR: {str(e)[:200]}")

print("\n" + "=" * 100)
print("ANALYSIS")
print("=" * 100)

print("\n1. If tokenizer.model_max_length != 2048:")
print("   → inferer.py modification NOT applied (module cached?)")
print("   → Solution: Restart Python or reload module")

print("\n2. If tokens > 2048:")
print("   → Input is too long even for 2048")
print("   → Solution: Increase to 4096 or preprocess input")

print("\n3. If truncation occurs:")
print("   → Important parts of prompt may be cut")
print("   → Model sees incomplete instruction")
print("   → May cause echo or gibberish")

print("\n4. If echo still occurs with proper tokenizer:")
print("   → Model was NOT trained on such long inputs")
print("   → Training data had shorter sequences")
print("   → Solution: Filter or summarize long inputs")

print("=" * 100)
