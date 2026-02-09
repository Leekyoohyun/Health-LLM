#!/usr/bin/env python3
"""
Truncation 문제 디버깅
"""

import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-7b")

# Load template
with open("medalpaca/prompt_templates/medalpaca.json") as f:
    prompt_template = json.load(f)

def format_prompt(instruction, input_text):
    prompt = prompt_template["primer"]
    prompt += prompt_template["instruction"] + instruction
    prompt += prompt_template["input"] + input_text
    prompt += prompt_template["output"]
    return prompt

# Load data
with open("evaluation-json-data/PMData_stress_train_all.json") as f:
    data = json.load(f)

sample = data[0]
instruction = sample['instruction']
input_text = sample['input']

# Format prompt
question = format_prompt(instruction, input_text)

print(f"Full prompt length: {len(question)} chars")
print()

# Tokenize with different max_lengths
for max_len in [256, 512, 1024]:
    inputs = tokenizer(
        question,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    token_count = inputs.input_ids.shape[1]
    truncated = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)

    has_response = "### Response:" in truncated

    print(f"Max length: {max_len}")
    print(f"Actual tokens: {token_count}")
    print(f"Has '### Response:': {'✅ YES' if has_response else '❌ NO'}")
    print(f"Last 150 chars: ...{truncated[-150:]}")
    print("-" * 80)
    print()
