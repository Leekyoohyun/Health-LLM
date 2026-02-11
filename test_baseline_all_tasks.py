#!/usr/bin/env python3
"""
Baseline (MedAlpaca-7b) 전체 태스크 샘플 테스트
8개 task에서 각각 첫 번째 샘플 1개씩 테스트
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========================================
# 8개 Task 설정
# ========================================
TASKS = [
    {"name": "PMData_stress", "file": "evaluation-json-data/PMData_stress_train_all.json"},
    {"name": "PMData_readiness", "file": "evaluation-json-data/PMData_readiness_train_all.json"},
    {"name": "PMData_sleep_quality", "file": "evaluation-json-data/PMData_sleep_quality_train_all.json"},
    {"name": "PMData_fatigue", "file": "evaluation-json-data/PMData_fatigue_train_all.json"},
    {"name": "LifeSnaps_stress_resilience", "file": "evaluation-json-data/LifeSnaps_stress_resilience_train_all.json"},
    {"name": "LifeSnaps_sleep_disorder", "file": "evaluation-json-data/LifeSnaps_sleep_disorder_train_all.json"},
    {"name": "AW_FB_activity", "file": "evaluation-json-data/AW_FB_activity_train_all.json"},
    {"name": "AW_FB_calories", "file": "evaluation-json-data/AW_FB_calories_train_all.json"},
]

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
# 모델 로드
# ========================================
print("=" * 80)
print("Loading MedAlpaca-7b...")
print("=" * 80)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

model = AutoModelForCausalLM.from_pretrained(
    "medalpaca/medalpaca-7b",
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-7b")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✓ Model loaded\n")

# ========================================
# 전체 Task 테스트 (각 1개 샘플)
# ========================================
print("=" * 80)
print("BASELINE EVALUATION - All Tasks (1 sample each)")
print("=" * 80)

all_results = []

for task_idx, task in enumerate(TASKS):
    print(f"\n{'=' * 80}")
    print(f"[Task {task_idx+1}/8] {task['name']}")
    print("=" * 80)

    try:
        # 데이터 로드
        with open(task['file']) as f:
            data = json.load(f)

        print(f"Total samples: {len(data)}")

        # 첫 번째 샘플
        sample = data[0]
        instruction = sample['instruction']
        input_text = sample['input']
        ground_truth = sample['output']

        # Prompt 생성
        question = format_prompt(instruction, input_text)

        print(f"\nInstruction: {instruction[:100]}...")
        print(f"Input (first 150 chars): {input_text[:150]}...")
        print(f"Ground truth: {ground_truth}")
        print(f"Prompt length: {len(question)} chars")

        # Tokenize
        inputs = tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=2048,  # 2048로 증가
            return_token_type_ids=False
        ).to(model.device)

        input_length = inputs.input_ids.shape[1]
        print(f"Input tokens: {input_length}")

        # 실제 truncated prompt 확인
        truncated_prompt = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        print(f"Truncated prompt (last 150 chars): ...{truncated_prompt[-150:]}")

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                repetition_penalty=1.5,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated_tokens = outputs[0][input_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        print(f"\n✅ Generated answer: {answer}")

        # 결과 저장
        all_results.append({
            'task': task['name'],
            'ground_truth': ground_truth,
            'predicted': answer,
            'input_tokens': input_length,
            'output_tokens': len(generated_tokens),
            'status': 'SUCCESS'
        })

    except FileNotFoundError:
        print(f"⚠️  File not found: {task['file']}")
        all_results.append({
            'task': task['name'],
            'status': 'FILE_NOT_FOUND'
        })
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)[:150]}")
        all_results.append({
            'task': task['name'],
            'status': 'ERROR',
            'error': str(e)[:150]
        })

# ========================================
# 결과 요약
# ========================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

for result in all_results:
    if result['status'] == 'SUCCESS':
        print(f"\n[{result['task']}]")
        print(f"  Ground Truth: {result['ground_truth']}")
        print(f"  Predicted:    {result['predicted']}")
        print(f"  Tokens: {result['input_tokens']} → {result['output_tokens']}")
    else:
        print(f"\n[{result['task']}] ❌ {result['status']}")

# 결과 저장
output_file = "baseline_all_tasks_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Results saved to: {output_file}")
print("=" * 80)
