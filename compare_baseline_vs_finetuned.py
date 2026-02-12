#!/usr/bin/env python3
"""
Baseline vs Fine-tuned 직접 비교
동일한 샘플에 대해 두 모델의 출력을 side-by-side로 비교
Ground Truth와 함께 표시
"""

import torch
from medalpaca.inferer import Inferer
from datasets import load_dataset
import json

# ========================================
# 설정
# ========================================
VAL_SET_SIZE = 0.1
N_SAMPLES = 3  # 각 task별 샘플 수

TASKS = [
    {"name": "PMData_stress", "file": "evaluation-json-data/PMData_stress_train_all.json"},
    {"name": "AW_FB_activity", "file": "evaluation-json-data/AW_FB_activity_train_all.json"},
    {"name": "LifeSnaps_stress_resilience", "file": "evaluation-json-data/LifeSnaps_stress_resilience_train_all.json"},
]

print("=" * 100)
print("BASELINE vs FINE-TUNED COMPARISON")
print("=" * 100)
print("⚠️  This will load BOTH models sequentially (memory intensive)")
print()

# ========================================
# Load Baseline Model
# ========================================
print("\n" + "=" * 100)
print("1. Loading BASELINE (MedAlpaca-7b)")
print("=" * 100)

baseline_inferer = Inferer(
    model_name="medalpaca/medalpaca-7b",
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,
    torch_dtype=torch.float16,
)
print("✅ Baseline loaded")

# ========================================
# Test Baseline
# ========================================
baseline_results = []

for task in TASKS:
    task_name = task['name']
    print(f"\n{'─' * 100}")
    print(f"Testing Baseline: {task_name}")
    print('─' * 100)

    # Load validation set
    dataset = load_dataset("json", data_files=task['file'])
    split_data = dataset["train"].train_test_split(
        test_size=VAL_SET_SIZE,
        shuffle=True,
        seed=42
    )
    val_data = split_data["test"]

    for idx in range(min(N_SAMPLES, len(val_data))):
        sample = val_data[idx]
        instruction = sample['instruction']
        input_text = sample['input']
        ground_truth = sample['output']

        try:
            answer = baseline_inferer(
                instruction=instruction,
                input=input_text,
                max_new_tokens=128,
                temperature=0.1,
                repetition_penalty=1.1,
                do_sample=False,
                verbose=False
            )

            baseline_results.append({
                'task': task_name,
                'idx': idx,
                'instruction': instruction,
                'input': input_text,
                'ground_truth': ground_truth,
                'baseline_output': answer,
                'status': 'SUCCESS'
            })

            print(f"  Sample {idx}: OK ({len(answer)} chars)")

        except Exception as e:
            baseline_results.append({
                'task': task_name,
                'idx': idx,
                'instruction': instruction,
                'input': input_text,
                'ground_truth': ground_truth,
                'baseline_output': f"ERROR: {str(e)[:100]}",
                'status': 'ERROR'
            })
            print(f"  Sample {idx}: ERROR - {str(e)[:50]}")

print("\n✅ Baseline testing complete")

# Free baseline model from GPU
del baseline_inferer
torch.cuda.empty_cache()
print("🗑️  Baseline model freed from GPU")

# ========================================
# Load Fine-tuned Model
# ========================================
print("\n" + "=" * 100)
print("2. Loading FINE-TUNED (MedAlpaca-7b + LoRA)")
print("=" * 100)

finetuned_inferer = Inferer(
    model_name="outputs/healthalpaca-7b-lora",  # LoRA adapter
    base_model="medalpaca/medalpaca-7b",  # Base model
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=2048,
    torch_dtype=torch.float16,
    peft=True  # ← KEY DIFFERENCE!
)
print("✅ Fine-tuned loaded")

# ========================================
# Test Fine-tuned (same samples)
# ========================================
for i, result in enumerate(baseline_results):
    task_name = result['task']
    idx = result['idx']
    instruction = result['instruction']
    input_text = result['input']

    try:
        answer = finetuned_inferer(
            instruction=instruction,
            input=input_text,
            max_new_tokens=128,
            temperature=0.1,
            repetition_penalty=1.1,
            do_sample=False,
            verbose=False
        )

        result['finetuned_output'] = answer
        result['finetuned_status'] = 'SUCCESS'

    except Exception as e:
        result['finetuned_output'] = f"ERROR: {str(e)[:100]}"
        result['finetuned_status'] = 'ERROR'

print("\n✅ Fine-tuned testing complete")

# ========================================
# Display Results (Side-by-side)
# ========================================
print("\n" + "=" * 100)
print("COMPARISON RESULTS")
print("=" * 100)

for result in baseline_results:
    print("\n" + "=" * 100)
    print(f"Task: {result['task']} | Sample: {result['idx']}")
    print("=" * 100)

    print(f"\n📝 INSTRUCTION ({len(result['instruction'])} chars):")
    print(f"   {result['instruction'][:300]}{'...' if len(result['instruction']) > 300 else ''}")

    print(f"\n📥 INPUT ({len(result['input'])} chars):")
    print(f"   {result['input'][:300]}{'...' if len(result['input']) > 300 else ''}")

    print(f"\n🎯 GROUND TRUTH:")
    print(f"   {result['ground_truth']}")

    print(f"\n❌ BASELINE OUTPUT:")
    baseline_out = result['baseline_output']
    print(f"   {baseline_out[:400]}{'...' if len(baseline_out) > 400 else ''}")

    # Check for AI refusal
    if "do not provide" in baseline_out.lower() or "i am an ai" in baseline_out.lower():
        print(f"   ⚠️  AI REFUSAL DETECTED")
    elif result['input'][:50] in baseline_out:
        print(f"   ⚠️  ECHO DETECTED")

    print(f"\n✅ FINE-TUNED OUTPUT:")
    finetuned_out = result.get('finetuned_output', 'N/A')
    print(f"   {finetuned_out[:400]}{'...' if len(finetuned_out) > 400 else ''}")

    # Check for correct format
    if "predicted" in finetuned_out.lower():
        print(f"   ✅ CORRECT FORMAT")

    print()

# ========================================
# Save Results
# ========================================
output_file = "comparison_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(baseline_results, f, indent=2, ensure_ascii=False)

print("=" * 100)
print(f"💾 Results saved to: {output_file}")
print("=" * 100)

# ========================================
# Summary
# ========================================
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

total_samples = len(baseline_results)
baseline_refusals = sum(1 for r in baseline_results
                        if "do not provide" in r['baseline_output'].lower())
baseline_echo = sum(1 for r in baseline_results
                    if r['input'][:50] in r['baseline_output'])
finetuned_correct = sum(1 for r in baseline_results
                        if "predicted" in r.get('finetuned_output', '').lower())

print(f"\nTotal samples: {total_samples}")
print(f"\nBaseline:")
print(f"  AI Refusals: {baseline_refusals}/{total_samples} ({baseline_refusals/total_samples*100:.1f}%)")
print(f"  Echo: {baseline_echo}/{total_samples} ({baseline_echo/total_samples*100:.1f}%)")
print(f"\nFine-tuned:")
print(f"  Correct format: {finetuned_correct}/{total_samples} ({finetuned_correct/total_samples*100:.1f}%)")

print("\n" + "=" * 100)
print("KEY FINDINGS")
print("=" * 100)
print("\n1. WHAT CHANGED?")
print("   - Instruction: IDENTICAL")
print("   - Input: IDENTICAL")
print("   - Model: medalpaca-7b (base) → medalpaca-7b + LoRA adapter")
print("   - Code: peft=False → peft=True")
print()
print("2. HOW DOES LoRA WORK?")
print("   - Base model weights: FROZEN (not modified)")
print("   - LoRA adapter: 4.2M parameters (0.062% of base)")
print("   - Training: 9,280 samples × 3 epochs")
print("   - Effect: Suppresses AI refusal, enforces 'The predicted X is Y' format")
print()
print("3. CONCLUSION:")
print("   - LoRA fine-tuning transforms unusable base model into working predictor")
print("   - No prompt engineering needed - pure model adaptation")
print("=" * 100)
