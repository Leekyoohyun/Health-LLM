#!/usr/bin/env python3
"""
Fine-tuned만 평가 (기존 baseline 결과 재사용)
"""
import os
import gc
import json
import re
import torch
import numpy as np
import subprocess
import time
from datetime import datetime
from datasets import load_dataset
from medalpaca.inferer import Inferer

# 기존 baseline 결과 파일 경로
BASELINE_INTERIM = "results/baseline_interim_20260205_073432.json"

# 설정
MAX_SAMPLES_PER_TASK = 10
S3_BUCKET = "khlee-healthllm-checkpoints"
OUTPUT_DIR = "results"
TIMESTAMP = "20260205_073432"  # Baseline과 동일한 타임스탬프 사용

TASKS = {
    "PMData_stress": {
        "data_path": "PMData_stress_train_all.json",
        "task_type": "regression",
        "description": "Stress Prediction (1-5)"
    },
}

def load_finetuned_model(model_max_length=2048):
    print("   Loading Finetuned HealthAlpaca...")
    model = Inferer(
        model_name="outputs/healthalpaca-7b-lora",
        base_model="medalpaca/medalpaca-7b",
        prompt_template="medalpaca/prompt_templates/medalpaca.json",
        model_max_length=model_max_length,
        torch_dtype=torch.float16,
        peft=True,
    )
    mem = torch.cuda.memory_allocated(0) / 1024**2
    print(f"   Finetuned model loaded ({mem:.0f} MB)")
    return model

def run_inference(model, test_data, label="Model", max_new_tokens=256, max_samples=None):
    samples = test_data if max_samples is None else test_data.select(range(min(max_samples, len(test_data))))
    total = len(samples)
    outputs = []
    
    for i, sample in enumerate(samples):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"      [{label}] [{i+1}/{total}]", end='\r')
        
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        
        output = model(
            instruction=instruction if instruction else None,
            input=input_text,
            max_new_tokens=max_new_tokens
        )
        outputs.append(output)
    
    print(f"      [{label}] {total}/{total} done          ")
    return outputs

def extract_number(text):
    try:
        return float(text.strip())
    except:
        pass
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[0])
        except:
            pass
    return None

def calculate_task_metrics(results, task_type):
    baseline_scores = []
    finetuned_scores = []
    
    for r in results:
        gt = r['ground_truth']
        baseline_pred = r['baseline_output']
        finetuned_pred = r['finetuned_output']
        
        if task_type == "regression":
            gt_num = extract_number(gt)
            baseline_num = extract_number(baseline_pred)
            finetuned_num = extract_number(finetuned_pred)
            
            if gt_num is not None:
                if baseline_num is not None:
                    baseline_scores.append(abs(gt_num - baseline_num))
                else:
                    baseline_scores.append(abs(gt_num))
                
                if finetuned_num is not None:
                    finetuned_scores.append(abs(gt_num - finetuned_num))
                else:
                    finetuned_scores.append(abs(gt_num))
    
    return {
        'metric': 'MAE',
        'baseline': np.mean(baseline_scores) if baseline_scores else None,
        'finetuned': np.mean(finetuned_scores) if finetuned_scores else None,
        'num_samples': len(baseline_scores)
    }

def load_test_data(data_path, seed=42):
    print(f"      Loading: {data_path}")
    data = load_dataset("json", data_files=data_path)
    split = data["train"].train_test_split(test_size=0.1, shuffle=True, seed=seed)
    test_data = split["test"]
    print(f"      Test samples: {len(test_data)}")
    return test_data

print("=" * 80)
print("Fine-tuned Only Evaluation (Reusing Baseline Results)")
print("=" * 80)

# Load baseline results
print(f"\n[Step 1] Loading baseline results from {BASELINE_INTERIM}")
with open(BASELINE_INTERIM) as f:
    baseline_interim = json.load(f)

baseline_all_outputs = {}
for task_name, data in baseline_interim.items():
    baseline_all_outputs[task_name] = {
        'outputs': data['outputs'],
        'duration_seconds': data['duration_seconds']
    }
    print(f"   ✓ Loaded {task_name}: {data['num_samples']} samples")

# Load finetuned model
print("\n[Step 2] Loading Finetuned model...")
finetuned_model = load_finetuned_model(model_max_length=2048)

# Run finetuned inference
print("\n[Step 3] Running Finetuned inference...")

all_test_data = {}
for task_name, task_config in TASKS.items():
    print(f"\n   --- Task: {task_name} ---")
    
    test_data = load_test_data(task_config['data_path'])
    all_test_data[task_name] = test_data
    
    task_start = time.time()
    finetuned_outputs = run_inference(
        finetuned_model, test_data,
        label=f"Finetuned/{task_name}",
        max_samples=MAX_SAMPLES_PER_TASK
    )
    finetuned_duration = time.time() - task_start
    
    baseline_outputs = baseline_all_outputs[task_name]['outputs']
    baseline_duration = baseline_all_outputs[task_name]['duration_seconds']
    
    # Combine results
    samples = test_data if MAX_SAMPLES_PER_TASK is None else test_data.select(range(min(MAX_SAMPLES_PER_TASK, len(test_data))))
    results = []
    for i, sample in enumerate(samples):
        results.append({
            'sample_id': i + 1,
            'instruction': sample.get('instruction', ''),
            'input': sample.get('input', ''),
            'ground_truth': sample.get('output', ''),
            'baseline_output': baseline_outputs[i],
            'finetuned_output': finetuned_outputs[i]
        })
    
    # Calculate metrics
    metrics = calculate_task_metrics(results, task_config['task_type'])
    
    task_result = {
        'task_name': task_name,
        'description': task_config['description'],
        'task_type': task_config['task_type'],
        'metrics': metrics,
        'results': results,
        'duration_seconds': baseline_duration + finetuned_duration,
        'baseline_duration_seconds': baseline_duration,
        'finetuned_duration_seconds': finetuned_duration,
    }
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    detail_file = f"{OUTPUT_DIR}/{task_name}_{TIMESTAMP}.json"
    with open(detail_file, 'w', encoding='utf-8') as f:
        json.dump(task_result, f, indent=2, ensure_ascii=False)
    print(f"      -> Saved: {detail_file}")
    
    # Print metrics
    m = metrics
    print(f"      {m['metric']}: Baseline={m['baseline']:.4f}, Finetuned={m['finetuned']:.4f}")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
