#!/usr/bin/env python3
"""
HealthAlpaca 모델 평가 스크립트 (태스크별 개별 평가)
Baseline (MedAlpaca-7b) vs Fine-tuned (HealthAlpaca) 성능 비교

평가 내용:
- Regression 태스크: MAE (Mean Absolute Error)
- Classification 태스크: Accuracy
"""

import os
import json
import re
import torch
import numpy as np
from datetime import datetime
from datasets import load_dataset
from medalpaca.inferer import Inferer


# ============================================================================
# 태스크별 설정 (논문 Table 16 기준)
# ============================================================================
TASKS = {
    # PMData 태스크들
    "PMData_stress": {
        "data_path": "PMData_stress_train_all.json",
        "task_type": "regression",  # MAE
        "description": "Stress Prediction (1-5)"
    },
    "PMData_readiness": {
        "data_path": "PMData_readiness_train_all.json",
        "task_type": "regression",  # MAE
        "description": "Readiness Prediction (0-10)"
    },
    "PMData_sleep_quality": {
        "data_path": "PMData_sleep_quality_train_all.json",
        "task_type": "regression",  # MAE
        "description": "Sleep Quality Prediction (1-5)"
    },
    "PMData_fatigue": {
        "data_path": "PMData_fatigue_train_all.json",
        "task_type": "classification",  # Accuracy
        "description": "Fatigue Prediction (1-5)"
    },

    # AW_FB 태스크들
    "AW_FB_activity": {
        "data_path": "AW_FB_activity_train_all.json",
        "task_type": "classification",  # Accuracy
        "description": "Activity Recognition"
    },
    "AW_FB_calories": {
        "data_path": "AW_FB_calories_train_all.json",
        "task_type": "regression",  # MAE
        "description": "Calorie Burn Estimation"
    },

    # LifeSnaps 태스크들
    "LifeSnaps_stress_resilience": {
        "data_path": "LifeSnaps_stress_resilience_train_all.json",
        "task_type": "regression",  # MAE
        "description": "Stress Resilience (0.2-5)"
    },
    "LifeSnaps_sleep_disorder": {
        "data_path": "LifeSnaps_sleep_disorder_train_all.json",
        "task_type": "classification",  # Accuracy
        "description": "Sleep Disorder Detection (0/1)"
    },

    # GLOBEM 태스크들
    "GLOBEM_depression": {
        "data_path": "GLOBEM_depression_train_all.json",
        "task_type": "regression",  # MAE
        "description": "PHQ-4 Depression (0-4)"
    },
    "GLOBEM_anxiety": {
        "data_path": "GLOBEM_anxiety_train_all.json",
        "task_type": "regression",  # MAE
        "description": "PHQ-4 Anxiety (0-4)"
    },
}


def load_test_data(data_path, seed=42):
    """
    Test set 로드 (90:10 split)
    """
    print(f"      Loading: {data_path}")

    if not os.path.exists(data_path):
        print(f"      ⚠ File not found: {data_path}")
        return None

    data = load_dataset("json", data_files=data_path)

    split = data["train"].train_test_split(
        test_size=0.1,
        shuffle=True,
        seed=seed
    )

    test_data = split["test"]
    print(f"      ✓ Test samples: {len(test_data)}")
    return test_data


def load_baseline_model():
    """Baseline 모델 로드 (MedAlpaca-7b)"""
    print("   Loading Baseline (MedAlpaca-7b)...")
    model = Inferer(
        model_name="medalpaca/medalpaca-7b",
        prompt_template="medalpaca/prompt_templates/medalpaca.json",
        model_max_length=256,
        torch_dtype=torch.float16
    )
    print("   ✓ Baseline loaded")
    return model


def attach_lora_adapter(inferer, adapter_path="outputs/healthalpaca-7b-lora"):
    """기존 모델에 LoRA 어댑터 결합"""
    from peft import PeftModel

    print(f"   Attaching LoRA adapter from {adapter_path}...")
    inferer.model = PeftModel.from_pretrained(
        inferer.model,
        adapter_path,
        torch_dtype=torch.float16
    )
    inferer.model.eval()
    print("   ✓ LoRA adapter attached")
    return inferer


def extract_number(text):
    """텍스트에서 숫자 추출"""
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


def extract_classification_answer(text):
    """분류 답변에서 핵심 키워드 추출"""
    # "is XXX" 패턴 찾기
    match = re.search(r'is\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return text.strip().lower()


def run_inference(model, test_data, max_samples=None):
    """모델 추론 수행"""
    outputs = []
    samples = test_data if max_samples is None else test_data[:max_samples]
    total = len(samples)

    for i, sample in enumerate(samples):
        print(f"      [{i+1}/{total}] Processing...", end='\r')

        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')

        output = model(
            instruction=instruction if instruction else None,
            input=input_text,
            max_new_tokens=128
        )
        outputs.append(output)

    print(f"      ✓ {total} samples evaluated          ")
    return outputs


def calculate_task_metrics(results, task_type):
    """
    태스크 타입별 메트릭 계산

    Args:
        results: 추론 결과 리스트
        task_type: "regression" (MAE) 또는 "classification" (Accuracy)
    """
    baseline_scores = []
    finetuned_scores = []

    for r in results:
        gt = r['ground_truth']
        baseline_pred = r['baseline_output']
        finetuned_pred = r['finetuned_output']

        if task_type == "regression":
            # MAE 계산
            gt_num = extract_number(gt)
            baseline_num = extract_number(baseline_pred)
            finetuned_num = extract_number(finetuned_pred)

            if gt_num is not None:
                if baseline_num is not None:
                    baseline_scores.append(abs(gt_num - baseline_num))
                else:
                    baseline_scores.append(abs(gt_num))  # 실패시 큰 에러

                if finetuned_num is not None:
                    finetuned_scores.append(abs(gt_num - finetuned_num))
                else:
                    finetuned_scores.append(abs(gt_num))

        else:  # classification
            # Accuracy 계산
            gt_label = extract_classification_answer(gt)
            baseline_label = extract_classification_answer(baseline_pred)
            finetuned_label = extract_classification_answer(finetuned_pred)

            baseline_scores.append(1 if gt_label == baseline_label else 0)
            finetuned_scores.append(1 if gt_label == finetuned_label else 0)

    if task_type == "regression":
        return {
            'metric': 'MAE',
            'baseline': np.mean(baseline_scores) if baseline_scores else None,
            'finetuned': np.mean(finetuned_scores) if finetuned_scores else None,
            'num_samples': len(baseline_scores)
        }
    else:
        return {
            'metric': 'Accuracy',
            'baseline': np.mean(baseline_scores) if baseline_scores else None,
            'finetuned': np.mean(finetuned_scores) if finetuned_scores else None,
            'num_samples': len(baseline_scores)
        }


def evaluate_single_task(task_name, task_config, baseline_model, finetuned_model, max_samples=None):
    """단일 태스크 평가"""
    print(f"\n   [{task_name}] {task_config['description']}")
    print(f"      Task type: {task_config['task_type']}")

    # 데이터 로드
    test_data = load_test_data(task_config['data_path'])
    if test_data is None:
        return None

    # Baseline 추론
    print("      Running Baseline inference...")
    baseline_outputs = run_inference(baseline_model, test_data, max_samples)

    # Fine-tuned 추론
    print("      Running Fine-tuned inference...")
    finetuned_outputs = run_inference(finetuned_model, test_data, max_samples)

    # 결과 병합
    samples = test_data if max_samples is None else test_data[:max_samples]
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

    # 메트릭 계산
    metrics = calculate_task_metrics(results, task_config['task_type'])

    return {
        'task_name': task_name,
        'description': task_config['description'],
        'task_type': task_config['task_type'],
        'metrics': metrics,
        'results': results
    }


def print_summary(all_results):
    """전체 결과 요약 출력"""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    # 회귀 태스크 (MAE)
    print("\n[Regression Tasks - MAE (↓ lower is better)]")
    print("-" * 70)
    print(f"{'Task':<35} {'Baseline':<12} {'Fine-tuned':<12} {'Improve':<10}")
    print("-" * 70)

    for r in all_results:
        if r is None:
            continue
        if r['task_type'] == 'regression':
            baseline = r['metrics']['baseline']
            finetuned = r['metrics']['finetuned']
            if baseline and finetuned:
                improve = (baseline - finetuned) / baseline * 100
                print(f"{r['task_name']:<35} {baseline:<12.4f} {finetuned:<12.4f} {improve:+.1f}%")
            else:
                print(f"{r['task_name']:<35} {'N/A':<12} {'N/A':<12} {'N/A':<10}")

    # 분류 태스크 (Accuracy)
    print("\n[Classification Tasks - Accuracy (↑ higher is better)]")
    print("-" * 70)
    print(f"{'Task':<35} {'Baseline':<12} {'Fine-tuned':<12} {'Improve':<10}")
    print("-" * 70)

    for r in all_results:
        if r is None:
            continue
        if r['task_type'] == 'classification':
            baseline = r['metrics']['baseline']
            finetuned = r['metrics']['finetuned']
            if baseline and finetuned:
                improve = (finetuned - baseline) / max(baseline, 0.001) * 100
                print(f"{r['task_name']:<35} {baseline:<12.2%} {finetuned:<12.2%} {improve:+.1f}%")
            else:
                print(f"{r['task_name']:<35} {'N/A':<12} {'N/A':<12} {'N/A':<10}")

    print("\n" + "="*80)


def save_all_results(all_results, output_dir="results"):
    """전체 결과 저장"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 요약 결과 저장
    summary = []
    for r in all_results:
        if r is None:
            continue
        summary.append({
            'task_name': r['task_name'],
            'description': r['description'],
            'task_type': r['task_type'],
            'metric': r['metrics']['metric'],
            'baseline': r['metrics']['baseline'],
            'finetuned': r['metrics']['finetuned'],
            'num_samples': r['metrics']['num_samples']
        })

    summary_file = f"{output_dir}/evaluation_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'tasks': summary
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Summary saved to: {summary_file}")

    # 상세 결과 저장 (각 태스크별)
    for r in all_results:
        if r is None:
            continue
        detail_file = f"{output_dir}/{r['task_name']}_{timestamp}.json"
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump(r, f, indent=2, ensure_ascii=False)

    print(f"✓ Detailed results saved to: {output_dir}/")


def main():
    """메인 실행 함수"""
    print("="*80)
    print("HealthAlpaca Model Evaluation (Task-wise)")
    print("Baseline (MedAlpaca-7b) vs Fine-tuned (HealthAlpaca)")
    print("="*80)

    # 설정
    MAX_SAMPLES_PER_TASK = None  # None이면 전체, 숫자면 해당 개수만

    # 1. 모델 로드
    print("\n[Step 1] Loading models...")
    baseline_model = load_baseline_model()
    finetuned_model = attach_lora_adapter(baseline_model)

    # 2. 태스크별 평가
    print("\n[Step 2] Evaluating tasks...")
    all_results = []

    for task_name, task_config in TASKS.items():
        result = evaluate_single_task(
            task_name,
            task_config,
            baseline_model,
            finetuned_model,
            max_samples=MAX_SAMPLES_PER_TASK
        )
        all_results.append(result)

    # 3. 결과 출력 및 저장
    print("\n[Step 3] Saving results...")
    print_summary(all_results)
    save_all_results(all_results)


if __name__ == "__main__":
    main()
