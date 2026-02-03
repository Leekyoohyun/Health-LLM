#!/usr/bin/env python3
"""
HealthAlpaca 모델 평가 스크립트
Baseline (MedAlpaca-7b) vs Fine-tuned (HealthAlpaca) 성능 비교

평가 내용:
- 정량 평가: MAE (숫자형), Accuracy (분류형)
- 정성 평가: 샘플별 답변 비교
"""

import os
import json
import re
import torch
import numpy as np
from datetime import datetime
from datasets import load_dataset
from medalpaca.inferer import Inferer


def load_test_data(data_path, seed=42):
    """
    Test set 로드 (학습 시와 동일한 split 재현)

    Args:
        data_path: finetune_data.json 경로
        seed: Random seed (학습 시와 동일하게 42)

    Returns:
        test_data: 213개 샘플 (학습에 사용 안 한 데이터)
    """
    print(f"   Loading data from: {data_path}")
    data = load_dataset("json", data_files=data_path)

    split = data["train"].train_test_split(
        test_size=0.1,
        shuffle=True,
        seed=seed
    )

    test_data = split["test"]
    print(f"   ✓ Test samples loaded: {len(test_data)}")
    return test_data


def load_baseline_model():
    """
    Baseline 모델 로드 (MedAlpaca-7b)

    Returns:
        Inferer 객체
    """
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
    """
    기존 모델에 LoRA 어댑터 결합

    Args:
        inferer: Baseline Inferer 객체
        adapter_path: LoRA 어댑터 경로
            - adapter_model.bin (pytorch_model.bin에서 이름 변경)
            - adapter_config.json

    Returns:
        LoRA가 결합된 모델
    """
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
    """
    텍스트에서 숫자 추출 (정규식 사용)

    Args:
        text: 입력 텍스트

    Returns:
        float 또는 None
    """
    # 먼저 전체 텍스트를 숫자로 변환 시도
    try:
        return float(text.strip())
    except:
        pass

    # 정규식으로 숫자 찾기 (소수점 포함)
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[0])
        except:
            pass

    return None


def normalize_text(text):
    """
    텍스트 정규화 (소문자 변환, 공백 제거)

    Args:
        text: 입력 텍스트

    Returns:
        정규화된 텍스트
    """
    return text.strip().lower()


def extract_classification_answer(text):
    """
    분류 답변에서 핵심 키워드 추출

    예: "The predicted activity type is Lying" -> "lying"

    Args:
        text: 입력 텍스트

    Returns:
        추출된 키워드
    """
    # "is XXX" 패턴 찾기
    match = re.search(r'is\s+(\w+)', text, re.IGNORECASE)
    if match:
        return normalize_text(match.group(1))

    # 패턴이 없으면 전체 텍스트 정규화
    return normalize_text(text)


def is_pure_numeric(text):
    """
    텍스트가 순수 숫자인지 판별 (숫자만 포함되어 있는지)

    Args:
        text: 입력 텍스트

    Returns:
        True if 순수 숫자, False otherwise
    """
    try:
        float(text.strip())
        return True
    except:
        return False


def calculate_metrics(results):
    """
    평가 지표 계산 (MAE, Accuracy)

    Args:
        results: run_inference()의 결과

    Returns:
        metrics: 계산된 지표들
    """
    # 숫자형 예측용
    baseline_errors = []
    finetuned_errors = []

    # 분류 예측용
    baseline_correct_classification = 0
    finetuned_correct_classification = 0
    total_classification = 0

    # 전체 정확도
    baseline_correct_total = 0
    finetuned_correct_total = 0

    for r in results:
        gt = r['ground_truth']
        baseline_pred = r['baseline_output']
        finetuned_pred = r['finetuned_output']

        # Ground truth가 순수 숫자인지 판별 (중요!)
        # "1.22" → True (숫자형)
        # "Running 3 METs" → False (텍스트형, 숫자 포함되어 있어도)
        if is_pure_numeric(gt):
            # 숫자형 예측 (MAE 계산)
            gt_num = float(gt.strip())
            baseline_num = extract_number(baseline_pred)
            finetuned_num = extract_number(finetuned_pred)

            if baseline_num is not None:
                baseline_errors.append(abs(gt_num - baseline_num))
                # 숫자형에서는 근사 일치 (오차 0.1 이하)를 정확으로 간주
                if abs(gt_num - baseline_num) < 0.1:
                    baseline_correct_total += 1

            if finetuned_num is not None:
                finetuned_errors.append(abs(gt_num - finetuned_num))
                if abs(gt_num - finetuned_num) < 0.1:
                    finetuned_correct_total += 1
        else:
            # 텍스트형 예측 (Accuracy 계산)
            total_classification += 1

            gt_normalized = extract_classification_answer(gt)
            baseline_normalized = extract_classification_answer(baseline_pred)
            finetuned_normalized = extract_classification_answer(finetuned_pred)

            if baseline_normalized == gt_normalized:
                baseline_correct_classification += 1
                baseline_correct_total += 1

            if finetuned_normalized == gt_normalized:
                finetuned_correct_classification += 1
                finetuned_correct_total += 1

    # MAE 계산 (숫자형만)
    baseline_mae = np.mean(baseline_errors) if baseline_errors else None
    finetuned_mae = np.mean(finetuned_errors) if finetuned_errors else None

    # Accuracy 계산 (분류형만)
    baseline_acc_classification = baseline_correct_classification / total_classification if total_classification > 0 else None
    finetuned_acc_classification = finetuned_correct_classification / total_classification if total_classification > 0 else None

    # 전체 정확도
    total_samples = len(results)
    baseline_acc_total = baseline_correct_total / total_samples if total_samples > 0 else 0
    finetuned_acc_total = finetuned_correct_total / total_samples if total_samples > 0 else 0

    return {
        'baseline_mae': float(baseline_mae) if baseline_mae is not None else None,
        'finetuned_mae': float(finetuned_mae) if finetuned_mae is not None else None,
        'baseline_accuracy_classification': float(baseline_acc_classification) if baseline_acc_classification is not None else None,
        'finetuned_accuracy_classification': float(finetuned_acc_classification) if finetuned_acc_classification is not None else None,
        'baseline_accuracy_total': float(baseline_acc_total),
        'finetuned_accuracy_total': float(finetuned_acc_total),
        'total_samples': total_samples,
        'numeric_samples': len(baseline_errors),
        'classification_samples': total_classification
    }


def save_results(results, metrics, output_dir="results"):
    """
    결과 저장

    Args:
        results: 추론 결과
        metrics: 평가 지표
        output_dir: 저장 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 전체 결과 저장
    results_file = f"{output_dir}/evaluation_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'metrics': metrics,
            'samples': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {results_file}")

    # 정성 평가용 샘플 저장 (처음 5개)
    case_study_file = f"{output_dir}/case_study_{timestamp}.json"
    with open(case_study_file, 'w', encoding='utf-8') as f:
        json.dump(results[:5], f, indent=2, ensure_ascii=False)

    print(f"✓ Case study samples saved to: {case_study_file}")


def print_metrics(metrics):
    """평가 지표 출력"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    print(f"\nTotal samples evaluated: {metrics['total_samples']}")
    print(f"  - Numeric samples (MAE): {metrics['numeric_samples']}")
    print(f"  - Classification samples (Accuracy): {metrics['classification_samples']}")

    if metrics['baseline_mae'] is not None:
        print(f"\n[MAE (Mean Absolute Error) - Numeric Predictions Only]")
        print(f"  Baseline:    {metrics['baseline_mae']:.4f}")
        print(f"  Fine-tuned:  {metrics['finetuned_mae']:.4f}")
        improvement = (metrics['baseline_mae'] - metrics['finetuned_mae']) / metrics['baseline_mae'] * 100
        print(f"  Improvement: {improvement:+.2f}%")

    if metrics['baseline_accuracy_classification'] is not None:
        print(f"\n[Accuracy - Classification Tasks Only]")
        print(f"  Baseline:    {metrics['baseline_accuracy_classification']:.2%}")
        print(f"  Fine-tuned:  {metrics['finetuned_accuracy_classification']:.2%}")
        improvement = (metrics['finetuned_accuracy_classification'] - metrics['baseline_accuracy_classification']) / max(metrics['baseline_accuracy_classification'], 0.001) * 100
        print(f"  Improvement: {improvement:+.2f}%")

    print(f"\n[Overall Accuracy - All Tasks]")
    print(f"  Baseline:    {metrics['baseline_accuracy_total']:.2%}")
    print(f"  Fine-tuned:  {metrics['finetuned_accuracy_total']:.2%}")
    improvement = (metrics['finetuned_accuracy_total'] - metrics['baseline_accuracy_total']) / max(metrics['baseline_accuracy_total'], 0.001) * 100
    print(f"  Improvement: {improvement:+.2f}%")

    print("\n" + "="*80)


def run_inference_single(model, test_data, num_samples=10):
    """
    단일 모델로 추론 수행

    Args:
        model: Inferer 객체
        test_data: Test set
        num_samples: 평가할 샘플 수

    Returns:
        outputs: 각 샘플별 출력 리스트
    """
    outputs = []

    print(f"   Evaluating {num_samples} samples...")

    for i, sample in enumerate(test_data[:num_samples]):
        print(f"   [{i+1}/{num_samples}] Processing...", end='\r')

        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')

        output = model(
            instruction=instruction if instruction else None,
            input=input_text,
            max_new_tokens=128
        )

        outputs.append(output)

    print(f"   ✓ {num_samples} samples evaluated")
    return outputs


def main():
    """메인 실행 함수"""
    print("="*80)
    print("HealthAlpaca Model Evaluation")
    print("Baseline (MedAlpaca-7b) vs Fine-tuned (HealthAlpaca)")
    print("="*80)

    # 1. Test 데이터 로드
    print("\n[1/4] Loading test data...")
    test_data = load_test_data("data/finetune_data.json")
    num_samples = 10  # 필요시 변경 (전체: 213)

    # 2. Baseline 모델 로드 및 평가
    print("\n[2/4] Loading Baseline model and running inference...")
    model = load_baseline_model()
    baseline_outputs = run_inference_single(model, test_data, num_samples)

    # 3. LoRA 어댑터 결합 후 Fine-tuned 평가
    print("\n[3/4] Attaching LoRA adapter and running inference...")
    model = attach_lora_adapter(model)
    finetuned_outputs = run_inference_single(model, test_data, num_samples)

    # 4. 결과 병합 및 평가
    print("\n[4/4] Calculating metrics...")
    results = []
    for i, sample in enumerate(test_data[:num_samples]):
        results.append({
            'sample_id': i + 1,
            'instruction': sample.get('instruction', ''),
            'input': sample.get('input', ''),
            'ground_truth': sample.get('output', ''),
            'baseline_output': baseline_outputs[i],
            'finetuned_output': finetuned_outputs[i]
        })

    metrics = calculate_metrics(results)

    # 결과 저장 및 출력
    save_results(results, metrics)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
