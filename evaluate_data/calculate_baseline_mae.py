#!/usr/bin/env python3
"""
Baseline 결과에서 MAE 계산 (숫자 추출 + 파싱)
단일 샘플 또는 다중 샘플 지원
"""

import json
import re
import argparse
from typing import Optional
from collections import defaultdict
import numpy as np


def extract_number(text: str) -> Optional[float]:
    """
    Extract predicted number from text (smart parsing)

    Returns None if:
    - Echo (input repetition)
    - AI refusal
    - Error message
    - Too short/empty output

    Examples:
        "Sleep_quality = 3/5" -> 3
        "The predicted stress level is 3." -> 3.0
        "0.2" -> 0.2
        "15 calories" -> 15.0
        "error 304" -> None (error message)
        "The recent 14-days..." -> None (echo)
        "As an AI, I cannot..." -> None (AI refusal)
    """
    # Remove special tokens
    text = text.replace('</s>', '').replace('<s>', '').strip()

    # Check for invalid outputs BEFORE parsing
    if not text:
        return None

    # Check for Echo (input repetition)
    if text.startswith("The recent") or text.startswith("The latest"):
        return None

    # Check for AI refusal
    text_lower = text.lower()
    ai_refusal_patterns = [
        "ai language model", "as an ai", "i am an ai",
        "cannot predict", "unable to predict", "i don't know",
        "i apologize", "sorry"
    ]
    for pattern in ai_refusal_patterns:
        if pattern in text_lower:
            return None

    # Check for error messages
    error_patterns = ["error", "exception", "failed", "invalid"]
    for pattern in error_patterns:
        if pattern in text_lower:
            return None

    # Remove common input echo patterns (to avoid extracting "14" from "14-days")
    cleaned_text = re.sub(r'The recent \d+-days? sensor readings.*?(?=predicted|answer|level|score|is|$)',
                  '', text, flags=re.DOTALL | re.IGNORECASE)

    # Check for "between X and Y" range-only responses (no specific prediction)
    # e.g., "readiness between 3 and 5." -> should be None (range, not a value)
    if re.search(r'between\s+\d+\.?\d*\s+and\s+\d+\.?\d*\s*[.!?]?\s*$', cleaned_text, re.IGNORECASE):
        return None

    # Handle "X/Y" fraction -> numerator (e.g., "2/5" -> 2, not 5)
    fraction_match = re.match(r'^\s*(\d+\.?\d*)/\d+\s*$', cleaned_text)
    if fraction_match:
        return float(fraction_match.group(1))

    # Try to find number after keywords (most reliable)
    keyword_patterns = [
        r'(?:predicted|answer|level|score|value|result|index)\s+(?:is\s+)?(\d+\.?\d*)',
        r'(\d+\.?\d*)\s+out\s+of\s+\d+',  # "7 out of 10" -> 7
        r'(?:is|:)\s+(\d+\.?\d*)',
    ]

    for pattern in keyword_patterns:
        match = re.search(pattern, cleaned_text, re.IGNORECASE)
        if match:
            return float(match.group(1))

    # Fallback: last number in remaining text
    matches = re.findall(r'[-+]?\d*\.?\d+', cleaned_text)
    if matches:
        return float(matches[-1])  # Last number (most likely the answer)

    return None


def extract_classification(text: str, task_name: str) -> Optional[str]:
    """
    Extract classification result for categorical tasks

    PMData_fatigue: 1-5 levels
    AW_FB_activity: "Running 3 METs", "Lying", etc.
    LifeSnaps_sleep_disorder: binary (0/1)
    """
    text = text.replace('</s>', '').replace('<s>', '').strip().lower()

    if task_name == 'PMData_fatigue':
        # Fatigue levels (1-5) — 범위 밖 숫자는 거부 처리
        number = extract_number(text)
        if number is not None and 1 <= number <= 5:
            level = round(number)
            return str(level)
        return None

    elif task_name == 'AW_FB_activity':
        # Extract activity type
        activities = ['running 7 mets', 'running 5 mets', 'running 3 mets',
                     'self pace walk', 'sitting', 'lying']
        for activity in activities:
            if activity in text:
                return activity.title().replace(' Mets', ' METs')
        return None

    elif task_name == 'LifeSnaps_sleep_disorder':
        # Binary classification (0/1)
        number = extract_number(text)
        if number is not None:
            return str(int(number))
        return None

    return None


def calculate_mae_single(gt: str, pred: str, task_name: str) -> tuple:
    """
    Calculate MAE for a single sample

    Returns:
        (gt_value, pred_value, mae)
    """
    # Regression tasks (MAE)
    regression_tasks = [
        'PMData_stress', 'PMData_readiness', 'PMData_sleep_quality',
        'LifeSnaps_stress_resilience', 'AW_FB_calories'
    ]

    # Classification tasks (Accuracy)
    classification_tasks = [
        'PMData_fatigue',  # 1-5 levels (논문에서 Accuracy로 측정)
        'AW_FB_activity',  # 6 types
        'LifeSnaps_sleep_disorder'  # binary 0/1
    ]

    # Valid output ranges per task (sensor data 오추출 방지)
    task_valid_ranges = {
        'PMData_stress':        (1, 5),
        'PMData_readiness':     (0, 10),
        'PMData_sleep_quality': (1, 5),
        'PMData_fatigue':       (1, 5),
        'LifeSnaps_stress_resilience': (0, 5),
        'LifeSnaps_sleep_disorder':    (0, 1),
        'AW_FB_calories':       (0, 5000),
    }

    if task_name in regression_tasks:
        # Extract ground truth number
        gt_num = extract_number(gt)
        pred_num = extract_number(pred)

        if gt_num is None or pred_num is None:
            return (gt_num, pred_num, None)

        # 유효 범위 밖이면 파싱 실패 처리 (센서값 오추출 방지)
        valid_range = task_valid_ranges.get(task_name)
        if valid_range:
            lo, hi = valid_range
            if not (lo <= pred_num <= hi):
                return (gt_num, None, None)

        mae = abs(gt_num - pred_num)
        return (gt_num, pred_num, mae)

    elif task_name in classification_tasks:
        # For classification, exact match = 1.0, mismatch = 0.0
        gt_class = extract_classification(gt, task_name) or gt.strip()
        pred_class = extract_classification(pred, task_name)

        if pred_class is None:
            return (gt_class, None, None)

        accuracy = 1.0 if gt_class.lower() == pred_class.lower() else 0.0
        return (gt_class, pred_class, accuracy)

    return (None, None, None)


def main():
    parser = argparse.ArgumentParser(description='Calculate MAE from evaluation results')
    parser.add_argument('--input', type=str, default='baseline_inferer_results.json',
                       help='Input JSON file with evaluation results')
    parser.add_argument('--task', type=str, default=None,
                       help='Evaluate specific task only (e.g., PMData_stress)')
    parser.add_argument('--mode', type=str, default='zeroshot',
                       choices=['zeroshot', 'fewshot', 'finetuned'],
                       help='Which paper baseline to compare against (default: zeroshot)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed results for each sample')
    args = parser.parse_args()

    # Load results
    print("=" * 80)
    print("BASELINE MAE CALCULATION")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Mode (paper baseline): {args.mode}")
    if args.task:
        print(f"Filtering task: {args.task}")

    with open(args.input) as f:
        results = json.load(f)

    # Filter by task if specified
    if args.task:
        results = [r for r in results if r.get('task') == args.task]
        if not results:
            print(f"❌ No results found for task: {args.task}")
            return
        print(f"Found {len(results)} samples for {args.task}")

    # Paper baselines (Table 3, MedAlpaca-7b)
    # zeroshot: MedAlpaca zero-shot
    # fewshot:  MedAlpaca few-shot
    # finetuned: HealthAlpaca-lora-7b
    paper_baselines = {
        'zeroshot': {
            'PMData_stress':        {'value': 0.76, 'std': 0.1,  'metric': 'MAE'},
            'PMData_readiness':     {'value': 2.18, 'std': 0.1,  'metric': 'MAE'},
            'PMData_sleep_quality': {'value': 0.68, 'std': 0.0,  'metric': 'MAE'},
            'PMData_fatigue':       {'value': 46.8, 'std': 11.0, 'metric': 'Acc'},
            'LifeSnaps_stress_resilience': {'value': 1.17, 'std': 0.0, 'metric': 'MAE'},
            'LifeSnaps_sleep_disorder':    {'value': 40.3, 'std': 0.0, 'metric': 'Acc'},
            'AW_FB_activity':  {'value': 21.7, 'std': 0.0, 'metric': 'Acc'},
            'AW_FB_calories':  {'value': 35.0, 'std': 0.0, 'metric': 'MAE'},
        },
        'fewshot': {
            'PMData_stress':        {'value': 0.78, 'std': 0.1,  'metric': 'MAE'},
            'PMData_readiness':     {'value': 1.94, 'std': 0.2,  'metric': 'MAE'},
            'PMData_sleep_quality': {'value': 0.69, 'std': 0.1,  'metric': 'MAE'},
            'PMData_fatigue':       {'value': 36.2, 'std': 12.0, 'metric': 'Acc'},
        },
        'finetuned': {
            'PMData_stress':        {'value': 0.53, 'std': 0.0,  'metric': 'MAE'},
            'PMData_readiness':     {'value': 1.40, 'std': 0.1,  'metric': 'MAE'},
            'PMData_sleep_quality': {'value': 0.58, 'std': 0.1,  'metric': 'MAE'},
            'PMData_fatigue':       {'value': 50.0, 'std': 13.0, 'metric': 'Acc'},
        },
    }
    paper_baseline = paper_baselines.get(args.mode, paper_baselines['zeroshot'])

    # Group by task
    task_samples = defaultdict(list)

    for result in results:
        if result.get('status') != 'SUCCESS':
            continue

        task = result['task']
        gt = result['ground_truth']
        pred = result['predicted']

        gt_val, pred_val, metric = calculate_mae_single(gt, pred, task)

        if args.verbose and len(task_samples[task]) < 3:
            print(f"\n[{task}] Sample {len(task_samples[task])}")
            print(f"  GT: {gt}")
            print(f"  Pred: {pred[:100]}{'...' if len(pred) > 100 else ''}")
            print(f"  Parsed: {gt_val} -> {pred_val} (metric={metric})")

        if metric is not None:
            task_samples[task].append({
                'gt': gt_val,
                'pred': pred_val,
                'metric': metric
            })

    # Calculate statistics per task
    print("\n" + "=" * 80)
    print("RESULTS BY TASK")
    print("=" * 80)

    task_stats = {}

    for task, samples in sorted(task_samples.items()):
        metrics = [s['metric'] for s in samples]

        if not metrics:
            print(f"\n[{task}]")
            print(f"  ❌ No valid samples")
            continue

        mean_metric = np.mean(metrics)
        std_metric = np.std(metrics)
        n_samples = len(metrics)

        task_stats[task] = {
            'mean': mean_metric,
            'std': std_metric,
            'n': n_samples
        }

        # Classification vs Regression
        # PMData_fatigue is classification (Accuracy), not regression (MAE)
        is_classification = 'fatigue' in task or 'activity' in task or 'sleep_disorder' in task

        print(f"\n[{task}]")
        print(f"  Samples: {n_samples}")

        paper_entry = paper_baseline.get(task)

        if is_classification:
            accuracy = mean_metric * 100  # Convert to percentage
            print(f"  Accuracy: {accuracy:.2f}% (±{std_metric*100:.2f}%)")

            if paper_entry and paper_entry['metric'] == 'Acc':
                paper_val = paper_entry['value']
                paper_std = paper_entry.get('std', 0)
                print(f"  Paper Acc: {paper_val}% (±{paper_std}%)")
                diff = accuracy - paper_val
                status = '✅' if diff > -5 else '⚠️' if diff > -10 else '❌'
                print(f"  Difference: {diff:+.2f}% {status}")
        else:
            print(f"  MAE: {mean_metric:.4f} (±{std_metric:.4f})")

            if paper_entry and paper_entry['metric'] == 'MAE':
                paper_val = paper_entry['value']
                paper_std = paper_entry.get('std', 0)
                print(f"  Paper MAE: {paper_val} (±{paper_std})")
                diff = mean_metric - paper_val
                status = '✅' if diff < 0.5 else '⚠️' if diff < 1.0 else '❌'
                print(f"  Difference: {diff:+.4f} {status}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # PMData_fatigue is classification (Accuracy), not regression (MAE)
    classification_tasks = [t for t in task_stats.keys()
                           if 'fatigue' in t or 'activity' in t or 'sleep_disorder' in t]
    regression_tasks = [t for t in task_stats.keys() if t not in classification_tasks]

    if regression_tasks:
        print(f"\nRegression Tasks (MAE ↓) — Paper: {args.mode}")
        print(f"  {'Task':<35} {'Our MAE':>10} {'Paper':>10} {'±':>6} {'Diff':>10} {'Status':>8}")
        print("  " + "-" * 80)
        for task in sorted(regression_tasks):
            stats = task_stats[task]
            paper_entry = paper_baseline.get(task, {})
            paper_val = paper_entry.get('value', 0)
            paper_std = paper_entry.get('std', 0)
            diff = stats['mean'] - paper_val
            status = '✅' if diff < 0.5 else '⚠️' if diff < 1.0 else '❌'
            print(f"  {task:<35} {stats['mean']:>10.4f} {paper_val:>10.2f} {paper_std:>5.1f} {diff:>+10.4f} {status:>8}")

    if classification_tasks:
        print(f"\nClassification Tasks (Accuracy ↑) — Paper: {args.mode}")
        print(f"  {'Task':<35} {'Our Acc':>10} {'Paper':>10} {'±':>6} {'Diff':>10} {'Status':>8}")
        print("  " + "-" * 80)
        for task in sorted(classification_tasks):
            stats = task_stats[task]
            paper_entry = paper_baseline.get(task, {})
            paper_val = paper_entry.get('value', 0)
            paper_std = paper_entry.get('std', 0)
            our_acc = stats['mean'] * 100
            diff = our_acc - paper_val
            status = '✅' if diff > -5 else '⚠️' if diff > -10 else '❌'
            print(f"  {task:<35} {our_acc:>9.2f}% {paper_val:>9.1f}% {paper_std:>5.1f} {diff:>+10.2f} {status:>8}")

    # Overall summary
    total_samples = sum(s['n'] for s in task_stats.values())
    print(f"\n📊 Total samples evaluated: {total_samples}")

    # Success criteria
    regression_good = sum(1 for t in regression_tasks
                         if task_stats[t]['mean'] - paper_baseline[t].get('mae', float('inf')) < 1.0)
    classification_good = sum(1 for t in classification_tasks
                             if task_stats[t]['mean'] * 100 - paper_baseline[t].get('acc', 0) > -10)

    total_tasks = len(regression_tasks) + len(classification_tasks)
    good_tasks = regression_good + classification_good

    print(f"✅ Tasks close to paper: {good_tasks}/{total_tasks}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
