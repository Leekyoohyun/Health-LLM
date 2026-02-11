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

    Examples:
        "Sleep_quality = 3/5" -> 3
        "The predicted stress level is 3." -> 3.0
        "0.2" -> 0.2
        "15 calories" -> 15.0
        "The recent 14-days sensor readings show: ... predicted stress is 3" -> 3
    """
    # Remove special tokens
    text = text.replace('</s>', '').replace('<s>', '').strip()

    # Remove common input echo patterns (to avoid extracting "14" from "14-days")
    text = re.sub(r'The recent \d+-days? sensor readings.*?(?=predicted|answer|level|score|is|$)',
                  '', text, flags=re.DOTALL | re.IGNORECASE)

    # Try to find number after keywords (most reliable)
    keyword_patterns = [
        r'(?:predicted|answer|level|score|value|result|index)\s+(?:is\s+)?(\d+\.?\d*)',
        r'(?:is|:)\s+(\d+\.?\d*)',
    ]

    for pattern in keyword_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))

    # Fallback: last number in remaining text
    matches = re.findall(r'[-+]?\d*\.?\d+', text)
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
        # Fatigue levels (1-5)
        number = extract_number(text)
        if number is not None:
            # Round to nearest integer and clamp to 1-5
            level = max(1, min(5, round(number)))
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

    if task_name in regression_tasks:
        # Extract ground truth number
        gt_num = extract_number(gt)
        pred_num = extract_number(pred)

        if gt_num is None or pred_num is None:
            return (gt_num, pred_num, None)

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
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed results for each sample')
    args = parser.parse_args()

    # Load results
    print("=" * 80)
    print("BASELINE MAE CALCULATION")
    print("=" * 80)
    print(f"Input file: {args.input}")
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

    # Paper baseline (Table 3 from paper image)
    paper_baseline = {
        'PMData_stress': {'mae': 0.76, 'metric': 'MAE'},
        'PMData_readiness': {'mae': 2.18, 'metric': 'MAE'},
        'PMData_sleep_quality': {'mae': 0.68, 'metric': 'MAE'},  # SQ from paper
        'PMData_fatigue': {'acc': 46.8, 'metric': 'Acc'},  # FATG from paper
        'LifeSnaps_stress_resilience': {'mae': 1.17, 'metric': 'MAE'},  # SR from paper
        'LifeSnaps_sleep_disorder': {'acc': 40.3, 'metric': 'Acc'},  # SD from paper
        'AW_FB_activity': {'acc': 21.7, 'metric': 'Acc'},  # ACT from paper
        'AW_FB_calories': {'mae': 35.0, 'metric': 'MAE'},  # CAL from paper
    }

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
        is_classification = 'activity' in task or 'sleep_disorder' in task

        print(f"\n[{task}]")
        print(f"  Samples: {n_samples}")

        if is_classification:
            accuracy = mean_metric * 100  # Convert to percentage
            print(f"  Accuracy: {accuracy:.2f}% (±{std_metric*100:.2f}%)")

            paper_acc = paper_baseline[task].get('acc')
            if paper_acc:
                print(f"  Paper Acc: {paper_acc}%")
                diff = accuracy - paper_acc
                status = '✅' if diff > -5 else '⚠️' if diff > -10 else '❌'
                print(f"  Difference: {diff:+.2f}% {status}")
        else:
            print(f"  MAE: {mean_metric:.2f} (±{std_metric:.2f})")

            paper_mae = paper_baseline[task].get('mae')
            if paper_mae:
                print(f"  Paper MAE: {paper_mae}")
                diff = mean_metric - paper_mae
                status = '✅' if diff < 0.5 else '⚠️' if diff < 1.0 else '❌'
                print(f"  Difference: {diff:+.2f} {status}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    regression_tasks = [t for t in task_stats.keys()
                       if 'activity' not in t and 'sleep_disorder' not in t]
    classification_tasks = [t for t in task_stats.keys()
                           if 'activity' in t or 'sleep_disorder' in t]

    if regression_tasks:
        print("\nRegression Tasks (MAE ↓):")
        print(f"  {'Task':<35} {'Our MAE':>10} {'Paper':>10} {'Diff':>10} {'Status':>8}")
        print("  " + "-" * 75)
        for task in sorted(regression_tasks):
            stats = task_stats[task]
            paper_mae = paper_baseline[task].get('mae', 0)
            diff = stats['mean'] - paper_mae
            status = '✅' if diff < 0.5 else '⚠️' if diff < 1.0 else '❌'
            print(f"  {task:<35} {stats['mean']:>10.2f} {paper_mae:>10.2f} {diff:>+10.2f} {status:>8}")

    if classification_tasks:
        print("\nClassification Tasks (Accuracy ↑):")
        print(f"  {'Task':<35} {'Our Acc':>10} {'Paper':>10} {'Diff':>10} {'Status':>8}")
        print("  " + "-" * 75)
        for task in sorted(classification_tasks):
            stats = task_stats[task]
            paper_acc = paper_baseline[task].get('acc', 0)
            our_acc = stats['mean'] * 100
            diff = our_acc - paper_acc
            status = '✅' if diff > -5 else '⚠️' if diff > -10 else '❌'
            print(f"  {task:<35} {our_acc:>9.2f}% {paper_acc:>9.1f}% {diff:>+10.2f} {status:>8}")

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
