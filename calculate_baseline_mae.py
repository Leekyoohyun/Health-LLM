#!/usr/bin/env python3
"""
Baseline 결과에서 MAE 계산 (숫자 추출 + 파싱)
"""

import json
import re
from typing import Optional

def extract_number(text: str) -> Optional[float]:
    """
    Extract first number from text

    Examples:
        "Sleep_quality = 3/5" -> 3
        "The predicted stress level is 3." -> 3.0
        "0.2" -> 0.2
        "15 calories" -> 15.0
        "긴 설명문" -> None
    """
    # Remove special tokens like </s><s>
    text = text.replace('</s>', '').replace('<s>', '').strip()

    # Find first number (integer or float)
    match = re.search(r'[-+]?\d*\.?\d+', text)
    return float(match.group()) if match else None


def extract_classification(text: str, task_name: str) -> Optional[str]:
    """
    Extract classification result for categorical tasks

    AW_FB_activity: "Running 3 METs", "Lying", etc.
    LifeSnaps_sleep_disorder: binary (0/1)
    """
    text = text.replace('</s>', '').replace('<s>', '').strip().lower()

    if task_name == 'AW_FB_activity':
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
    # Regression tasks
    regression_tasks = [
        'PMData_stress', 'PMData_readiness', 'PMData_sleep_quality',
        'PMData_fatigue', 'LifeSnaps_stress_resilience', 'AW_FB_calories'
    ]

    # Classification tasks
    classification_tasks = [
        'AW_FB_activity', 'LifeSnaps_sleep_disorder'
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
        # For classification, exact match = 0, mismatch = 1
        gt_class = extract_classification(gt, task_name) or gt.strip()
        pred_class = extract_classification(pred, task_name)

        if pred_class is None:
            return (gt_class, None, None)

        match = 1.0 if gt_class.lower() == pred_class.lower() else 0.0
        return (gt_class, pred_class, match)

    return (None, None, None)


def main():
    # Load results
    results_file = "baseline_inferer_results.json"
    with open(results_file) as f:
        results = json.load(f)

    print("=" * 80)
    print("BASELINE MAE CALCULATION")
    print("=" * 80)

    # Paper baseline (Table 3)
    paper_baseline = {
        'PMData_stress': 0.76,
        'PMData_readiness': 2.18,
        'PMData_sleep_quality': 0.43,
        'PMData_fatigue': None,  # Accuracy metric, not MAE
        'LifeSnaps_stress_resilience': 0.76,
        'LifeSnaps_sleep_disorder': None,  # Accuracy metric
        'AW_FB_activity': None,  # Accuracy metric
        'AW_FB_calories': 45.7,
    }

    task_metrics = {}

    for result in results:
        if result['status'] != 'SUCCESS':
            print(f"\n[{result['task']}] ❌ {result['status']}")
            continue

        task = result['task']
        gt = result['ground_truth']
        pred = result['predicted']

        gt_val, pred_val, metric = calculate_mae_single(gt, pred, task)

        print(f"\n[{task}]")
        print(f"  Ground Truth: {gt}")
        print(f"  Predicted:    {pred[:100]}{'...' if len(pred) > 100 else ''}")
        print(f"  Parsed GT:    {gt_val}")
        print(f"  Parsed Pred:  {pred_val}")

        if metric is not None:
            if 'activity' in task or 'sleep_disorder' in task:
                print(f"  Match:        {metric}")
                print(f"  Paper:        N/A (Accuracy metric)")
            else:
                print(f"  MAE:          {metric:.2f}")
                paper_mae = paper_baseline.get(task)
                if paper_mae:
                    print(f"  Paper MAE:    {paper_mae}")
                    diff = metric - paper_mae
                    print(f"  Difference:   {diff:+.2f} {'❌' if diff > 1.0 else '⚠️' if diff > 0.5 else '✅'}")

            task_metrics[task] = metric
        else:
            print(f"  ❌ Parsing failed!")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    regression_tasks = [t for t in task_metrics.keys()
                       if 'activity' not in t and 'sleep_disorder' not in t]

    if regression_tasks:
        print("\nRegression Tasks (MAE):")
        for task in regression_tasks:
            mae = task_metrics[task]
            paper_mae = paper_baseline.get(task)
            if paper_mae:
                print(f"  {task:35} {mae:6.2f} (Paper: {paper_mae:.2f})")
            else:
                print(f"  {task:35} {mae:6.2f}")

    classification_tasks = [t for t in task_metrics.keys()
                           if 'activity' in t or 'sleep_disorder' in t]

    if classification_tasks:
        print("\nClassification Tasks (Match = 1.0, Mismatch = 0.0):")
        for task in classification_tasks:
            match = task_metrics[task]
            print(f"  {task:35} {match:.1f}")

    print("\n" + "=" * 80)
    print("⚠️  Note: This is just 1 sample per task!")
    print("    For full evaluation, run on all samples in evaluation-json-data/")
    print("=" * 80)


if __name__ == "__main__":
    main()
