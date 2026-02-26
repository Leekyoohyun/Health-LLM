#!/usr/bin/env python3
"""
13B 평가 결과 MAE/Accuracy 계산

Task별 분리된 JSON 또는 통합 JSON 모두 지원.

사용법:
  # 통합 파일 (전체 task 한번에)
  python calculate_mae_13b.py --input results_13b/results_13b_zeroshot_half.json

  # Task별 파일
  python calculate_mae_13b.py --input results_13b/PMData_stress_zeroshot.json

  # 특정 task만
  python calculate_mae_13b.py --input results_13b/results_13b_zeroshot_half.json --task PMData_stress

  # 상세 출력
  python calculate_mae_13b.py --input results_13b/results_13b_zeroshot_half.json --verbose
"""

import json
import re
import argparse
from typing import Optional
from collections import defaultdict

# ── Task 정의 ──
# metric: "mae" (regression) 또는 "acc" (classification)
# valid_range: 파싱된 숫자의 유효 범위
# paper_zeroshot: 논문 Table 3 ZeroShot baseline
TASK_META = {
    # ── PMData ──
    "PMData_stress": {
        "metric": "mae",
        "valid_range": (1, 5),
        "paper_zeroshot": 0.76,   # ± 0.1
        "paper_fewshot": 0.78,    # ± 0.1
        "paper_finetuned": 0.21,  # ± 0.0 (13B)
    },
    "PMData_readiness": {
        "metric": "mae",
        "valid_range": (0, 10),
        "paper_zeroshot": 2.18,   # ± 0.1
        "paper_fewshot": 1.94,    # ± 0.1
        "paper_finetuned": 1.08,  # ± 0.2 (13B)
    },
    "PMData_sleep_quality": {
        "metric": "mae",
        "valid_range": (1, 5),
        "paper_zeroshot": 0.68,   # ± 0.0
        "paper_fewshot": 0.69,    # ± 0.1
        "paper_finetuned": 0.14,  # ± 0.0 (13B)
    },
    "PMData_fatigue": {
        "metric": "acc",
        "valid_range": (1, 5),
        "paper_zeroshot": 46.8,   # ± 11
        "paper_fewshot": 36.2,    # ± 11
        "paper_finetuned": 61.2,  # ± 3.4 (13B)
    },
    # ── LifeSnaps ──
    "LifeSnaps_stress_resilience": {
        "metric": "mae",
        "valid_range": (0, 5),
        "paper_zeroshot": 1.17,   # ± 0.1
        "paper_fewshot": 0.94,    # ± 0.2
        "paper_finetuned": 0.32,  # ± 0.1 (13B)
    },
    "LifeSnaps_sleep_disorder": {
        "metric": "acc",
        "valid_labels": ["0", "1"],
        "paper_zeroshot": 40.3,   # ± 1.6
        "paper_fewshot": 49.6,    # ± 11
        "paper_finetuned": 93.9,  # ± 3.1 (13B)
    },
    # ── AW_FB ──
    "AW_FB_activity": {
        "metric": "acc",
        "valid_labels": [
            "Running 7 METs", "Running 5 METs", "Running 3 METs",
            "Self Pace walk", "Sitting", "Lying",
        ],
        "paper_zeroshot": 21.7,   # ± 4.4
        "paper_fewshot": 19.3,    # ± 8.1
        "paper_finetuned": 51.0,  # ± 3.5 (13B)
    },
    "AW_FB_calories": {
        "metric": "mae",
        "valid_range": (0, 500),
        "paper_zeroshot": 35.0,   # ± 6.0
        "paper_fewshot": 36.7,    # ± 5.6
        "paper_finetuned": 28.5,  # ± 5.6 (13B)
    },
}


# ═══════════════════════════════════════════════════════════════
# 파싱 함수
# ═══════════════════════════════════════════════════════════════

def extract_number(text: str, task_name: str = "") -> Optional[float]:
    """
    모델 응답에서 숫자 추출 (regression + fatigue 공통)

    Returns None:
    - Echo (입력 반복)
    - AI 거부
    - 에러 메시지
    - 유효 범위 밖
    """
    if text is None:
        return None

    text = text.replace('</s>', '').replace('<s>', '').strip()

    if not text or len(text) < 1:
        return None

    text_lower = text.lower()

    # ── Echo 체크 (튜플) ──
    echo_starts = (
        "the recent", "the latest", "the user is",
        "stress level is an indicator",
        "readiness score is an indicator",
        "sleep quality is an indicator",
        "fatigue level is an indicator",
        "activity recognition involves",
        "calorie burn estimate",
        "stress resilience index",
        "sleep disorder prediction",
        "[example ",
    )
    for prefix in echo_starts:
        if text_lower.startswith(prefix):
            return None

    # ── Echo 체크 (regex) ──
    if re.match(r'the user is \d+-year-old', text_lower):
        return None
    if re.match(r'```', text):
        return None

    # ── AI 거부 ──
    ai_refusal = [
        "as an ai", "ai language model", "i am an ai",
        "cannot predict", "unable to predict",
        "i don't know", "i cannot",
    ]
    for p in ai_refusal:
        if p in text_lower:
            return None

    # ── soft refusal (숫자 없을 때만) ──
    soft_refusal = ["sorry", "apologize", "apologi"]
    if not re.search(r'\d', text):
        for p in soft_refusal:
            if p in text_lower:
                return None

    # ── 에러 패턴 ──
    error_patterns = ["error", "exception", "traceback"]
    for p in error_patterns:
        if p in text_lower:
            return None

    # ── 입력 에코 제거 ──
    cleaned = re.sub(
        r'The recent \d+-days? sensor readings.*?(?=predicted|answer|level|score|is|$)',
        '', text, flags=re.DOTALL | re.IGNORECASE
    )

    # ── "between X and/to Y" 끝 응답 → None ──
    if re.search(r'between\s+[\d.]+\s+(?:and|to)\s+[\d.]+\s*[.!?]?\s*$', cleaned, re.IGNORECASE):
        return None

    # ── 시간 단위 숫자 제거 (fallback 오추출 방지) ──
    cleaned = re.sub(r'\d+\s*(?:weeks?|days?|hours?|minutes?|months?|years?)', '', cleaned, flags=re.IGNORECASE)

    # ── inline "between X and/to Y" 제거 ──
    cleaned = re.sub(r'between\s+[\d.]+\s+(?:and|to)\s+[\d.]+', '', cleaned, flags=re.IGNORECASE)

    # ── "X out of Y" → X ──
    m = re.search(r'(\d+\.?\d*)\s+out\s+of\s+\d+', cleaned, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # ── "X/Y" 분수 → X ──
    m = re.search(r'(\d+\.?\d*)\s*/\s*\d+', cleaned)
    if m:
        return float(m.group(1))

    # ── Keyword 패턴 ──
    keyword_patterns = [
        r'(?:answer|predicted|prediction)\s*(?:is\s*)?[:\s]+(\d+\.?\d*)',
        r'(?:level|score|value|result|index)\s+(?:is\s+|of\s+)?(\d+\.?\d*)',
        r'(?:is|:|=)\s+(\d+\.?\d*)',
    ]
    for pat in keyword_patterns:
        m = re.search(pat, cleaned, re.IGNORECASE)
        if m:
            return float(m.group(1))

    # ── Fallback: 마지막 숫자 ──
    matches = re.findall(r'[-+]?\d*\.?\d+', cleaned)
    if matches:
        return float(matches[-1])

    return None


def extract_activity(text: str) -> Optional[str]:
    """AW_FB_activity 응답에서 활동 유형 추출"""
    if text is None:
        return None

    text_lower = text.lower().replace('</s>', '').replace('<s>', '').strip()

    # 정확한 매칭 (우선순위: 구체적 → 일반적)
    activity_map = {
        "running 7 mets": "Running 7 METs",
        "running 5 mets": "Running 5 METs",
        "running 3 mets": "Running 3 METs",
        "self pace walk": "Self Pace walk",
        "self-pace walk": "Self Pace walk",
        "walking": "Self Pace walk",
        "sitting": "Sitting",
        "lying": "Lying",
        "lying down": "Lying",
    }

    for key, val in activity_map.items():
        if key in text_lower:
            return val

    # "running" 만 있을 때 (METs 없음) → 매칭 불가
    if "running" in text_lower:
        # METs 숫자 찾기
        m = re.search(r'running\s*(\d)\s*mets?', text_lower)
        if m:
            mets = m.group(1)
            return f"Running {mets} METs"
        return None

    return None


def extract_sleep_disorder(text: str) -> Optional[str]:
    """LifeSnaps_sleep_disorder 응답에서 0/1 추출"""
    if text is None:
        return None

    text = text.replace('</s>', '').replace('<s>', '').strip().lower()

    # "no sleep disorder" → 0, "sleep disorder" → 1
    if "no sleep disorder" in text or "no disorder" in text:
        return "0"
    if "sleep disorder" in text and "no" not in text:
        return "1"

    # 숫자로 추출
    num = extract_number(text, "LifeSnaps_sleep_disorder")
    if num is not None:
        return str(int(round(num)))

    return None


# ═══════════════════════════════════════════════════════════════
# 평가 함수
# ═══════════════════════════════════════════════════════════════

def evaluate_sample(task_name: str, ground_truth: str, predicted: str) -> dict:
    """
    단일 샘플 평가

    Returns:
        {
            'gt_parsed': parsed ground truth,
            'pred_parsed': parsed prediction,
            'valid': True/False (파싱 성공 여부),
            'metric_value': MAE 또는 1/0 (accuracy),
        }
    """
    meta = TASK_META.get(task_name, {})
    metric_type = meta.get("metric", "mae")

    result = {'gt_parsed': None, 'pred_parsed': None, 'valid': False, 'metric_value': None}

    if task_name == "AW_FB_activity":
        # ── Activity: 문자열 분류 ──
        gt_label = ground_truth.strip()
        pred_label = extract_activity(predicted)

        result['gt_parsed'] = gt_label
        result['pred_parsed'] = pred_label

        if pred_label is not None:
            result['valid'] = True
            result['metric_value'] = 1.0 if gt_label == pred_label else 0.0

    elif task_name == "LifeSnaps_sleep_disorder":
        # ── Sleep disorder: 이진 분류 ──
        gt_label = ground_truth.strip()
        pred_label = extract_sleep_disorder(predicted)

        result['gt_parsed'] = gt_label
        result['pred_parsed'] = pred_label

        if pred_label is not None:
            result['valid'] = True
            result['metric_value'] = 1.0 if gt_label == pred_label else 0.0

    elif metric_type == "acc":
        # ── PMData_fatigue: 정수 분류 ──
        gt_num = float(ground_truth.strip())
        pred_num = extract_number(predicted, task_name)

        vmin, vmax = meta.get("valid_range", (float('-inf'), float('inf')))

        result['gt_parsed'] = int(gt_num)

        if pred_num is not None:
            pred_int = max(int(vmin), min(int(vmax), round(pred_num)))
            result['pred_parsed'] = pred_int

            if vmin <= pred_num <= vmax * 1.5:  # 약간의 여유
                result['valid'] = True
                result['metric_value'] = 1.0 if int(gt_num) == pred_int else 0.0

    else:
        # ── Regression (MAE): stress, readiness, sleep_quality, calories, stress_resilience ──
        try:
            gt_num = float(ground_truth.strip())
        except ValueError:
            return result

        pred_num = extract_number(predicted, task_name)

        vmin, vmax = meta.get("valid_range", (float('-inf'), float('inf')))

        result['gt_parsed'] = gt_num

        if pred_num is not None:
            result['pred_parsed'] = pred_num

            if vmin <= pred_num <= vmax * 1.5:  # 약간의 여유
                result['valid'] = True
                result['metric_value'] = abs(gt_num - pred_num)

    return result


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='13B 평가 결과 MAE/Accuracy 계산')
    parser.add_argument('--input', required=True, help='결과 JSON (통합 또는 task별)')
    parser.add_argument('--mode', default=None, help='논문 비교용 모드 (zeroshot/fewshot/finetuned)')
    parser.add_argument('--task', default=None, help='특정 task만 계산')
    parser.add_argument('--verbose', action='store_true', help='샘플별 상세 출력')
    args = parser.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    if args.task:
        results = [r for r in results if r.get('task') == args.task]

    print("=" * 80)
    print(f"13B MAE/Accuracy 계산")
    print(f"Input: {args.input}  ({len(results)} samples)")
    if args.mode:
        print(f"Mode:  {args.mode}")
    print("=" * 80)

    # ── Task별 그룹핑 ──
    task_groups = defaultdict(list)
    for r in results:
        if r.get('status') != 'SUCCESS':
            continue
        task_groups[r['task']].append(r)

    # ── Task별 계산 ──
    summary = {}

    for task_name in sorted(task_groups.keys()):
        samples = task_groups[task_name]
        meta = TASK_META.get(task_name, {})
        metric_type = meta.get("metric", "mae")

        valid_metrics = []
        parse_fail = 0
        range_fail = 0

        for s in samples:
            ev = evaluate_sample(task_name, s['ground_truth'], s['predicted'])

            if ev['pred_parsed'] is None:
                parse_fail += 1
            elif not ev['valid']:
                range_fail += 1
            else:
                valid_metrics.append(ev['metric_value'])

            if args.verbose and len(valid_metrics) + parse_fail + range_fail <= 5:
                print(f"\n  [{task_name}] idx={s.get('val_idx', '?')}")
                print(f"    GT:   {s['ground_truth']}")
                pred_short = (s['predicted'] or '')[:120]
                print(f"    Pred: {pred_short}{'...' if len(s.get('predicted', '') or '') > 120 else ''}")
                print(f"    → gt={ev['gt_parsed']}, pred={ev['pred_parsed']}, valid={ev['valid']}, metric={ev['metric_value']}")

        n_total = len(samples)
        n_valid = len(valid_metrics)

        print(f"\n{'─' * 60}")
        print(f"[{task_name}]")
        print(f"  Total: {n_total}, Valid: {n_valid}, Parse fail: {parse_fail}, Range fail: {range_fail}")
        print(f"  Parse rate: {n_valid/n_total*100:.1f}%")

        if n_valid == 0:
            print(f"  ❌ No valid predictions")
            continue

        if metric_type == "mae":
            mae = sum(valid_metrics) / n_valid
            print(f"  MAE: {mae:.4f} (n={n_valid})")

            # 논문 비교
            paper_key = f"paper_{args.mode}" if args.mode else None
            paper_val = meta.get(paper_key) if paper_key else None
            if paper_val is not None:
                diff = mae - paper_val
                icon = "✅" if diff < 0 else "⚠️" if diff < 0.5 else "❌"
                print(f"  Paper: {paper_val:.2f}  Δ={diff:+.4f} {icon}")

            summary[task_name] = {'metric': 'MAE', 'value': mae, 'n': n_valid}

        else:  # acc
            acc = sum(valid_metrics) / n_valid * 100
            print(f"  Accuracy: {acc:.2f}% (n={n_valid})")

            paper_key = f"paper_{args.mode}" if args.mode else None
            paper_val = meta.get(paper_key) if paper_key else None
            if paper_val is not None:
                diff = acc - paper_val
                icon = "✅" if diff > 0 else "⚠️" if diff > -5 else "❌"
                print(f"  Paper: {paper_val:.1f}%  Δ={diff:+.2f}% {icon}")

            summary[task_name] = {'metric': 'Acc', 'value': acc, 'n': n_valid}

    # ── 최종 요약 테이블 ──
    if len(summary) > 1:
        print(f"\n{'=' * 80}")
        print("SUMMARY TABLE")
        print("=" * 80)

        paper_mode = args.mode or "zeroshot"

        print(f"\n  {'Task':<35} {'Metric':>6} {'Ours':>10} {'Paper':>10} {'Δ':>10}")
        print(f"  {'─'*75}")

        for task_name in sorted(summary.keys()):
            s = summary[task_name]
            meta = TASK_META.get(task_name, {})
            paper_val = meta.get(f"paper_{paper_mode}")

            if s['metric'] == 'MAE':
                ours_str = f"{s['value']:.4f}"
                paper_str = f"{paper_val:.2f}" if paper_val else "—"
                diff_str = f"{s['value'] - paper_val:+.4f}" if paper_val else "—"
            else:
                ours_str = f"{s['value']:.2f}%"
                paper_str = f"{paper_val:.1f}%" if paper_val else "—"
                diff_str = f"{s['value'] - paper_val:+.2f}%" if paper_val else "—"

            print(f"  {task_name:<35} {s['metric']:>6} {ours_str:>10} {paper_str:>10} {diff_str:>10}  (n={s['n']})")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
