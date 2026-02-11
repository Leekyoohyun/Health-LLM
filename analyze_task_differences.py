#!/usr/bin/env python3
"""
세 PMData task의 출력 차이 분석
왜 sleep_quality는 86% 성공하는데 stress/readiness는 9.7%만 성공하는가?
"""

import json

with open('baseline_full_results.json') as f:
    results = json.load(f)

# 각 task별 샘플 확인
for task_name in ['PMData_stress', 'PMData_readiness', 'PMData_sleep_quality']:
    task_results = [r for r in results if r['task'] == task_name]

    print(f"\n{'=' * 80}")
    print(f"{task_name} ({len(task_results)} samples)")
    print("=" * 80)

    # Echo 통계
    echo_count = 0
    ai_refusal_count = 0
    proper_count = 0

    for r in task_results:
        pred = r['predicted']
        if pred.startswith("The recent"):
            echo_count += 1
        elif "AI language model" in pred or "As an AI" in pred:
            ai_refusal_count += 1
        else:
            proper_count += 1

    print(f"\n📊 Statistics:")
    print(f"  Echo (input repeat):     {echo_count}/{len(task_results)} ({echo_count/len(task_results)*100:.1f}%)")
    print(f"  AI refusal:              {ai_refusal_count}/{len(task_results)} ({ai_refusal_count/len(task_results)*100:.1f}%)")
    print(f"  Proper response:         {proper_count}/{len(task_results)} ({proper_count/len(task_results)*100:.1f}%)")

    # 처음 3개 샘플 출력
    print(f"\n🔍 First 3 samples:")
    for i in range(min(3, len(task_results))):
        r = task_results[i]
        print(f"\n  [{i}] GT: {r['ground_truth']}")
        print(f"      Pred: {r['predicted'][:200]}...")

        # 분류
        if r['predicted'].startswith("The recent"):
            category = "❌ Echo"
        elif "AI language model" in r['predicted'] or "As an AI" in r['predicted']:
            category = "❌ AI refusal"
        else:
            category = "✅ Proper"
        print(f"      Type: {category}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\n💡 Hypothesis:")
print("   If sleep_quality has fewer echoes → input might be different")
print("   If all have similar echo rates → extract_number() might be the issue")
print("=" * 80)
