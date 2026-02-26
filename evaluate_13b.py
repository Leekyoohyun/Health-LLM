#!/usr/bin/env python3
"""
13B 모델 통합 평가 스크립트

data/evaluation_tasks/ 에 있는 *_eval_zeroshot.json 파일을 자동 탐색하여 평가.
task 파일만 추가/제거하면 평가 대상이 자동 변경됨.

프롬프트 형식: hc_text + uc_text + sensor_data + question + format_prompt
  (논문 inference.py 방식과 동일)

Usage:
  # ZeroShot baseline (medalpaca-13b)
  python evaluate_13b.py --mode zeroshot --model medalpaca/medalpaca-13b

  # FewShot baseline (medalpaca-13b, 3-shot, train data에서 예시 추출)
  python evaluate_13b.py --mode fewshot --model medalpaca/medalpaca-13b --train_data data/finetune_data.json

  # Finetuned (full fine-tuned 13b, zeroshot 형식)
  python evaluate_13b.py --mode finetuned --model outputs/healthalpaca-13b-full

  # PMData/AW_FB 반으로 자르기
  python evaluate_13b.py --mode zeroshot --model medalpaca/medalpaca-13b --max_samples 0.5
"""

import argparse
import json
import os
import random
import time
import torch
from glob import glob
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig

# ── 추론 설정 (논문 방식 동일) ──
GEN_KWARGS = dict(
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
)
MAX_NEW_TOKENS = 128

# ── Task별 Health Context + Format Example 정의 ──
# hc_text: 논문 Table 2, Section 4.1 기반
# format_example: 각 task의 중간값 (앵커링 편향 최소화)
# train_keyword: finetune_data.json에서 해당 task 샘플 매칭용
TASK_CONFIG = {
    # ── PMData (논문 제공) ──
    "PMData_stress": {
        "hc_text": "Stress level is an indicator of an individual's physiological and psychological stress state, reflecting how well a person is coping with daily pressures and challenges.",
        "format_example": "3",
        "train_keyword": "predict user's stress",
    },
    "PMData_readiness": {
        "hc_text": "Readiness score is an indicator of how prepared our body is for physical activity and daily tasks, reflecting overall recovery and physiological preparedness.",
        "format_example": "5",
        "train_keyword": "predict user's readiness",
    },
    "PMData_sleep_quality": {
        "hc_text": "Sleep quality is an indicator of the overall quality and restfulness of sleep, reflecting how well a person slept during the night.",
        "format_example": "3",
        "train_keyword": "predict user's sleep quality",
    },
    "PMData_fatigue": {
        "hc_text": "Fatigue level is an indicator of signs of tiredness or exhaustion, reflecting the degree of physical or mental weariness experienced by an individual.",
        "format_example": "3",
        "train_keyword": "predict user's fatigue",
    },
    # ── AW_FB (논문 Table 2 + Section 4.1 기반, 직접 작성) ──
    "AW_FB_activity": {
        "hc_text": "Activity Recognition involves identifying the specific type of physical activity an individual is performing, such as sitting, lying, or running. It is determined based on sensor data including step counts, heart rate, and burned calories.",
        "format_example": "Sitting",
        "train_keyword": "predict the activity type",
    },
    "AW_FB_calories": {
        "hc_text": "Calorie Burn Estimate refers to the amount of energy expended by an individual during physical activities. It is estimated by analyzing the correlation between step counts, heart rate, and the intensity of the physical movement.",
        "format_example": "50.00",
        "train_keyword": "predict the burned calories",
    },
    # ── LifeSnaps (논문 Table 2 + Section 4.1 기반, 직접 작성) ──
    "LifeSnaps_stress_resilience": {
        "hc_text": "Stress Resilience Index (SSI) refers to an individual's ability to recover from or adapt positively to stressors. It is determined by stress scores, positive and negative affect scores, physical activity duration, and sleep quality metrics such as efficiency and sleep stage ratios.",
        "format_example": "2.50",
        "train_keyword": "stress resilience",
    },
    "LifeSnaps_sleep_disorder": {
        "hc_text": "Sleep disorder prediction aims to identify potential sleep issues such as insomnia or sleep apnea. It is identified through metrics like sleep duration, awake time, sleep efficiency, sleep stage ratios (REM, deep, light), heart rate variability (RMSSD), SPO2, and breathing rate during sleep.",
        "format_example": "0",
        "train_keyword": "sleep disorder",
    },
}


def load_prompt_template(path):
    with open(path) as f:
        return json.load(f)


def make_format_prompt(format_example):
    """논문 inference.py L144 형식의 format_prompt 생성"""
    return f"\n\nFor example, the answer should be in the following format:\nAnswer: {format_example}"


def wrap_input(task_name, raw_input):
    """raw input에 hc_text (앞) + format_prompt (뒤) 추가"""
    config = TASK_CONFIG.get(task_name, {})
    hc = config.get("hc_text", "")
    fmt_ex = config.get("format_example", "")

    result = raw_input
    if hc:
        result = hc + " " + result
    if fmt_ex:
        result = result + make_format_prompt(fmt_ex)
    return result


def build_medalpaca_prompt(template, instruction, input_text):
    """medalpaca 프롬프트 형식 생성 (DataHandler.generate_prompt 동일)"""
    return (
        f'{template["primer"]}'
        f'{template["instruction"]}{instruction}'
        f'{template["input"]}{input_text}'
        f'{template["output"]}'
    )


def build_fewshot_input(examples, actual_input_wrapped, max_chars=600):
    """
    Few-shot 프롬프트 구성
    - examples: train data에서 뽑은 예시 (raw input, hc_text/format_prompt 없음)
    - actual_input_wrapped: hc_text + raw_input + format_prompt (이미 wrap된 상태)
    """
    parts = []
    for i, ex in enumerate(examples, 1):
        ex_input = ex['input']
        if len(ex_input) > max_chars:
            ex_input = ex_input[:max_chars] + "..."
        parts.append(f"[Example {i}]\n{ex_input}\nAnswer: {ex['output']}")

    few_shot_block = "\n\n".join(parts)
    return (
        "Here are some example predictions:\n\n"
        + few_shot_block
        + "\n\nNow predict for the following case:\n"
        + actual_input_wrapped
    )


def discover_tasks(task_dir):
    """task_dir에서 *_eval_zeroshot.json 파일 자동 탐색"""
    pattern = os.path.join(task_dir, "*_eval_zeroshot.json")
    files = sorted(glob(pattern))
    tasks = []
    for f in files:
        basename = os.path.basename(f)
        task_name = basename.replace("_eval_zeroshot.json", "")
        tasks.append({"name": task_name, "file": f})
    return tasks


def load_train_pools(train_data_path):
    """finetune_data.json을 로드하고 task별로 분류"""
    with open(train_data_path) as f:
        all_data = json.load(f)

    pools = {}
    for task_name, config in TASK_CONFIG.items():
        keyword = config.get("train_keyword", "")
        if keyword:
            pool = [s for s in all_data if keyword.lower() in s['input'].lower()]
            if pool:
                pools[task_name] = pool

    return pools


def main():
    parser = argparse.ArgumentParser(description="13B 모델 통합 평가")
    parser.add_argument("--mode", required=True, choices=["zeroshot", "fewshot", "finetuned"])
    parser.add_argument("--model", required=True, help="모델 경로 (HF hub 또는 로컬)")
    parser.add_argument("--task_dir", default="data/evaluation_tasks/")
    parser.add_argument("--prompt_template", default="medalpaca/prompt_templates/medalpaca.json")
    parser.add_argument("--train_data", default="data/finetune_data.json",
                        help="Few-shot 예시용 학습 데이터 (fewshot 모드 전용)")
    parser.add_argument("--output", default=None, help="결과 JSON 경로 (미지정시 자동 생성)")
    parser.add_argument("--max_samples", type=float, default=0,
                        help="Task별 최대 샘플 수 (0=전체, 0<x<1=비율, x>=1=개수)")
    parser.add_argument("--n_shots", type=int, default=3, help="Few-shot 예시 수")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Output 파일명 자동 생성
    if args.output is None:
        suffix = "_half" if args.max_samples == 0.5 else ""
        args.output = f"results_13b_{args.mode}{suffix}.json"

    # ── Task 탐색 ──
    tasks = discover_tasks(args.task_dir)
    if not tasks:
        print(f"ERROR: No *_eval_zeroshot.json files found in {args.task_dir}")
        return

    print("=" * 80)
    print(f"13B EVALUATION — Mode: {args.mode.upper()}")
    print("=" * 80)
    print(f"Model:        {args.model}")
    print(f"Tasks:        {len(tasks)}")
    for t in tasks:
        with open(t['file']) as f:
            n = len(json.load(f))
        config = TASK_CONFIG.get(t['name'], {})
        hc_status = "hc_text OK" if config.get("hc_text") else "NO hc_text"
        print(f"  - {t['name']}: {n} samples ({hc_status})")
    print(f"Max samples:  {args.max_samples if args.max_samples > 0 else 'all'}")
    print(f"Output:       {args.output}")
    print(f"Seed:         {args.seed}")
    if args.mode == "fewshot":
        print(f"N-shots:      {args.n_shots}")
        print(f"Train data:   {args.train_data}")
    print("=" * 80)

    # ── Fewshot: 학습 데이터 로드 ──
    train_pools = {}
    if args.mode == "fewshot":
        if os.path.exists(args.train_data):
            print(f"\nLoading train data for few-shot examples: {args.train_data}")
            train_pools = load_train_pools(args.train_data)
            for task_name, pool in train_pools.items():
                print(f"  - {task_name}: {len(pool)} train samples")
        else:
            print(f"\nWARNING: Train data not found: {args.train_data}")
            print("  → Falling back to eval data leave-one-out")

    # ── 프롬프트 템플릿 로드 ──
    template = load_prompt_template(args.prompt_template)

    # ── 모델 로드 (multi-GPU 자동 분산) ──
    print(f"\nLoading model: {args.model}")
    t0 = time.time()

    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = LlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Device 확인
    if hasattr(model, 'hf_device_map'):
        devices = sorted(set(str(v) for v in model.hf_device_map.values()))
        print(f"✓ Model loaded on: {devices} ({time.time()-t0:.0f}s)")
    else:
        print(f"✓ Model loaded ({time.time()-t0:.0f}s)")

    gen_config = GenerationConfig(**GEN_KWARGS)

    # ── 평가 루프 ──
    all_results = []
    total_samples = 0
    total_errors = 0
    start_time = time.time()

    for task_idx, task in enumerate(tasks):
        task_name = task['name']

        # Load eval data
        with open(task['file']) as f:
            eval_data = json.load(f)

        # Max samples 적용
        if args.max_samples > 0:
            if args.max_samples < 1:
                n = max(1, int(len(eval_data) * args.max_samples))
            else:
                n = min(int(args.max_samples), len(eval_data))
            eval_data = eval_data[:n]

        print(f"\n{'=' * 80}")
        print(f"[Task {task_idx+1}/{len(tasks)}] {task_name} ({len(eval_data)} samples)")
        print("=" * 80)

        for idx in tqdm(range(len(eval_data)), desc=task_name):
            sample = eval_data[idx]
            instruction = sample['instruction']
            raw_input = sample['input']
            ground_truth = sample['output']

            # ── input 구성: hc_text + raw_input + format_prompt ──
            wrapped_input = wrap_input(task_name, raw_input)

            if args.mode == "fewshot":
                # Fewshot: train pool에서 예시 추출 (없으면 eval leave-one-out)
                pool = train_pools.get(task_name)
                if pool is None:
                    pool = [s for j, s in enumerate(eval_data) if j != idx]

                # 짧은 예시 우선 선택 (토큰 초과 방지)
                pool_sorted = sorted(pool, key=lambda x: len(x['input']))
                candidates = pool_sorted[:args.n_shots * 3]
                n_pick = min(args.n_shots, len(candidates))
                examples = random.sample(candidates, n_pick)

                # 예시는 raw input (hc_text/format_prompt 없음), 실제 query만 wrapped
                input_text = build_fewshot_input(examples, wrapped_input)
            else:
                # ZeroShot / Finetuned: hc_text + raw_input + format_prompt
                input_text = wrapped_input

            # 프롬프트 생성 (medalpaca 템플릿)
            prompt = build_medalpaca_prompt(template, instruction, input_text)

            try:
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                n_input_tokens = input_ids.shape[1]
                input_ids = input_ids.to(model.device)

                with torch.no_grad():
                    output = model.generate(
                        input_ids=input_ids,
                        generation_config=gen_config,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )

                n_output_tokens = output[0].shape[0] - n_input_tokens
                decoded = tokenizer.decode(output[0], skip_special_tokens=False)
                response = decoded.split(template["response_split"])[-1].strip()

                result = {
                    'task': task_name,
                    'val_idx': idx,
                    'ground_truth': ground_truth,
                    'predicted': response,
                    'prompt': prompt,
                    'raw_output': decoded,
                    'input_tokens': n_input_tokens,
                    'output_tokens': n_output_tokens,
                    'status': 'SUCCESS',
                }

            except Exception as e:
                total_errors += 1
                result = {
                    'task': task_name,
                    'val_idx': idx,
                    'ground_truth': ground_truth,
                    'predicted': None,
                    'prompt': prompt,
                    'raw_output': None,
                    'input_tokens': n_input_tokens if 'n_input_tokens' in dir() else None,
                    'output_tokens': None,
                    'status': 'ERROR',
                    'error': str(e)[:200],
                }

            all_results.append(result)

            # 중간 저장 (100개마다) — 스팟 인스턴스 대비
            if len(all_results) % 100 == 0:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)

        total_samples += len(eval_data)
        print(f"✅ {task_name}: {len(eval_data)} samples done")

    # ── 결과 저장 ──
    elapsed = time.time() - start_time

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"Mode:           {args.mode}")
    print(f"Total samples:  {total_samples}")
    print(f"Errors:         {total_errors}")
    print(f"Success rate:   {(total_samples - total_errors) / max(total_samples, 1) * 100:.1f}%")
    print(f"Time:           {elapsed/60:.1f} min ({elapsed/max(total_samples,1):.1f} sec/sample)")
    print(f"Results:        {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
