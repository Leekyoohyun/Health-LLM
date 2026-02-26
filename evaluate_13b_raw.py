#!/usr/bin/env python3
"""
13B Finetuned 모델 평가 (hc_text / format_prompt 없이 raw input만 사용)

학습 데이터(finetune_data.json)와 동일한 프롬프트 형식으로 평가.
evaluate_13b.py와 동일하지만, input에 hc_text/format_prompt를 추가하지 않음.

Usage:
  python evaluate_13b_raw.py \
      --model outputs/healthalpaca-13b-full \
      --output results_13b/results_13b_finetuned_raw_half.json \
      --max_samples 0.5
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


def load_prompt_template(path):
    with open(path) as f:
        return json.load(f)


def build_medalpaca_prompt(template, instruction, input_text):
    """medalpaca 프롬프트 형식 생성"""
    return (
        f'{template["primer"]}'
        f'{template["instruction"]}{instruction}'
        f'{template["input"]}{input_text}'
        f'{template["output"]}'
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


def main():
    parser = argparse.ArgumentParser(description="13B Finetuned 평가 (raw input, hc/format 없음)")
    parser.add_argument("--model", required=True, help="모델 경로")
    parser.add_argument("--task_dir", default="data/evaluation_tasks/")
    parser.add_argument("--prompt_template", default="medalpaca/prompt_templates/medalpaca.json")
    parser.add_argument("--output", default="results_13b/results_13b_finetuned_raw.json")
    parser.add_argument("--max_samples", type=float, default=0,
                        help="Task별 최대 샘플 수 (0=전체, 0<x<1=비율, x>=1=개수)")
    parser.add_argument("--tasks", default=None,
                        help="특정 task만 실행 (쉼표 구분)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Task 탐색 ──
    tasks = discover_tasks(args.task_dir)
    if not tasks:
        print(f"ERROR: No *_eval_zeroshot.json files found in {args.task_dir}")
        return

    if args.tasks:
        task_filter = set(t.strip() for t in args.tasks.split(","))
        tasks = [t for t in tasks if t['name'] in task_filter]
        if not tasks:
            print(f"ERROR: No matching tasks for: {args.tasks}")
            return

    print("=" * 80)
    print("13B FINETUNED EVALUATION — RAW INPUT (no hc_text, no format_prompt)")
    print("=" * 80)
    print(f"Model:        {args.model}")
    print(f"Tasks:        {len(tasks)}")
    for t in tasks:
        with open(t['file']) as f:
            n = len(json.load(f))
        print(f"  - {t['name']}: {n} samples")
    print(f"Max samples:  {args.max_samples if args.max_samples > 0 else 'all'}")
    print(f"Output:       {args.output}")
    print("=" * 80)

    # ── 프롬프트 템플릿 로드 ──
    template = load_prompt_template(args.prompt_template)

    # ── 모델 로드 ──
    print(f"\nLoading model: {args.model}")
    t0 = time.time()

    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    tokenizer.model_max_length = 2048
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = LlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

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

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for task_idx, task in enumerate(tasks):
        task_name = task['name']

        with open(task['file']) as f:
            eval_data = json.load(f)

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

            # ── 핵심: raw input만 사용 (hc_text, format_prompt 없음) ──
            prompt = build_medalpaca_prompt(template, instruction, raw_input)

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
                    'input_tokens': None,
                    'output_tokens': None,
                    'status': 'ERROR',
                    'error': str(e)[:200],
                }

            all_results.append(result)

            if len(all_results) % 100 == 0:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)

        total_samples += len(eval_data)
        print(f"✅ {task_name}: {len(eval_data)} samples done")

    # ── 결과 저장 ──
    elapsed = time.time() - start_time

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Task별 파일 분리 저장
    output_dir = os.path.dirname(args.output) or "."
    task_groups = {}
    for r in all_results:
        task_groups.setdefault(r['task'], []).append(r)

    print(f"\n{'=' * 80}")
    print("RESULTS SAVED")
    print("=" * 80)
    print(f"  ALL: {args.output} ({len(all_results)} samples)")
    for tname, tresults in sorted(task_groups.items()):
        task_file = os.path.join(output_dir, f"{tname}_finetuned_raw.json")
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(tresults, f, indent=2, ensure_ascii=False)
        print(f"  {tname}: {task_file} ({len(tresults)} samples)")

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"Total samples:  {total_samples}")
    print(f"Errors:         {total_errors}")
    print(f"Success rate:   {(total_samples - total_errors) / max(total_samples, 1) * 100:.1f}%")
    print(f"Time:           {elapsed/60:.1f} min ({elapsed/max(total_samples,1):.1f} sec/sample)")
    print("=" * 80)


if __name__ == "__main__":
    main()
