#!/usr/bin/env python3
"""
HealthAlpaca 모델 평가 스크립트 (단일 GPU 순차 모델 교체) v4
Baseline (MedAlpaca-7b) vs Fine-tuned (HealthAlpaca) 성능 비교

수정사항 (v4):
- [PERF] 단일 GPU (g6.xlarge, L4 24GB) 최적화
  Phase 1: Baseline 로드 → 전체 태스크 추론 → 모델 해제
  Phase 2: Finetuned 로드 → 전체 태스크 추론 → 모델 해제
  Phase 3: 결과 결합 + 메트릭 계산 + 저장
- [BUG FIX] Baseline/Finetuned 별도 모델 인스턴스 (순차 로드)
- [BUG FIX] model_max_length 256 -> 2048
- [FEATURE] 태스크별 즉시 저장 + S3 업로드 (spot instance 안전)
- [FEATURE] max_new_tokens 128 -> 256
"""

import os
import gc
import json
import re
import torch
import numpy as np
import subprocess
import time
import threading
from datetime import datetime
from datasets import load_dataset
from medalpaca.inferer import Inferer


# ============================================================================
# GPU 메모리 모니터링 클래스
# ============================================================================
class GPUMonitor:
    """실시간 GPU 메모리 사용량 모니터링"""

    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.memory_history = []
        self.thread = None

    def _get_gpu_memory(self):
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            per_gpu = {}
            total_allocated = 0
            total_reserved = 0
            for i in range(num_gpus):
                alloc = torch.cuda.memory_allocated(i) / 1024**2
                resv = torch.cuda.memory_reserved(i) / 1024**2
                per_gpu[f'gpu{i}_allocated_mb'] = alloc
                per_gpu[f'gpu{i}_reserved_mb'] = resv
                total_allocated += alloc
                total_reserved += resv
            return {
                'total_allocated_mb': total_allocated,
                'total_reserved_mb': total_reserved,
                'max_allocated_mb': max(torch.cuda.max_memory_allocated(i) / 1024**2 for i in range(num_gpus)),
                'per_gpu': per_gpu,
                'timestamp': datetime.now().isoformat()
            }
        return None

    def _monitor_loop(self):
        while self.running:
            mem = self._get_gpu_memory()
            if mem:
                self.memory_history.append(mem)
            time.sleep(self.interval)

    def start(self):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        self.running = True
        self.memory_history = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("   GPU Monitor started")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

        if not self.memory_history:
            return None

        allocated_list = [m['total_allocated_mb'] for m in self.memory_history]
        reserved_list = [m['total_reserved_mb'] for m in self.memory_history]

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        peak_per_gpu = {}
        for i in range(num_gpus):
            peak_per_gpu[f'gpu{i}_peak_mb'] = torch.cuda.max_memory_allocated(i) / 1024**2

        stats = {
            'avg_allocated_mb': np.mean(allocated_list),
            'max_allocated_mb': max(allocated_list),
            'min_allocated_mb': min(allocated_list),
            'avg_reserved_mb': np.mean(reserved_list),
            'max_reserved_mb': max(reserved_list),
            'peak_per_gpu': peak_per_gpu,
            'samples_count': len(self.memory_history)
        }
        print(f"   GPU Monitor stopped")
        for gpu_name, peak in peak_per_gpu.items():
            print(f"      {gpu_name}: {peak:.1f} MB")
        return stats

    def get_current(self):
        return self._get_gpu_memory()


def get_nvidia_smi_info():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'total_memory_mb': float(parts[2]),
                    'used_memory_mb': float(parts[3]),
                    'free_memory_mb': float(parts[4]),
                    'utilization_percent': float(parts[5])
                })
            return gpus
    except Exception as e:
        print(f"   nvidia-smi failed: {e}")
    return None


# ============================================================================
# S3 업로드 함수
# ============================================================================
def upload_to_s3(local_path, s3_bucket, s3_prefix="evaluation_results"):
    try:
        if os.path.isdir(local_path):
            cmd = f"aws s3 sync {local_path} s3://{s3_bucket}/{s3_prefix}/ --quiet"
        else:
            filename = os.path.basename(local_path)
            cmd = f"aws s3 cp {local_path} s3://{s3_bucket}/{s3_prefix}/{filename} --quiet"

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"   -> Uploaded to s3://{s3_bucket}/{s3_prefix}/")
            return True
        else:
            print(f"   S3 upload failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   S3 upload error: {e}")
        return False


# ============================================================================
# 태스크별 설정 (8개 task - GLOBEM 제외)
# ============================================================================
TASKS = {
    "PMData_stress": {
        "data_path": "PMData_stress_train_all.json",
        "task_type": "regression",
        "description": "Stress Prediction (1-5)"
    },
    "PMData_readiness": {
        "data_path": "PMData_readiness_train_all.json",
        "task_type": "regression",
        "description": "Readiness Prediction (0-10)"
    },
    "PMData_sleep_quality": {
        "data_path": "PMData_sleep_quality_train_all.json",
        "task_type": "regression",
        "description": "Sleep Quality Prediction (1-5)"
    },
    "PMData_fatigue": {
        "data_path": "PMData_fatigue_train_all.json",
        "task_type": "classification",
        "description": "Fatigue Prediction (1-5)"
    },
    "LifeSnaps_stress_resilience": {
        "data_path": "LifeSnaps_stress_resilience_train_all.json",
        "task_type": "regression",
        "description": "Stress Resilience (0.2-5)"
    },
    "LifeSnaps_sleep_disorder": {
        "data_path": "LifeSnaps_sleep_disorder_train_all.json",
        "task_type": "classification",
        "description": "Sleep Disorder Detection (0/1)"
    },
    "AW_FB_activity": {
        "data_path": "AW_FB_activity_train_all.json",
        "task_type": "classification",
        "description": "Activity Recognition"
    },
    "AW_FB_calories": {
        "data_path": "AW_FB_calories_train_all.json",
        "task_type": "regression",
        "description": "Calorie Burn Estimation"
    },
}


def load_test_data(data_path, seed=42):
    print(f"      Loading: {data_path}")

    if not os.path.exists(data_path):
        print(f"      File not found: {data_path}")
        return None

    data = load_dataset("json", data_files=data_path)
    split = data["train"].train_test_split(
        test_size=0.1,
        shuffle=True,
        seed=seed
    )

    test_data = split["test"]
    print(f"      Test samples: {len(test_data)}")
    return test_data


def load_baseline_model(model_max_length=2048):
    """Baseline MedAlpaca-7b 로드 (GPU 0, Inferer 내부에서 device_map 자동 설정)"""
    print("   Loading Baseline MedAlpaca-7b...")

    model = Inferer(
        model_name="medalpaca/medalpaca-7b",
        prompt_template="medalpaca/prompt_templates/medalpaca.json",
        model_max_length=model_max_length,
        torch_dtype=torch.float16,
    )

    mem = torch.cuda.memory_allocated(0) / 1024**2
    print(f"   Baseline model loaded ({mem:.0f} MB)")
    return model


def load_finetuned_model(adapter_path="outputs/healthalpaca-7b-lora", model_max_length=2048):
    """Finetuned HealthAlpaca 로드 (Inferer 내장 PEFT 지원 사용)"""
    print(f"   Loading Finetuned HealthAlpaca (LoRA: {adapter_path})...")

    model = Inferer(
        model_name=adapter_path,
        base_model="medalpaca/medalpaca-7b",
        prompt_template="medalpaca/prompt_templates/medalpaca.json",
        model_max_length=model_max_length,
        torch_dtype=torch.float16,
        peft=True,
    )

    mem = torch.cuda.memory_allocated(0) / 1024**2
    print(f"   Finetuned model loaded ({mem:.0f} MB)")
    return model


def unload_model(model):
    """모델을 GPU에서 완전히 해제"""
    print("   Unloading model...")
    mem_before = torch.cuda.memory_allocated(0) / 1024**2

    del model.model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    mem_after = torch.cuda.memory_allocated(0) / 1024**2
    print(f"   GPU 0: {mem_before:.0f} MB -> {mem_after:.0f} MB (freed {mem_before - mem_after:.0f} MB)")


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


def extract_classification_answer(text):
    match = re.search(r'is\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return text.strip().lower()


def run_inference(model, test_data, label="Model", max_new_tokens=256, max_samples=None):
    """단일 모델로 테스트 데이터 전체 추론"""
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

        else:  # classification
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


def save_task_result(result, output_dir, timestamp):
    os.makedirs(output_dir, exist_ok=True)
    detail_file = f"{output_dir}/{result['task_name']}_{timestamp}.json"
    with open(detail_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"      -> Saved: {detail_file}")
    return detail_file


def save_baseline_interim(baseline_all_outputs, output_dir, timestamp):
    """Baseline 추론 결과 중간 저장 (spot instance 안전)"""
    interim_file = f"{output_dir}/baseline_interim_{timestamp}.json"
    serializable = {}
    for task_name, data in baseline_all_outputs.items():
        serializable[task_name] = {
            'outputs': data['outputs'],
            'num_samples': len(data['outputs']),
            'duration_seconds': data['duration_seconds']
        }
    with open(interim_file, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"   -> Baseline interim saved: {interim_file}")
    return interim_file


def print_summary(all_results):
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print("\n[Regression Tasks - MAE (lower is better)]")
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

    print("\n[Classification Tasks - Accuracy (higher is better)]")
    print("-" * 70)
    print(f"{'Task':<35} {'Baseline':<12} {'Fine-tuned':<12} {'Improve':<10}")
    print("-" * 70)

    for r in all_results:
        if r is None:
            continue
        if r['task_type'] == 'classification':
            baseline = r['metrics']['baseline']
            finetuned = r['metrics']['finetuned']
            if baseline is not None and finetuned is not None:
                improve = (finetuned - baseline) / max(baseline, 0.001) * 100
                print(f"{r['task_name']:<35} {baseline:<12.2%} {finetuned:<12.2%} {improve:+.1f}%")
            else:
                print(f"{r['task_name']:<35} {'N/A':<12} {'N/A':<12} {'N/A':<10}")

    print("\n" + "=" * 80)


def save_final_summary(all_results, gpu_overall_stats, output_dir, timestamp, model_config):
    nvidia_info = get_nvidia_smi_info()

    summary = []
    total_duration = 0
    for r in all_results:
        if r is None:
            continue
        task_summary = {
            'task_name': r['task_name'],
            'description': r['description'],
            'task_type': r['task_type'],
            'metric': r['metrics']['metric'],
            'baseline': r['metrics']['baseline'],
            'finetuned': r['metrics']['finetuned'],
            'num_samples': r['metrics']['num_samples'],
            'duration_seconds': r.get('duration_seconds', 0)
        }
        summary.append(task_summary)
        total_duration += r.get('duration_seconds', 0)

    summary_file = f"{output_dir}/evaluation_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'total_duration_seconds': total_duration,
            'gpu_info': nvidia_info,
            'gpu_overall_stats': gpu_overall_stats,
            'model_config': model_config,
            'tasks': summary
        }, f, indent=2, ensure_ascii=False)

    print(f"\n   Summary saved to: {summary_file}")
    return summary_file


def main():
    print("=" * 80)
    print("HealthAlpaca Model Evaluation v4 (Single GPU - Sequential Model Swap)")
    print("Baseline (MedAlpaca-7b) vs Fine-tuned (HealthAlpaca)")
    print("=" * 80)

    # =========================================================================
    # 설정
    # =========================================================================
    MAX_SAMPLES_PER_TASK = None  # None = 전체 샘플 평가 (quick test: 10)
    S3_BUCKET = "khlee-healthllm-checkpoints"
    S3_PREFIX = "evaluation_results"
    UPLOAD_TO_S3 = True
    OUTPUT_DIR = "results"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_config = {
        'mode': 'single-GPU sequential swap',
        'gpu': 0,
        'model_max_length': 2048,
        'max_new_tokens': 256
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # 0. GPU 확인 및 모니터 시작
    # =========================================================================
    print("\n[Step 0] GPU Setup...")

    num_gpus = torch.cuda.device_count()
    print(f"   Available GPUs: {num_gpus}")
    print("   Mode: Single GPU sequential (Baseline first -> swap -> Finetuned)")

    nvidia_info = get_nvidia_smi_info()
    if nvidia_info:
        for gpu in nvidia_info:
            print(f"   GPU {gpu['index']}: {gpu['name']} ({gpu['total_memory_mb']:.0f} MB total, {gpu['used_memory_mb']:.0f} MB used)")

    gpu_monitor = GPUMonitor(interval=2.0)
    gpu_monitor.start()

    total_start_time = time.time()

    # =========================================================================
    # 1. Phase 1: Baseline 추론 (전체 태스크)
    # =========================================================================
    print("\n" + "=" * 80)
    print("[Phase 1] BASELINE (MedAlpaca-7b) Inference")
    print("=" * 80)

    baseline_model = load_baseline_model(model_max_length=2048)

    # 테스트 데이터 미리 로드 (Phase 2에서 동일한 데이터 재사용)
    all_test_data = {}
    baseline_all_outputs = {}

    for task_idx, (task_name, task_config) in enumerate(TASKS.items()):
        print(f"\n   --- Task {task_idx+1}/{len(TASKS)}: {task_name} ({task_config['description']}) ---")

        test_data = load_test_data(task_config['data_path'])
        if test_data is None:
            print(f"      SKIPPED (data not found)")
            continue

        all_test_data[task_name] = test_data

        task_start = time.time()
        outputs = run_inference(
            baseline_model, test_data,
            label=f"Baseline/{task_name}",
            max_samples=MAX_SAMPLES_PER_TASK
        )
        task_duration = time.time() - task_start

        baseline_all_outputs[task_name] = {
            'outputs': outputs,
            'duration_seconds': task_duration
        }
        print(f"      Done in {task_duration/60:.1f} min ({len(outputs)} samples)")

    # Baseline 중간 저장 (spot instance 안전)
    interim_file = save_baseline_interim(baseline_all_outputs, OUTPUT_DIR, TIMESTAMP)
    if UPLOAD_TO_S3:
        upload_to_s3(interim_file, S3_BUCKET, f"{S3_PREFIX}/{TIMESTAMP}")

    # Baseline 모델 해제
    print("\n   Unloading Baseline model...")
    unload_model(baseline_model)

    # =========================================================================
    # 2. Phase 2: Finetuned 추론 + 결과 결합
    # =========================================================================
    print("\n" + "=" * 80)
    print("[Phase 2] FINETUNED (HealthAlpaca) Inference")
    print("=" * 80)

    finetuned_model = load_finetuned_model(model_max_length=2048)

    all_results = []

    for task_idx, (task_name, task_config) in enumerate(TASKS.items()):
        if task_name not in all_test_data:
            continue

        print(f"\n   --- Task {task_idx+1}/{len(TASKS)}: {task_name} ({task_config['description']}) ---")

        test_data = all_test_data[task_name]

        task_start = time.time()
        finetuned_outputs = run_inference(
            finetuned_model, test_data,
            label=f"Finetuned/{task_name}",
            max_samples=MAX_SAMPLES_PER_TASK
        )
        finetuned_duration = time.time() - task_start

        baseline_outputs = baseline_all_outputs[task_name]['outputs']
        baseline_duration = baseline_all_outputs[task_name]['duration_seconds']

        # 결과 결합
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

        # 메트릭 계산
        metrics = calculate_task_metrics(results, task_config['task_type'])
        total_task_duration = baseline_duration + finetuned_duration

        task_result = {
            'task_name': task_name,
            'description': task_config['description'],
            'task_type': task_config['task_type'],
            'metrics': metrics,
            'results': results,
            'duration_seconds': total_task_duration,
            'baseline_duration_seconds': baseline_duration,
            'finetuned_duration_seconds': finetuned_duration,
        }

        all_results.append(task_result)

        # 즉시 저장
        saved_file = save_task_result(task_result, OUTPUT_DIR, TIMESTAMP)

        # 즉시 S3 업로드
        if UPLOAD_TO_S3:
            upload_to_s3(saved_file, S3_BUCKET, f"{S3_PREFIX}/{TIMESTAMP}")

        # 중간 메트릭 출력
        m = metrics
        print(f"      {m['metric']}: Baseline={m['baseline']}, Finetuned={m['finetuned']} ({m['num_samples']} samples)")
        print(f"      Duration: baseline {baseline_duration/60:.1f}min + finetuned {finetuned_duration/60:.1f}min = {total_task_duration/60:.1f}min")

    # Finetuned 모델 해제
    print("\n   Unloading Finetuned model...")
    unload_model(finetuned_model)

    # =========================================================================
    # 3. GPU 모니터 중지
    # =========================================================================
    print("\n[Step 3] Stopping GPU Monitor...")
    gpu_overall_stats = gpu_monitor.stop()

    total_duration = time.time() - total_start_time
    print(f"   Total evaluation time: {total_duration/60:.1f} minutes")

    # =========================================================================
    # 4. 결과 요약
    # =========================================================================
    print("\n[Step 4] Final results...")
    print_summary(all_results)

    summary_file = save_final_summary(all_results, gpu_overall_stats, OUTPUT_DIR, TIMESTAMP, model_config)

    if gpu_overall_stats:
        print(f"\n[GPU Memory Statistics]")
        print(f"   Avg Allocated: {gpu_overall_stats['avg_allocated_mb']:.1f} MB")
        print(f"   Max Allocated: {gpu_overall_stats['max_allocated_mb']:.1f} MB")
        if gpu_overall_stats.get('peak_per_gpu'):
            for gpu_name, peak in gpu_overall_stats['peak_per_gpu'].items():
                print(f"   {gpu_name}: {peak:.1f} MB")

    # =========================================================================
    # 5. 최종 S3 업로드
    # =========================================================================
    if UPLOAD_TO_S3:
        print(f"\n[Step 5] Uploading final summary to S3...")
        s3_path = f"{S3_PREFIX}/{TIMESTAMP}"
        upload_to_s3(summary_file, S3_BUCKET, s3_path)
        print(f"   Results at: s3://{S3_BUCKET}/{s3_path}/")
    else:
        print("\n[Step 5] S3 upload skipped")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print(f"Results in: {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
