#!/usr/bin/env python3
"""
HealthAlpaca 모델 평가 스크립트 (태스크별 개별 평가)
Baseline (MedAlpaca-7b) vs Fine-tuned (HealthAlpaca) 성능 비교

평가 내용:
- Regression 태스크: MAE (Mean Absolute Error)
- Classification 태스크: Accuracy
- GPU 메모리 사용량 모니터링
- 결과 S3 자동 업로드
"""

import os
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
        """현재 GPU 메모리 사용량 (MB)"""
        if torch.cuda.is_available():
            return {
                'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
                'timestamp': datetime.now().isoformat()
            }
        return None

    def _monitor_loop(self):
        """백그라운드 모니터링 루프"""
        while self.running:
            mem = self._get_gpu_memory()
            if mem:
                self.memory_history.append(mem)
            time.sleep(self.interval)

    def start(self):
        """모니터링 시작"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.running = True
        self.memory_history = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("   ✓ GPU Monitor started")

    def stop(self):
        """모니터링 중지 및 통계 반환"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

        if not self.memory_history:
            return None

        allocated_list = [m['allocated_mb'] for m in self.memory_history]
        reserved_list = [m['reserved_mb'] for m in self.memory_history]

        stats = {
            'avg_allocated_mb': np.mean(allocated_list),
            'max_allocated_mb': max(allocated_list),
            'min_allocated_mb': min(allocated_list),
            'avg_reserved_mb': np.mean(reserved_list),
            'max_reserved_mb': max(reserved_list),
            'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            'samples_count': len(self.memory_history)
        }
        print(f"   ✓ GPU Monitor stopped (Peak: {stats['peak_memory_mb']:.1f} MB)")
        return stats

    def get_current(self):
        """현재 GPU 메모리 상태"""
        return self._get_gpu_memory()


def get_nvidia_smi_info():
    """nvidia-smi로 GPU 정보 조회"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'gpu_name': parts[0],
                'total_memory_mb': float(parts[1]),
                'used_memory_mb': float(parts[2]),
                'free_memory_mb': float(parts[3]),
                'gpu_utilization_percent': float(parts[4])
            }
    except Exception as e:
        print(f"   ⚠ nvidia-smi failed: {e}")
    return None


# ============================================================================
# S3 업로드 함수
# ============================================================================
def upload_to_s3(local_path, s3_bucket, s3_prefix="evaluation_results"):
    """결과 파일을 S3에 업로드"""
    try:
        if os.path.isdir(local_path):
            # 디렉토리 전체 업로드
            cmd = f"aws s3 sync {local_path} s3://{s3_bucket}/{s3_prefix}/ --quiet"
        else:
            # 단일 파일 업로드
            filename = os.path.basename(local_path)
            cmd = f"aws s3 cp {local_path} s3://{s3_bucket}/{s3_prefix}/{filename} --quiet"

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"   ✓ Uploaded to s3://{s3_bucket}/{s3_prefix}/")
            return True
        else:
            print(f"   ⚠ S3 upload failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ⚠ S3 upload error: {e}")
        return False


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


def run_inference(model, test_data, max_samples=None, gpu_monitor=None):
    """모델 추론 수행 (GPU 메모리 로깅 포함)"""
    outputs = []
    samples = test_data if max_samples is None else test_data[:max_samples]
    total = len(samples)
    inference_gpu_snapshots = []

    for i, sample in enumerate(samples):
        # GPU 메모리 스냅샷
        if gpu_monitor and i % 10 == 0:  # 10개마다 기록
            mem = gpu_monitor.get_current()
            if mem:
                inference_gpu_snapshots.append({
                    'sample_idx': i,
                    'allocated_mb': mem['allocated_mb']
                })

        print(f"      [{i+1}/{total}] Processing... (GPU: {torch.cuda.memory_allocated()/1024**2:.0f}MB)", end='\r')

        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')

        output = model(
            instruction=instruction if instruction else None,
            input=input_text,
            max_new_tokens=128
        )
        outputs.append(output)

    print(f"      ✓ {total} samples evaluated          ")
    return outputs, inference_gpu_snapshots


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


def evaluate_single_task(task_name, task_config, baseline_model, finetuned_model, gpu_monitor=None, max_samples=None):
    """단일 태스크 평가 (GPU 메모리 추적 포함)"""
    print(f"\n   [{task_name}] {task_config['description']}")
    print(f"      Task type: {task_config['task_type']}")

    task_start_time = time.time()

    # 데이터 로드
    test_data = load_test_data(task_config['data_path'])
    if test_data is None:
        return None

    # GPU 메모리 초기 상태
    gpu_before = gpu_monitor.get_current() if gpu_monitor else None

    # Baseline 추론
    print("      Running Baseline inference...")
    baseline_outputs, baseline_gpu_snapshots = run_inference(
        baseline_model, test_data, max_samples, gpu_monitor
    )

    # Fine-tuned 추론
    print("      Running Fine-tuned inference...")
    finetuned_outputs, finetuned_gpu_snapshots = run_inference(
        finetuned_model, test_data, max_samples, gpu_monitor
    )

    # GPU 메모리 최종 상태
    gpu_after = gpu_monitor.get_current() if gpu_monitor else None

    task_duration = time.time() - task_start_time

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
        'results': results,
        'gpu_stats': {
            'before': gpu_before,
            'after': gpu_after,
            'baseline_snapshots': baseline_gpu_snapshots,
            'finetuned_snapshots': finetuned_gpu_snapshots
        },
        'duration_seconds': task_duration
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


def save_all_results(all_results, gpu_overall_stats=None, output_dir="results"):
    """전체 결과 저장 (GPU 통계 포함)"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # nvidia-smi 정보 조회
    nvidia_info = get_nvidia_smi_info()

    # 요약 결과 저장
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
        # GPU 스냅샷 요약
        if r.get('gpu_stats') and r['gpu_stats'].get('after'):
            task_summary['gpu_peak_mb'] = r['gpu_stats']['after'].get('max_allocated_mb', 0)
        summary.append(task_summary)
        total_duration += r.get('duration_seconds', 0)

    summary_file = f"{output_dir}/evaluation_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'total_duration_seconds': total_duration,
            'gpu_info': nvidia_info,
            'gpu_overall_stats': gpu_overall_stats,
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

    return output_dir, timestamp


def main():
    """메인 실행 함수"""
    print("="*80)
    print("HealthAlpaca Model Evaluation (Task-wise)")
    print("Baseline (MedAlpaca-7b) vs Fine-tuned (HealthAlpaca)")
    print("="*80)

    # =========================================================================
    # 설정
    # =========================================================================
    MAX_SAMPLES_PER_TASK = None  # None이면 전체, 숫자면 해당 개수만
    S3_BUCKET = "khlee-healthllm-checkpoints"  # S3 버킷 이름
    S3_PREFIX = "evaluation_results"  # S3 경로 prefix
    UPLOAD_TO_S3 = True  # S3 업로드 여부

    # =========================================================================
    # 0. GPU 모니터 시작
    # =========================================================================
    print("\n[Step 0] Starting GPU Monitor...")
    gpu_monitor = GPUMonitor(interval=2.0)  # 2초마다 기록
    gpu_monitor.start()

    # 초기 GPU 정보 출력
    nvidia_info = get_nvidia_smi_info()
    if nvidia_info:
        print(f"   GPU: {nvidia_info['gpu_name']}")
        print(f"   Total Memory: {nvidia_info['total_memory_mb']:.0f} MB")
        print(f"   Current Used: {nvidia_info['used_memory_mb']:.0f} MB")

    total_start_time = time.time()

    # =========================================================================
    # 1. 모델 로드
    # =========================================================================
    print("\n[Step 1] Loading models...")
    baseline_model = load_baseline_model()
    finetuned_model = attach_lora_adapter(baseline_model)

    # 모델 로드 후 GPU 상태
    if torch.cuda.is_available():
        print(f"   GPU Memory after model load: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

    # =========================================================================
    # 2. 태스크별 평가
    # =========================================================================
    print("\n[Step 2] Evaluating tasks...")
    all_results = []

    for task_name, task_config in TASKS.items():
        result = evaluate_single_task(
            task_name,
            task_config,
            baseline_model,
            finetuned_model,
            gpu_monitor=gpu_monitor,
            max_samples=MAX_SAMPLES_PER_TASK
        )
        all_results.append(result)

    # =========================================================================
    # 3. GPU 모니터 중지 및 통계 수집
    # =========================================================================
    print("\n[Step 3] Stopping GPU Monitor...")
    gpu_overall_stats = gpu_monitor.stop()

    total_duration = time.time() - total_start_time
    print(f"   Total evaluation time: {total_duration/60:.1f} minutes")

    # =========================================================================
    # 4. 결과 출력 및 저장
    # =========================================================================
    print("\n[Step 4] Saving results...")
    print_summary(all_results)
    output_dir, timestamp = save_all_results(all_results, gpu_overall_stats)

    # GPU 통계 출력
    if gpu_overall_stats:
        print(f"\n[GPU Memory Statistics]")
        print(f"   Peak Memory: {gpu_overall_stats['peak_memory_mb']:.1f} MB")
        print(f"   Avg Allocated: {gpu_overall_stats['avg_allocated_mb']:.1f} MB")
        print(f"   Max Allocated: {gpu_overall_stats['max_allocated_mb']:.1f} MB")

    # =========================================================================
    # 5. S3 업로드
    # =========================================================================
    if UPLOAD_TO_S3:
        print(f"\n[Step 5] Uploading results to S3...")
        s3_path = f"{S3_PREFIX}/{timestamp}"
        success = upload_to_s3(output_dir, S3_BUCKET, s3_path)
        if success:
            print(f"   ✓ Results available at: s3://{S3_BUCKET}/{s3_path}/")
    else:
        print("\n[Step 5] S3 upload skipped (UPLOAD_TO_S3=False)")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
