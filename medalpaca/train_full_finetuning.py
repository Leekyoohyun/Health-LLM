#!/usr/bin/env python3
"""
Full Fine-Tuning with FSDP + CPU Offload
MedAlpaca-7b on PMData, 4x L4 GPUs (24GB each)

FSDP full_shard: 파라미터 + 그래디언트 + optimizer state를 GPU들에 분산
CPU Offload: optimizer state를 CPU RAM으로 내려서 GPU 메모리 절약
Gradient Checkpointing: activation 메모리를 재계산으로 대체하여 절약

Usage:
    torchrun --nproc_per_node=4 train_full_finetuning.py \
        --model medalpaca/medalpaca-7b \
        --data_path ../data/finetune_data.json \
        --output_dir ../outputs/healthalpaca-7b-full
"""

import os
import sys
import csv
import subprocess
from datetime import datetime
from typing import Union, Optional

import fire
import torch
from datasets import load_dataset
from handler import DataHandler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class GPUMemoryLogger(TrainerCallback):
    """
    매 logging_steps마다 GPU 메모리 사용량을 CSV 파일에 기록.
    학습이 OOM으로 죽어도 기록이 남음 (즉시 flush).
    """

    def __init__(self, log_path: str, local_rank: int):
        self.log_path = log_path
        self.local_rank = local_rank
        self.csv_file = None
        self.csv_writer = None

    def on_train_begin(self, args, state, control, **kwargs):
        if self.local_rank == 0:
            self.csv_file = open(self.log_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "timestamp", "step", "epoch",
                "gpu0_alloc_gb", "gpu0_reserved_gb", "gpu0_max_alloc_gb",
                "loss", "learning_rate",
            ])
            self.csv_file.flush()
            print(f"  GPU memory log: {self.log_path}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.local_rank != 0 or self.csv_writer is None:
            return

        step = state.global_step
        epoch = state.epoch or 0

        # GPU 0 메모리 (rank 0 프로세스 기준)
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            max_alloc = torch.cuda.max_memory_allocated(0) / 1024**3
        else:
            alloc = reserved = max_alloc = 0

        loss = logs.get("loss", "") if logs else ""
        lr = logs.get("learning_rate", "") if logs else ""

        self.csv_writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            step, f"{epoch:.2f}",
            f"{alloc:.2f}", f"{reserved:.2f}", f"{max_alloc:.2f}",
            loss, lr,
        ])
        self.csv_file.flush()  # 즉시 디스크에 쓰기 (OOM 대비)

    def on_train_end(self, args, state, control, **kwargs):
        if self.csv_file:
            self.csv_file.close()


class S3SyncCallback(TrainerCallback):
    """
    체크포인트 저장 후 output_dir을 S3에 동기화.
    Spot 인스턴스 종료 대비: 체크포인트 + 로그 + GPU 메모리 로그 모두 S3에 보존.
    """

    def __init__(self, output_dir: str, s3_path: str, local_rank: int):
        self.output_dir = output_dir
        self.s3_path = s3_path
        self.local_rank = local_rank

    def _sync(self, tag: str = ""):
        """aws s3 sync 실행 (rank 0만)"""
        if self.local_rank != 0:
            return
        try:
            cmd = ["aws", "s3", "sync", self.output_dir, self.s3_path, "--quiet"]
            subprocess.run(cmd, check=True, timeout=600)
            print(f"  [S3] Synced to {self.s3_path} ({tag})")
        except subprocess.TimeoutExpired:
            print(f"  [S3] Sync timeout ({tag}), will retry next save")
        except Exception as e:
            print(f"  [S3] Sync failed ({tag}): {e}")

    def on_save(self, args, state, control, **kwargs):
        """체크포인트 저장 직후 S3 동기화"""
        self._sync(tag=f"step {state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        """학습 완료 후 최종 동기화"""
        self._sync(tag="final")


def main(
    model: str = "medalpaca/medalpaca-7b",
    val_set_size: Union[int, float] = 0.1,
    prompt_template: str = "prompts/medalpaca.json",
    model_max_length: int = 2048,
    train_on_inputs: bool = True,
    data_path: str = "../data/finetune_data.json",
    per_device_batch_size: int = 1,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    global_batch_size: int = 128,
    output_dir: str = "../outputs/healthalpaca-7b-full",
    save_total_limit: int = 3,
    save_steps: int = 50,
    group_by_length: bool = False,
    wandb_run_name: str = "full-finetune",
    use_wandb: bool = False,
    wandb_project: str = "medalpaca",
    optim: str = "adamw_torch",
    lr_scheduler_type: str = "cosine",
    bf16: bool = True,
    warmup_steps: int = 100,
    resume_from_checkpoint: str = None,
    s3_path: str = "",
    task_filter: str = "",
    **kwargs
):
    """
    Full Fine-Tuning (no LoRA) with FSDP + CPU Offload.

    기존 train.py 대비 변경점:
    - LoRA, 8-bit 관련 코드 제거
    - FSDP full_shard + CPU Offload 활성화
    - Gradient Checkpointing 기본 활성화
    - device_map 제거 (FSDP가 디바이스 배치 담당)
    - bf16 기본 사용 (L4 GPU 지원, fp16보다 안정적)
    - per_device_batch_size=1 (CPU Offload 시 메모리 절약)
    - S3 동기화: 체크포인트 저장 시마다 S3에 업로드 (Spot 대비)
    """

    model_name = model
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # gradient accumulation 계산
    gradient_accumulation_steps = global_batch_size // (per_device_batch_size * max(world_size, 1))

    if use_wandb and len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

    # ── 로깅 (rank 0만) ──
    if local_rank == 0:
        print("=" * 80)
        print("FULL FINE-TUNING (FSDP + CPU Offload)")
        print("=" * 80)
        print(f"Model:                    {model_name}")
        print(f"Data:                     {data_path}")
        print(f"World size:               {world_size}")
        print(f"Per device batch size:    {per_device_batch_size}")
        print(f"Gradient accum steps:     {gradient_accumulation_steps}")
        print(f"Effective batch size:     {per_device_batch_size * gradient_accumulation_steps * world_size}")
        print(f"Max token length:         {model_max_length}")
        print(f"BF16:                     {bf16}")
        print(f"Epochs:                   {num_epochs}")
        print(f"Learning rate:            {learning_rate}")
        print(f"FSDP:                     full_shard + auto_wrap + offload")
        print(f"Gradient Checkpointing:   True")
        print(f"Task filter:              {task_filter if task_filter else 'all (no filter)'}")
        print(f"S3 sync:                  {s3_path if s3_path else 'disabled'}")
        print("=" * 80)

    # ── 모델 로드 (CPU에 로드, FSDP가 분산 처리) ──
    if local_rank == 0:
        print("\nLoading model to CPU...")

    if "llama" in model_name.lower():
        load_model = LlamaForCausalLM
    else:
        load_model = AutoModelForCausalLM

    model = load_model.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        # device_map 지정하지 않음 → CPU에 로드 → FSDP가 sharding 처리
    )

    model.config.use_cache = False  # gradient checkpointing과 호환 불가

    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model loaded: {total_params:,} parameters")

    # ── 토크나이저 ──
    if "llama" in model_name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # ── 데이터 로드 & 토큰화 ──
    data_handler = DataHandler(
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        model_max_length=model_max_length,
        train_on_inputs=train_on_inputs,
    )
    data = load_dataset("json", data_files=data_path)

    # task_filter가 지정되면 해당 task만 필터링
    # e.g. --task_filter stress → "predict user's stress" 포함 샘플만 사용
    if task_filter:
        keyword = task_filter.replace("_", " ")  # sleep_quality → sleep quality
        before = len(data["train"])
        data["train"] = data["train"].filter(
            lambda x: f"predict user's {keyword}" in x["input"]
        )
        after = len(data["train"])
        if local_rank == 0:
            print(f"  Task filter: '{task_filter}' → {before} → {after} samples")

    if val_set_size > 0:
        data = (
            data["train"]
            .train_test_split(test_size=val_set_size, shuffle=True, seed=42)
            .map(data_handler.generate_and_tokenize_prompt)
        )
    else:
        data = data.shuffle(seed=42).map(data_handler.generate_and_tokenize_prompt)

    if local_rank == 0:
        print(f"  Train samples: {len(data['train'])}")
        if val_set_size > 0:
            print(f"  Val samples:   {len(data['test'])}")

    # ── FSDP 설정 ──
    fsdp_config = {
        "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
    }

    # ── TrainingArguments ──
    training_args = TrainingArguments(
        # 배치 & 학습
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        # precision
        fp16=False,
        bf16=bf16,
        # logging
        logging_steps=10,
        # optimizer
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        # eval & save (Spot 인스턴스 대비: step 단위 저장)
        evaluation_strategy="epoch" if val_set_size > 0 else "no",
        save_strategy="steps",
        save_steps=save_steps,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        # grouping
        group_by_length=group_by_length,
        # wandb
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
        # FSDP + CPU Offload
        fsdp="full_shard auto_wrap offload",
        fsdp_config=fsdp_config,
        **kwargs
    )

    # ── 콜백 설정 ──
    os.makedirs(output_dir, exist_ok=True)
    callbacks = []

    # GPU 메모리 로거
    gpu_log_path = os.path.join(output_dir, "gpu_memory_log.csv")
    callbacks.append(GPUMemoryLogger(log_path=gpu_log_path, local_rank=local_rank))

    # S3 동기화 (경로가 지정된 경우만)
    if s3_path:
        callbacks.append(S3SyncCallback(
            output_dir=output_dir, s3_path=s3_path, local_rank=local_rank
        ))

    # ── Trainer ──
    trainer = Trainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"] if val_set_size > 0 else None,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=callbacks,
    )

    # ── 학습 ──
    if local_rank == 0:
        print("\nStarting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ── 저장 (FSDP: rank 0에서 full state dict 수집 후 저장) ──
    if local_rank == 0:
        print("\nSaving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    if local_rank == 0:
        print(f"\nFull fine-tuning complete! Model saved to: {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
