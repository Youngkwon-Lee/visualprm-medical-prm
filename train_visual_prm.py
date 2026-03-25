#!/usr/bin/env python3
"""
Step-level PRM training script.

This script trains a text-only PRM on step-level JSON/JSONL rows produced by
`build_step_training_json.py` or the frontend export.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "rows" in data:
        return data["rows"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported training file format: {path}")


def label_to_text(label: str) -> str:
    return "correct" if label == "+" else "incorrect"


def format_prompt(row: dict[str, Any]) -> str:
    options = row.get("options", [])
    options_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
    prefix_steps = row.get("prefix_steps", [])
    prefix_text = "\n".join(
        f"{i+1}. {step.get('title', 'Step')} :: {step.get('text', '')}"
        for i, step in enumerate(prefix_steps)
    )
    current_step = row.get("current_step", {})
    current_title = current_step.get("title", "Step")
    current_text = current_step.get("text", "")

    return (
        "You are a process reward model for medical reasoning.\n\n"
        f"Dataset: {row.get('dataset', '')}\n"
        f"Case type: {row.get('case_type', '')}\n"
        f"Modality: {row.get('modality', '')}\n\n"
        f"Question: {row.get('question', '')}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Previous reasoning steps:\n{prefix_text if prefix_text else '[none]'}\n\n"
        f"Current step:\n{current_title} :: {current_text}\n\n"
        "Classify the current step as either correct or incorrect.\n"
        "Answer with a single label."
    )


class StepPRMDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int = 1536):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        prompt = format_prompt(row)
        target = label_to_text(row.get("label", "-"))
        full_text = f"{prompt}\nLabel: {target}"

        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            f"{prompt}\nLabel:",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = int(prompt_enc["attention_mask"].sum().item())
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def save_model(model, tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def train(
    *,
    model_name: str,
    train_file: Path,
    val_file: Path | None,
    output_dir: Path,
    batch_size: int,
    grad_accum: int,
    epochs: int,
    learning_rate: float,
    warmup_ratio: float,
    max_length: int,
    save_interval: int,
    use_lora: bool,
    gradient_checkpointing: bool,
) -> None:
    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        model_dtype = torch.float32
    logger.info("Using model dtype: %s", model_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if use_lora:
        logger.info("Applying LoRA adapters")
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    train_rows = load_rows(train_file)
    val_rows = load_rows(val_file) if val_file and val_file.exists() else []
    logger.info("Loaded %s training rows", len(train_rows))
    if val_rows:
        logger.info("Loaded %s validation rows", len(val_rows))

    train_ds = StepPRMDataset(train_rows, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = max(1, (len(train_loader) * epochs) // max(1, grad_accum))
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(epochs):
        epoch_loss = 0.0
        logger.info("Epoch %s/%s", epoch + 1, epochs)
        for step_idx, batch in enumerate(train_loader, start=1):
            outputs = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                labels=batch["labels"].to(model.device),
            )
            loss = outputs.loss / max(1, grad_accum)
            epoch_loss += loss.item()
            loss.backward()

            if step_idx % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    logger.info(
                        "  step=%s loss=%.4f lr=%.2e",
                        global_step,
                        loss.item() * grad_accum,
                        scheduler.get_last_lr()[0],
                    )

                if save_interval > 0 and global_step % save_interval == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    logger.info("Saving checkpoint to %s", checkpoint_dir)
                    save_model(model, tokenizer, checkpoint_dir)

        avg_loss = epoch_loss / max(1, len(train_loader))
        logger.info("Epoch %s complete | avg_loss=%.4f", epoch + 1, avg_loss)

    final_dir = output_dir / "final"
    logger.info("Saving final model to %s", final_dir)
    save_model(model, tokenizer, final_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a step-level PRM on training JSON/JSONL rows.")
    parser.add_argument("--model_name", type=str, default=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--train_file", type=str, default=os.getenv("TRAIN_FILE", "/workspace/data/train.jsonl"))
    parser.add_argument("--val_file", type=str, default=os.getenv("VAL_FILE", "/workspace/data/val.jsonl"))
    parser.add_argument("--output_dir", type=str, default=os.getenv("OUTPUT_DIR", str(ROOT / "models")))
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("TRAINING_BATCH_SIZE", "1")))
    parser.add_argument("--grad_accum", type=int, default=int(os.getenv("TRAINING_GRAD_ACCUM", "8")))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("TRAINING_EPOCHS", "3")))
    parser.add_argument("--learning_rate", type=float, default=float(os.getenv("TRAINING_LEARNING_RATE", "2e-5")))
    parser.add_argument("--warmup_ratio", type=float, default=float(os.getenv("TRAINING_WARMUP_RATIO", "0.1")))
    parser.add_argument("--max_length", type=int, default=int(os.getenv("MAX_LENGTH", "1536")))
    parser.add_argument("--save_interval", type=int, default=int(os.getenv("TRAINING_SAVE_INTERVAL", "500")))
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    train(
        model_name=args.model_name,
        train_file=Path(args.train_file),
        val_file=Path(args.val_file) if args.val_file else None,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        save_interval=args.save_interval,
        use_lora=args.use_lora,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )


if __name__ == "__main__":
    main()
