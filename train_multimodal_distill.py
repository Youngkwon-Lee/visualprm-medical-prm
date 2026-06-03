#!/usr/bin/env python3
"""LoRA SFT for multimodal GPT visual-evidence distillation records.

Expected row format is conversation-style JSONL:
{
  "image": "medical_visual_process_bench/images/...",
  "conversations": [
    {"from": "human", "value": "<image> ..."},
    {"from": "gpt", "value": "{...teacher json...}"}
  ]
}

This script is intended for GPU/RunPod pilot runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_prompt_and_target(row: dict[str, Any]) -> tuple[str, str]:
    prompt_parts: list[str] = []
    target_parts: list[str] = []
    for item in row.get("conversations", []):
        speaker = item.get("from", "")
        value = str(item.get("value", "")).strip()
        if not value:
            continue
        if speaker == "human":
            prompt_parts.append(value.replace("<image>", "").strip())
        elif speaker in {"gpt", "assistant"}:
            target_parts.append(value)
    prompt = "\n\n".join(prompt_parts).strip()
    target = "\n\n".join(target_parts).strip()
    if not prompt or not target:
        raise ValueError(f"Missing prompt/target in row: {row.get('id', 'unknown')}")
    return prompt, target


def resolve_image_path(row: dict[str, Any], image_root: Path) -> Path:
    image = Path(str(row.get("image") or ""))
    if image.is_absolute():
        return image
    return image_root / image


def build_messages(prompt: str, target: str | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


def apply_template(processor: Any, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    # Fallback for processors without a chat template.
    user_text = messages[0]["content"][1]["text"]
    if len(messages) > 1:
        assistant_text = messages[1]["content"][0]["text"]
        return f"<image>\nUser: {user_text}\nAssistant: {assistant_text}"
    return f"<image>\nUser: {user_text}\nAssistant:"


class MultimodalDistillDataset:
    def __init__(self, rows: list[dict[str, Any]], image_root: Path):
        self.rows = rows
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image

        row = self.rows[idx]
        prompt, target = extract_prompt_and_target(row)
        image_path = resolve_image_path(row, self.image_root)
        image = Image.open(image_path).convert("RGB")
        return {
            "id": row.get("id", f"row_{idx}"),
            "prompt": prompt,
            "target": target,
            "image": image,
        }


class MultimodalCollator:
    def __init__(self, processor: Any, max_length: int):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [item["image"] for item in batch]
        full_texts = [
            apply_template(
                self.processor,
                build_messages(item["prompt"], item["target"]),
                add_generation_prompt=False,
            )
            for item in batch
        ]
        prompt_texts = [
            apply_template(
                self.processor,
                build_messages(item["prompt"]),
                add_generation_prompt=True,
            )
            for item in batch
        ]

        full = self.processor(
            images=images,
            text=full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompts = self.processor(
            images=images,
            text=prompt_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = full["input_ids"].clone()
        prompt_lengths = prompts["attention_mask"].sum(dim=1).tolist()
        for row_idx, prompt_len in enumerate(prompt_lengths):
            labels[row_idx, : int(prompt_len)] = -100
        labels[full["attention_mask"] == 0] = -100
        full["labels"] = labels
        return full


def load_model(model_name: str, dtype: Any):
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:  # pragma: no cover - depends on transformers version
        AutoModelForImageTextToText = None

    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:  # pragma: no cover - depends on transformers version
        AutoModelForVision2Seq = None

    if AutoModelForImageTextToText is not None:
        try:
            return AutoModelForImageTextToText.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto",
            )
        except Exception as exc:
            logger.warning("AutoModelForImageTextToText failed: %s", exc)
    if AutoModelForVision2Seq is not None:
        return AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )
    raise RuntimeError("No compatible multimodal AutoModel class is available in this transformers version.")


def apply_lora(model, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def save_model(model, processor, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


def train(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor, get_linear_schedule_with_warmup

    train_rows = load_rows(Path(args.train_file))
    val_rows = load_rows(Path(args.val_file)) if args.val_file and Path(args.val_file).exists() else []
    logger.info("Loaded train=%s val=%s", len(train_rows), len(val_rows))

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = load_model(args.model_name, dtype=dtype)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.use_lora:
        model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    dataset = MultimodalDistillDataset(train_rows, Path(args.image_root))
    collator = MultimodalCollator(processor, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(1, (len(loader) * args.epochs) // max(1, args.grad_accum))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    model.train()
    optimizer.zero_grad()
    global_step = 0
    for epoch in range(args.epochs):
        logger.info("Epoch %s/%s", epoch + 1, args.epochs)
        epoch_loss = 0.0
        for step_idx, batch in enumerate(loader, start=1):
            batch = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / max(1, args.grad_accum)
            epoch_loss += float(loss.item())
            loss.backward()

            if step_idx % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.log_interval == 0:
                    logger.info("step=%s loss=%.4f", global_step, loss.item() * args.grad_accum)
                if args.save_interval > 0 and global_step % args.save_interval == 0:
                    save_model(model, processor, Path(args.output_dir) / f"checkpoint-{global_step}")
        logger.info("epoch=%s avg_loss=%.4f", epoch + 1, epoch_loss / max(1, len(loader)))

    save_model(model, processor, Path(args.output_dir) / "final")
    logger.info("Saved final model to %s", Path(args.output_dir) / "final")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=os.getenv("MODEL_NAME", "google/gemma-4-E4B-it"))
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file")
    parser.add_argument("--image_root", default=os.getenv("IMAGE_ROOT", str(ROOT / "data")))
    parser.add_argument("--output_dir", default=os.getenv("OUTPUT_DIR", str(ROOT / "models" / "visual_distill")))
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("TRAINING_BATCH_SIZE", "1")))
    parser.add_argument("--grad_accum", type=int, default=int(os.getenv("TRAINING_GRAD_ACCUM", "8")))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("TRAINING_EPOCHS", "1")))
    parser.add_argument("--learning_rate", type=float, default=float(os.getenv("TRAINING_LEARNING_RATE", "1e-5")))
    parser.add_argument("--warmup_ratio", type=float, default=float(os.getenv("TRAINING_WARMUP_RATIO", "0.03")))
    parser.add_argument("--max_length", type=int, default=int(os.getenv("MAX_LENGTH", "2048")))
    parser.add_argument("--save_interval", type=int, default=int(os.getenv("TRAINING_SAVE_INTERVAL", "100")))
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    args = parser.parse_args()
    args.gradient_checkpointing = not args.no_gradient_checkpointing
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
