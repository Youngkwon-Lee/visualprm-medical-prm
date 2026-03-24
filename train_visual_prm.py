#!/usr/bin/env python3
"""
VisualPRM Training Script for RunPod A100

Trains Qwen3-VL-30B on medical VQA datasets with step-level rewards.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

ROOT = Path(__file__).resolve().parent
WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", ROOT))
DATA_DIR = WORKSPACE_DIR / "data"
OUTPUT_DIR = WORKSPACE_DIR / "models"
CACHE_DIR = WORKSPACE_DIR / ".cache"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class MedicalVQADataset(Dataset):
    """Medical VQA dataset for training."""

    def __init__(self, data_file: Path, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(f"Loading data from {data_file}")
        with open(data_file, encoding="utf-8") as f:
            self.data = json.load(f)

        logger.info(f"Loaded {len(self.data)} cases")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        case = self.data[idx]

        # Format training sample
        question = case.get("question", "")
        answer = case.get("answer", "")
        solutions = case.get("solutions", [])

        if not solutions:
            solutions = ["No solution available."]

        # Use first solution for now (can be extended to multi-solution training)
        solution = solutions[0] if solutions else ""

        # Create training prompt
        prompt = f"Question: {question}\n\nAnswer: {answer}\n\nReasoning: {solution}"

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone(),
        }


def get_dataset_config(dataset_name: str) -> dict[str, Any]:
    """Get dataset configuration."""
    configs = {
        "mvp": {
            "train_file": DATA_DIR / "pathvqa_for_app.json",
            "val_file": None,
            "batch_size": 8,
            "num_samples": 35640,
        },
        "standard": {
            "train_file": DATA_DIR / "omnimedvqa_for_app.json",
            "val_file": None,
            "batch_size": 16,
            "num_samples": 162500,
        },
        "large": {
            "train_file": DATA_DIR / "pmcvqa_for_app.json",
            "val_file": None,
            "batch_size": 16,
            "num_samples": 389500,
        },
    }
    return configs.get(dataset_name, configs["standard"])


def train(
    model_name: str = "Qwen/Qwen3-VL-30B-Instruct",
    dataset_name: str = "standard",
    batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    use_lora: bool = True,
    use_mixed_precision: bool = True,
    gradient_checkpointing: bool = True,
    save_interval: int = 500,
) -> None:
    """Train VisualPRM model."""

    logger.info(f"Training {model_name} on {dataset_name} dataset")
    logger.info(f"Batch size: {batch_size}, Epochs: {num_epochs}, LR: {learning_rate}")

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if use_mixed_precision else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Configure LoRA if enabled
    if use_lora:
        logger.info("Configuring LoRA")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load dataset
    dataset_config = get_dataset_config(dataset_name)
    train_file = dataset_config["train_file"]

    if not train_file.exists():
        logger.warning(f"Dataset file not found: {train_file}")
        logger.info("Create dummy dataset for testing")
        train_data = [
            {
                "question": "Is this a test?",
                "answer": "Yes",
                "solutions": ["This is a test sample for training."],
            }
        ] * 100
        train_file.parent.mkdir(parents=True, exist_ok=True)
        with open(train_file, "w", encoding="utf-8") as f:
            json.dump(train_data, f)

    dataset = MedicalVQADataset(train_file, tokenizer)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Setup scheduler
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            epoch_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"  Step {batch_idx + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

            # Save checkpoint
            if save_interval > 0 and global_step % save_interval == 0:
                save_path = OUTPUT_DIR / f"checkpoint-{global_step}"
                logger.info(f"Saving checkpoint to {save_path}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

    # Save final model
    logger.info(f"Saving final model to {OUTPUT_DIR / 'final'}")
    model.save_pretrained(OUTPUT_DIR / "final")
    tokenizer.save_pretrained(OUTPUT_DIR / "final")

    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VisualPRM on medical VQA")
    parser.add_argument(
        "--model_name",
        type=str,
        default=os.getenv("OPEN_MODEL_GENERATE_MODEL", "Qwen/Qwen3-VL-30B-Instruct"),
        help="Model name from HuggingFace",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mvp", "standard", "large"],
        default=os.getenv("DATASET_NAME", "standard"),
        help="Dataset size: mvp (36K), standard (162K), large (389K)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(os.getenv("TRAINING_BATCH_SIZE", "16")),
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.getenv("TRAINING_EPOCHS", "3")),
        help="Number of epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=float(os.getenv("TRAINING_LEARNING_RATE", "2e-5")),
        help="Learning rate",
    )
    parser.add_argument(
        "--use_lora",
        type=bool,
        default=True,
        help="Use LoRA for efficient training",
    )
    parser.add_argument(
        "--use_mixed_precision",
        type=bool,
        default=os.getenv("MIXED_PRECISION", "fp16") == "fp16",
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=int(os.getenv("TRAINING_SAVE_INTERVAL", "500")),
        help="Save checkpoint every N steps",
    )

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_lora=args.use_lora,
        use_mixed_precision=args.use_mixed_precision,
        save_interval=args.save_interval,
    )
