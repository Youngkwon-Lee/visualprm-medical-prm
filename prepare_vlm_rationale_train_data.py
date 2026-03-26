#!/usr/bin/env python3
"""Convert VLM rationale generation results into multimodal SFT train/val JSONL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def option_lines(options: list[str]) -> str:
    return "\n".join(f"{chr(65 + idx)}. {opt}" for idx, opt in enumerate(options))


def normalize_image_path(path: str) -> str:
    marker = "/images/"
    if marker in path:
        return path[path.index(marker) + 1 :].replace("\\", "/")
    return path.replace("\\", "/")


def step_text(steps: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{idx + 1}. {step.get('text', '').strip()}"
        for idx, step in enumerate(steps)
        if step.get("text", "").strip()
    )


def build_record(row: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        "<image>\n"
        "### Question:\n"
        f"{row.get('question', '').strip()}\n\n"
        "### Metadata:\n"
        f"Dataset: PathVQA\n"
        f"Case type: {row.get('case_type', '')}\n"
        f"Modality: {row.get('modality', '')}\n\n"
        "### Options:\n"
        f"{option_lines(row.get('options', []))}\n\n"
        "### Instruction:\n"
        "Provide a concise step-by-step medical rationale that leads to the correct answer."
    )
    return {
        "id": f"PathVQA::{row.get('case_id')}",
        "image": normalize_image_path(str(row.get("image_url", ""))),
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": step_text(row.get("steps", []))},
        ],
        "metadata": {
            "case_id": row.get("case_id"),
            "gold_letter": row.get("gold_letter"),
            "gold_index": row.get("gold_index"),
            "final_answer_letter": row.get("final_answer_letter"),
            "final_answer_index": row.get("final_answer_index"),
            "num_steps": len(row.get("steps", [])),
            "source": "pathvqa_vlm100_generate_steps",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multimodal train/val set from VLM output")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    kept = [row for row in rows if row.get("final_answer_letter") == row.get("gold_letter")]
    records = [build_record(row) for row in kept]

    rng = random.Random(args.seed)
    rng.shuffle(records)

    val_count = max(1, int(round(len(records) * args.val_ratio))) if records else 0
    val_records = records[:val_count]
    train_records = records[val_count:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "pathvqa_vlm100_sft_train.jsonl"
    val_path = output_dir / "pathvqa_vlm100_sft_val.jsonl"
    manifest_path = output_dir / "pathvqa_vlm100_sft_manifest.json"

    train_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train_records) + ("\n" if train_records else ""), encoding="utf-8")
    val_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in val_records) + ("\n" if val_records else ""), encoding="utf-8")

    manifest = {
        "source_rows": len(rows),
        "kept_rows": len(records),
        "dropped_rows": len(rows) - len(records),
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
