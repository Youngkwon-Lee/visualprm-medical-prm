#!/usr/bin/env python3
"""Flatten VisualPRM-style case-solution rows into step-level scorer rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def flatten_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    steps = row.get("steps", [])
    flattened: list[dict[str, Any]] = []

    for idx, step in enumerate(steps):
        prefix_steps = [
            {
                "step_index": prev.get("step_index"),
                "title": prev.get("title", f"Step {j + 1}"),
                "text": prev.get("text", ""),
                "expected_accuracy": prev.get("expected_accuracy"),
                "label": prev.get("label"),
            }
            for j, prev in enumerate(steps[:idx])
        ]

        flattened.append(
            {
                "id": f"{row.get('id')}::step_{step.get('step_index', idx + 1)}",
                "image": row.get("image"),
                "dataset": row.get("dataset"),
                "case_id": row.get("case_id"),
                "case_type": row.get("case_type"),
                "modality": row.get("modality"),
                "question": row.get("question"),
                "options": row.get("options", []),
                "gold_index": row.get("gold_index"),
                "gold_letter": row.get("gold_letter"),
                "solution_index": row.get("solution_index"),
                "solution_expected_accuracy": row.get("solution_expected_accuracy"),
                "solution_label": row.get("solution_label"),
                "final_answer_index": row.get("final_answer_index"),
                "final_answer_letter": row.get("final_answer_letter"),
                "final_answer_correct": row.get("final_answer_correct"),
                "prefix_steps": prefix_steps,
                "current_step": {
                    "step_index": step.get("step_index", idx + 1),
                    "title": step.get("title", f"Step {idx + 1}"),
                    "text": step.get("text", ""),
                },
                "expected_accuracy": step.get("expected_accuracy"),
                "label": step.get("label"),
                "rollout_success": step.get("rollout_success"),
                "rollout_total": step.get("rollout_total"),
            }
        )

    return flattened


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten VisualPRM-style rows into step-level samples")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = load_rows(input_path)
    flattened = [sample for row in rows for sample in flatten_row(row)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in flattened:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"source_rows": len(rows), "step_rows": len(flattened), "output": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
