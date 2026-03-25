#!/usr/bin/env python3
"""Prepare train/val JSONL files for RunPod PRM training.

This merges multiple VisualPRM-style result JSON files, flattens them into
step-level rows, and performs a deterministic case-level train/val split.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_payload(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "cases" not in payload:
        raise ValueError(f"{path} is not a VisualPRM result JSON")
    return payload


def flatten_case(payload: dict[str, Any], case: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for solution in case.get("solutions", []):
        steps = solution.get("steps", [])
        for step_result in solution.get("step_results", []):
            step_id = int(step_result["step"])
            current_step = steps[step_id - 1]
            prefix_steps = steps[: step_id - 1]
            rows.append(
                {
                    "dataset": payload.get("dataset"),
                    "source_result_file": payload.get("input_file"),
                    "case_id": case.get("id"),
                    "case_type": case.get("case_type"),
                    "modality": case.get("modality"),
                    "image_url": case.get("image_url"),
                    "question": case.get("question"),
                    "options": case.get("options"),
                    "gold_index": case.get("gold"),
                    "gold_letter": chr(65 + int(case.get("gold", 0))),
                    "solution_index": solution.get("solution_index"),
                    "sampling": solution.get("sampling"),
                    "final_answer_index": solution.get("final_answer_index"),
                    "final_answer_letter": solution.get("final_answer_letter"),
                    "final_answer_correct": solution.get("final_answer_correct"),
                    "solution_mc_score": solution.get("solution_mc_score"),
                    "solution_label": solution.get("solution_label"),
                    "step_id": step_id,
                    "prefix_steps": prefix_steps,
                    "current_step": current_step,
                    "label": step_result.get("label"),
                    "mc_score": step_result.get("mc_score"),
                    "rollout_success": step_result.get("success"),
                    "rollout_total": step_result.get("total"),
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare standard RunPod training data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-cases", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    cases: list[tuple[str, list[dict[str, Any]]]] = []
    seen_case_ids: set[str] = set()

    for input_name in args.inputs:
        path = Path(input_name)
        payload = load_payload(path)
        for case in payload.get("cases", []):
            case_id = str(case.get("id"))
            if not case_id or case_id in seen_case_ids:
                continue
            rows = flatten_case(payload, case)
            if not rows:
                continue
            seen_case_ids.add(case_id)
            cases.append((case_id, rows))

    if len(cases) < 2:
        raise ValueError("Need at least 2 non-empty cases to build train/val split")

    rng = random.Random(args.seed)
    rng.shuffle(cases)

    val_case_count = max(1, min(args.val_cases, len(cases) - 1))
    val_cases = cases[:val_case_count]
    train_cases = cases[val_case_count:]

    train_rows = [row for _, rows in train_cases for row in rows]
    val_rows = [row for _, rows in val_cases for row in rows]

    output_dir = Path(args.output_dir)
    train_path = output_dir / "train_standard.jsonl"
    val_path = output_dir / "val_standard.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    summary = {
        "train_cases": [case_id for case_id, _ in train_cases],
        "val_cases": [case_id for case_id, _ in val_cases],
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "total_cases": len(cases),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
