#!/usr/bin/env python3
"""Prepare rationale-generation train/val JSONL from VisualPRM result files."""

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


def option_lines(options: list[str]) -> str:
    return "\n".join(f"{chr(65 + idx)}. {opt}" for idx, opt in enumerate(options))


def rationale_lines(steps: list[dict[str, Any]]) -> str:
    return "\n".join(f"{idx + 1}. {step.get('text', '').strip()}" for idx, step in enumerate(steps))


def step_label_summary(step_results: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"+": 0, "-": 0}
    for row in step_results:
        label = row.get("label", "-")
        summary["+" if label == "+" else "-"] += 1
    return summary


def solution_sort_key(solution: dict[str, Any]) -> tuple[Any, ...]:
    step_results = solution.get("step_results", [])
    label_summary = step_label_summary(step_results)
    return (
        int(bool(solution.get("final_answer_correct"))),
        float(solution.get("solution_mc_score", 0.0)),
        label_summary["+"] - label_summary["-"],
        len(solution.get("steps", [])),
        -int(solution.get("solution_index", 10**9)),
    )


def best_solution(case: dict[str, Any]) -> dict[str, Any] | None:
    solutions = case.get("solutions", [])
    if not solutions:
        return None
    ranked = sorted(solutions, key=solution_sort_key, reverse=True)
    for candidate in ranked:
        if candidate.get("final_answer_correct"):
            return candidate
    return ranked[0]


def build_record(payload: dict[str, Any], case: dict[str, Any], solution: dict[str, Any]) -> dict[str, Any]:
    options = case.get("options", [])
    steps = solution.get("steps", [])
    step_results = solution.get("step_results", [])
    human = (
        "### Question:\n"
        f"{case.get('question', '').strip()}\n\n"
        "### Metadata:\n"
        f"Dataset: {payload.get('dataset', '')}\n"
        f"Case type: {case.get('case_type', '')}\n"
        f"Modality: {case.get('modality', '')}\n\n"
        "### Options:\n"
        f"{option_lines(options)}\n\n"
        "### Instruction:\n"
        "Provide a concise step-by-step medical rationale that leads to the correct answer."
    )
    assistant = rationale_lines(steps)
    label_summary = step_label_summary(step_results)

    return {
        "id": f"{payload.get('dataset', 'dataset')}::{case.get('id')}",
        "dataset": payload.get("dataset"),
        "case_id": case.get("id"),
        "image_url": case.get("image_url"),
        "gold_index": case.get("gold"),
        "gold_letter": chr(65 + int(case.get("gold", 0))),
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": assistant},
        ],
        "metadata": {
            "source_result_file": payload.get("input_file"),
            "source_generated_at": payload.get("generated_at"),
            "solution_index": solution.get("solution_index"),
            "solution_mc_score": solution.get("solution_mc_score"),
            "solution_label": solution.get("solution_label"),
            "final_answer_correct": solution.get("final_answer_correct"),
            "final_answer_letter": solution.get("final_answer_letter"),
            "num_steps": len(steps),
            "step_label_summary": label_summary,
            "step_labels": [row.get("label", "-") for row in step_results],
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare rationale generation train set")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-name", default="rationale_train_sample.jsonl")
    parser.add_argument("--val-name", default="rationale_val_sample.jsonl")
    parser.add_argument("--manifest-name", default="rationale_manifest.json")
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--val-cases", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()

    for input_name in args.inputs:
        path = Path(input_name)
        payload = load_payload(path)
        for case in payload.get("cases", []):
            case_id = str(case.get("id"))
            if not case_id or case_id in seen_case_ids:
                continue
            solution = best_solution(case)
            if solution is None:
                continue
            seen_case_ids.add(case_id)
            records.append(build_record(payload, case, solution))
            if len(records) >= args.max_cases:
                break
        if len(records) >= args.max_cases:
            break

    if len(records) < 2:
        raise ValueError("Need at least 2 cases to build train/val split")

    rng = random.Random(args.seed)
    rng.shuffle(records)

    val_case_count = max(1, min(args.val_cases, len(records) - 1))
    val_records = records[:val_case_count]
    train_records = records[val_case_count:]

    output_dir = Path(args.output_dir)
    train_path = output_dir / args.train_name
    val_path = output_dir / args.val_name
    manifest_path = output_dir / args.manifest_name

    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)

    manifest = {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "total_cases": len(records),
        "train_cases": [row["case_id"] for row in train_records],
        "val_cases": [row["case_id"] for row in val_records],
        "source_inputs": args.inputs,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
