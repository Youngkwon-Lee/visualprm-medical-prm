#!/usr/bin/env python3
"""Flatten VisualPRM-style case results into step-level training JSON/JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_payload(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "cases" not in data:
        raise ValueError("Expected a VisualPRM-style result JSON with a top-level 'cases' field.")
    return data


def build_rows(payload: dict[str, Any], case_index: int = 0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cases = payload["cases"]
    if not cases:
        raise ValueError("No cases found in payload.")
    if case_index < 0 or case_index >= len(cases):
        raise IndexError(f"case_index {case_index} out of range for {len(cases)} cases")

    case = cases[case_index]
    rows: list[dict[str, Any]] = []

    for solution in case.get("solutions", []):
        steps = solution.get("steps", [])
        step_results = solution.get("step_results", [])
        for step_result in step_results:
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

    meta = {
        "dataset": payload.get("dataset"),
        "input_file": payload.get("input_file"),
        "generated_at": payload.get("generated_at"),
        "config": payload.get("config"),
        "case_index": case_index,
        "case_id": case.get("id"),
        "question": case.get("question"),
        "num_solutions": len(case.get("solutions", [])),
        "num_training_rows": len(rows),
    }
    return meta, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build step-level training JSON from a VisualPRM-style result file.")
    parser.add_argument("--input", required=True, help="Path to VisualPRM-style result JSON")
    parser.add_argument("--output", required=True, help="Path to output JSON or JSONL")
    parser.add_argument("--case-index", type=int, default=0, help="Zero-based case index to export")
    parser.add_argument("--format", choices=["json", "jsonl"], default="json")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    payload = load_payload(input_path)
    meta, rows = build_rows(payload, case_index=args.case_index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "rows": rows}, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
