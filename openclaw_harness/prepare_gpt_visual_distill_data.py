#!/usr/bin/env python3
"""Prepare GPT visual-evidence distillation data from VQA result JSONL.

The output is conversation-style JSONL with two tasks:

- descriptor: image + clinical stem, no answer options -> visual descriptor JSON
- verifier: image + question + options -> option-level visual verifier JSON

Important: if the source JSONL is a benchmark test set, this data is for
debug/pilot distillation only. Do not train on a test split and then report
performance on the same split as a fair benchmark result.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_image_path(path: str) -> str:
    path = path.replace("\\", "/")
    marker = "/data/medical_visual_process_bench/"
    if marker in path:
        return path[path.index(marker) + len("/data/") :]
    marker = "/images/"
    if marker in path:
        return path[path.index(marker) + 1 :]
    return path


def option_lines(options: list[str]) -> str:
    return "\n".join(f"{idx}. {option}" for idx, option in enumerate(options))


def get_steps(row: dict[str, Any]) -> list[str]:
    steps = row.get("steps") or []
    if not steps and row.get("attempts"):
        steps = row["attempts"][0].get("steps") or []
    return [normalize_text(step) for step in steps if normalize_text(step)]


def strip_decision_text(text: str, options: list[str]) -> str:
    """Remove obvious answer/option disclosures from visual descriptor targets."""
    cleaned = text
    cleaned = re.sub(r"(?i)^decision:\s*.*$", "", cleaned).strip()
    for option in options:
        if option:
            cleaned = re.sub(re.escape(option), "[option]", cleaned, flags=re.I)
    return normalize_text(cleaned)


def mentions_option(text: str, options: list[str]) -> bool:
    lowered = text.lower()
    return any(option and option.lower() in lowered for option in options)


def visual_descriptor_from_steps(row: dict[str, Any]) -> dict[str, Any]:
    options = [str(option) for option in row.get("options") or []]
    visual_findings: list[str] = []
    important_absences: list[str] = []
    uncertainties: list[str] = []
    for step in get_steps(row):
        lowered = step.lower()
        if lowered.startswith("visual evidence:") or lowered.startswith("concrete attributes:"):
            item = strip_decision_text(step, options)
            if item:
                visual_findings.append(item)
                if ("without " in lowered or "no " in lowered or "absence" in lowered) and not mentions_option(step, options):
                    important_absences.append(item)
        elif ("uncertain" in lowered or "difficult" in lowered) and not mentions_option(step, options):
            item = strip_decision_text(step, options)
            if item and not item.lower().startswith("decision:"):
                uncertainties.append(item)

    if not important_absences:
        for item in visual_findings:
            lowered = item.lower()
            if "without " in lowered or "no " in lowered or "absence" in lowered:
                important_absences.append(item)

    if not visual_findings:
        # Fall back to the first non-decision steps, but still remove option names.
        for step in get_steps(row):
            if step.lower().startswith("decision:") or mentions_option(step, options):
                continue
            item = strip_decision_text(step, options)
            if item:
                visual_findings.append(item)
            if len(visual_findings) >= 3:
                break

    return {
        "sample_id": row.get("sample_id"),
        "organ_or_site": "",
        "specimen_type": "",
        "visual_findings": visual_findings[:6],
        "important_absences": important_absences[:4],
        "uncertainties": uncertainties[:3],
    }


def verifier_target_from_row(row: dict[str, Any], descriptor: dict[str, Any]) -> dict[str, Any]:
    options = [str(option) for option in row.get("options") or []]
    pred = row.get("final_answer_index")
    if pred is None and row.get("attempts"):
        pred = row["attempts"][0].get("final_answer_index")
    answer = row.get("final_answer")
    if not answer and isinstance(pred, int) and 0 <= pred < len(options):
        answer = options[pred]
    rationale = normalize_text(row.get("rationale"))
    if not rationale and row.get("attempts"):
        rationale = normalize_text(row["attempts"][0].get("rationale"))
    if not rationale:
        decision_steps = [step for step in get_steps(row) if step.lower().startswith("decision:")]
        rationale = decision_steps[-1] if decision_steps else ""

    option_scores = []
    for idx, option in enumerate(options):
        is_pred = isinstance(pred, int) and idx == pred
        option_scores.append(
            {
                "index": idx,
                "option": option,
                "descriptor_support": "Best matches the teacher visual evidence and clinical stem." if is_pred else "",
                "descriptor_mismatch": "" if is_pred else "Less supported by the teacher visual evidence than the selected option.",
                "score": 5 if is_pred else 1,
            }
        )

    return {
        "sample_id": row.get("sample_id"),
        "visual_inventory": descriptor.get("visual_findings", []),
        "option_scores": option_scores,
        "final_answer_index": pred,
        "final_answer": answer,
        "confidence": row.get("confidence") or row.get("confidence_float"),
        "rationale": rationale,
    }


def build_descriptor_record(row: dict[str, Any], descriptor: dict[str, Any], source_path: Path) -> dict[str, Any]:
    prompt = (
        "<image>\n"
        "### Task:\n"
        "Describe only the concrete medical visual evidence in the image. Do not answer the question and do not mention answer options.\n\n"
        "### Clinical question stem:\n"
        f"{normalize_text(row.get('question'))}\n\n"
        "### Output JSON schema:\n"
        "{\"sample_id\": string, \"organ_or_site\": string, \"specimen_type\": string, \"visual_findings\": string[], \"important_absences\": string[], \"uncertainties\": string[]}"
    )
    return {
        "id": f"{row.get('sample_id')}::descriptor",
        "image": normalize_image_path(str(row.get("image_path") or "")),
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": json.dumps(descriptor, ensure_ascii=False)},
        ],
        "metadata": {
            "sample_id": row.get("sample_id"),
            "task": "visual_descriptor_distill",
            "source": str(source_path),
            "teacher_model": row.get("model") or "openai_reference",
            "teacher_correct": row.get("correct"),
        },
    }


def build_verifier_record(row: dict[str, Any], descriptor: dict[str, Any], target: dict[str, Any], source_path: Path) -> dict[str, Any]:
    prompt = (
        "<image>\n"
        "### Task:\n"
        "Use the image, the clinical question, and options to produce option-level visual verification JSON.\n\n"
        "### Question:\n"
        f"{normalize_text(row.get('question'))}\n\n"
        "### Options:\n"
        f"{option_lines([str(option) for option in row.get('options') or []])}\n\n"
        "### Teacher visual descriptor to imitate:\n"
        f"{json.dumps(descriptor, ensure_ascii=False)}\n\n"
        "### Output JSON schema:\n"
        "{\"sample_id\": string, \"visual_inventory\": string[], \"option_scores\": [{\"index\": int, \"option\": string, \"descriptor_support\": string, \"descriptor_mismatch\": string, \"score\": int}], \"final_answer_index\": int, \"final_answer\": string, \"confidence\": number, \"rationale\": string}"
    )
    return {
        "id": f"{row.get('sample_id')}::verifier",
        "image": normalize_image_path(str(row.get("image_path") or "")),
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": json.dumps(target, ensure_ascii=False)},
        ],
        "metadata": {
            "sample_id": row.get("sample_id"),
            "task": "option_verifier_distill",
            "source": str(source_path),
            "teacher_model": row.get("model") or "openai_reference",
            "teacher_correct": row.get("correct"),
            "gold_index": row.get("gold"),
            "teacher_final_answer_index": row.get("final_answer_index"),
        },
    }


def split_records(records: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    return shuffled[val_count:], shuffled[:val_count]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--only-correct", action="store_true", default=True)
    parser.add_argument("--include-incorrect", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_path = Path(args.source_jsonl)
    output_dir = Path(args.output_dir)
    rows = load_jsonl(source_path)
    if args.only_correct and not args.include_incorrect:
        kept_rows = [row for row in rows if row.get("correct") and row.get("valid_prediction", True)]
    else:
        kept_rows = [row for row in rows if row.get("valid_prediction", True)]

    descriptor_records = []
    verifier_records = []
    for row in kept_rows:
        descriptor = visual_descriptor_from_steps(row)
        if not descriptor.get("visual_findings"):
            continue
        verifier = verifier_target_from_row(row, descriptor)
        descriptor_records.append(build_descriptor_record(row, descriptor, source_path))
        verifier_records.append(build_verifier_record(row, descriptor, verifier, source_path))

    all_records = descriptor_records + verifier_records
    train_records, val_records = split_records(all_records, args.val_ratio, args.seed)

    write_jsonl(output_dir / "gpt_visual_distill_train.jsonl", train_records)
    write_jsonl(output_dir / "gpt_visual_distill_val.jsonl", val_records)
    write_jsonl(output_dir / "gpt_visual_distill_descriptor.jsonl", descriptor_records)
    write_jsonl(output_dir / "gpt_visual_distill_verifier.jsonl", verifier_records)

    manifest = {
        "source_jsonl": str(source_path),
        "source_rows": len(rows),
        "kept_teacher_rows": len(kept_rows),
        "descriptor_records": len(descriptor_records),
        "verifier_records": len(verifier_records),
        "all_records": len(all_records),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "warning": "Pilot/debug distillation only if source_jsonl is a benchmark test split; do not evaluate fairly on the same split.",
    }
    (output_dir / "gpt_visual_distill_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
