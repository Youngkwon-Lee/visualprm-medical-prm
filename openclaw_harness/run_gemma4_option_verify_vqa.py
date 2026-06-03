#!/usr/bin/env python3
"""Gemma4 direct VQA with option-specific visual verification.

This runner is for small debugging sets. It avoids OpenClaw/tools/web and calls
Ollama directly through /api/chat with think=false, which is required for
Gemma4 thinking models to return stable visible outputs.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_samples(path: Path, start_index: int, max_samples: int) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(text)
    end = None if max_samples <= 0 else start_index + max_samples
    samples = []
    for offset, row in enumerate(rows[start_index:end], start=start_index):
        options = row.get("options") or ["yes", "no"]
        samples.append(
            {
                "idx": int(row.get("idx", offset)),
                "id": row.get("id", row.get("sample_id", f"sample_{offset}")),
                "question": str(row.get("question", "")),
                "options": [str(option) for option in options],
                "gold": int(row.get("gold", row.get("answer_index", 0))),
                "gold_text": str(row.get("gold_text") or ""),
                "image_path": str(row.get("image_path") or row.get("image_url") or ""),
            }
        )
    return samples


def resolve_image_path(raw_path: str, samples_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    for candidate in (Path.cwd() / path, samples_path.parent / path):
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = strip_code_fence(text)
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def parse_prediction(raw_answer: str, options: list[str]) -> dict[str, Any]:
    parsed = parse_json_object(raw_answer)
    pred = parsed.get("final_answer_index")
    if isinstance(pred, str) and pred.strip().isdigit():
        pred = int(pred.strip())
    if not isinstance(pred, int):
        match = re.search(r"final_answer_index\D+(\d+)", raw_answer, flags=re.I)
        if match:
            pred = int(match.group(1))

    answer_text = normalize_text(parsed.get("final_answer"))
    if not isinstance(pred, int) and answer_text:
        normalized_answer = answer_text.lower()
        for idx, option in enumerate(options):
            option_l = option.lower()
            if normalized_answer == option_l or normalized_answer in option_l or option_l in normalized_answer:
                pred = idx
                break

    visual_inventory = parsed.get("visual_inventory") or parsed.get("visual_observations") or []
    if isinstance(visual_inventory, str):
        visual_inventory = [visual_inventory]
    if not isinstance(visual_inventory, list):
        visual_inventory = []

    option_scores = parsed.get("option_scores") or parsed.get("options_evaluation") or []
    if not isinstance(option_scores, list):
        option_scores = []

    rationale = normalize_text(parsed.get("rationale") or parsed.get("decision_rationale"))
    return {
        **parsed,
        "visual_inventory": [normalize_text(item) for item in visual_inventory if normalize_text(item)],
        "option_scores": option_scores,
        "final_answer_index": pred,
        "final_answer": answer_text or (options[pred] if isinstance(pred, int) and 0 <= pred < len(options) else None),
        "rationale": rationale,
        "parse_fallback": not bool(parsed),
    }


def load_teacher_visual_hints(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    hints: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            steps = row.get("steps") or []
            if not steps and row.get("attempts"):
                steps = row["attempts"][0].get("steps") or []
            kept = []
            for step in steps:
                text = normalize_text(step)
                lowered = text.lower()
                if lowered.startswith("visual evidence:") or lowered.startswith("concrete attributes:"):
                    kept.append(text)
            if kept:
                hints[str(row.get("sample_id"))] = " ".join(kept)
    return hints


def build_prompt(sample: dict[str, Any], *, teacher_visual_hint: str = "") -> str:
    options = "\n".join(f"{idx}: {option}" for idx, option in enumerate(sample["options"]))
    teacher_block = ""
    if teacher_visual_hint:
        teacher_block = (
            "\nDiagnostic teacher visual hint, extracted from a stronger vision model and stripped of final decision. "
            "Use it only to check what visual findings may be present; do not treat it as an answer:\n"
            f"{teacher_visual_hint}\n"
        )
    return (
        "You are a medical visual QA verifier. Use only the image, the question, and general medical knowledge. "
        "Do not use web search, benchmark memory, filenames, or dataset lookup. "
        "Return exactly one compact JSON object and no markdown.\n"
        f"{teacher_block}"
        "Task:\n"
        "1. Write visual_inventory: 3 to 6 short observations that are directly visible in the image.\n"
        "2. For every option, write one option_scores item with keys: index, option, visible_support, visible_mismatch, score.\n"
        "3. score must be an integer from 0 to 5. Give high score only when visible findings and the clinical stem both support the option.\n"
        "4. Choose final_answer_index as the option with the strongest visual support and weakest mismatch.\n"
        "JSON schema: {\"sample_id\": string, \"visual_inventory\": string[], \"option_scores\": [{\"index\": int, \"option\": string, \"visible_support\": string, \"visible_mismatch\": string, \"score\": int}], \"final_answer_index\": int, \"final_answer\": string, \"confidence\": number, \"rationale\": string}\n"
        f"Question: {sample['question']}\n"
        f"Options:\n{options}\n"
        f"final_answer_index must be between 0 and {len(sample['options']) - 1}."
    )


def call_ollama(args: argparse.Namespace, prompt: str, image_path: Path) -> tuple[float, dict[str, Any]]:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "stream": False,
        "think": False,
        "keep_alive": args.keep_alive,
        "options": {
            "temperature": args.temperature,
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
        },
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        result = json.loads(response.read())
    return time.time() - started, result


def run_sample(
    args: argparse.Namespace,
    sample: dict[str, Any],
    samples_path: Path,
    teacher_hints: dict[str, str],
) -> dict[str, Any]:
    image_path = resolve_image_path(sample["image_path"], samples_path)
    teacher_hint = teacher_hints.get(sample["id"], "") if args.teacher_mode == "visual_hint" else ""
    prompt = build_prompt(sample, teacher_visual_hint=teacher_hint)
    try:
        latency, result = call_ollama(args, prompt, image_path)
        ok = True
        error = None
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        latency = 0.0
        result = {}
        ok = False
        error = repr(exc)

    message = result.get("message") if isinstance(result, dict) else None
    raw_answer = str((message or {}).get("content") or result.get("response") or "")
    parsed = parse_prediction(raw_answer, sample["options"]) if ok else {}
    pred = parsed.get("final_answer_index")
    valid = isinstance(pred, int) and 0 <= pred < len(sample["options"]) and bool(parsed.get("option_scores"))
    return {
        "idx": sample["idx"],
        "sample_id": sample["id"],
        "agent": "gemma4-option-verify",
        "model": args.model,
        "teacher_mode": args.teacher_mode,
        "answer_type": "CLOSED",
        "question": sample["question"],
        "options": sample["options"],
        "gold": sample["gold"],
        "gold_text": sample["gold_text"],
        "image_path": str(image_path),
        "ok": ok,
        "latency_sec": round(latency, 3),
        "teacher_visual_hint": teacher_hint,
        "raw_answer_text": raw_answer,
        "response_json": result,
        "visual_inventory": parsed.get("visual_inventory") or [],
        "option_scores": parsed.get("option_scores") or [],
        "final_answer_index": pred,
        "final_answer": parsed.get("final_answer"),
        "confidence": parsed.get("confidence"),
        "rationale": parsed.get("rationale"),
        "valid_prediction": valid,
        "correct": pred == sample["gold"] if valid else False,
        "error": error,
    }


def summarize(rows: list[dict[str, Any]], out_jsonl: Path) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid_prediction")]
    correct = [row for row in valid if row.get("correct")]
    return {
        "requested_samples": len(rows),
        "completed": len(valid),
        "invalid_predictions": len(rows) - len(valid),
        "accuracy": round(len(correct) / len(valid), 4) if valid else None,
        "out_jsonl": str(out_jsonl),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-json", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--teacher-mode", choices=["none", "visual_hint"], default="none")
    parser.add_argument("--teacher-jsonl")
    parser.add_argument("--out-jsonl", required=True)
    args = parser.parse_args()

    samples_path = Path(args.samples_json)
    samples = load_samples(samples_path, args.start_index, args.max_samples)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    teacher_path = Path(args.teacher_jsonl) if args.teacher_jsonl else None
    teacher_hints = load_teacher_visual_hints(teacher_path)

    rows = []
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for sample in samples:
            row = run_sample(args, sample, samples_path, teacher_hints)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[progress] {len(rows)}/{len(samples)} valid={sum(1 for r in rows if r.get('valid_prediction'))} "
                f"correct={sum(1 for r in rows if r.get('correct'))} invalid={sum(1 for r in rows if not r.get('valid_prediction'))}",
                flush=True,
            )

    print(json.dumps(summarize(rows, out_jsonl), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
