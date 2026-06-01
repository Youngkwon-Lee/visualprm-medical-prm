#!/usr/bin/env python3
"""Fast direct Ollama multimodal VQA runner for medical visual benchmarks."""

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
                "image_path": str(row.get("image_path") or row.get("image_url") or ""),
            }
        )
    return samples


def resolve_image_path(raw_path: str, samples_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, samples_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_prompt(sample: dict[str, Any], rationale: bool) -> str:
    options = "\n".join(f"{idx}: {option}" for idx, option in enumerate(sample["options"]))
    rationale_key = ', "rationale": "<short visual reason>"' if rationale else ""
    return (
        "Inspect the medical image and answer the multiple-choice question. "
        "Use the image evidence, not dataset memorization. "
        f"Return only compact JSON: {{\"final_answer_index\": <integer>, \"final_answer\": \"<option text>\"{rationale_key}}}.\n"
        f"Question: {sample['question']}\n"
        f"Options:\n{options}\n"
        f"The final_answer_index must be between 0 and {len(sample['options']) - 1}."
    )


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_prediction(text: str, options: list[str]) -> dict[str, Any]:
    stripped = strip_code_fence(text)
    parsed: dict[str, Any] = {}
    for candidate in (stripped,):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed = value
            break
    if not parsed:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if match:
            try:
                value = json.loads(match.group(0))
                if isinstance(value, dict):
                    parsed = value
            except json.JSONDecodeError:
                pass

    pred = parsed.get("final_answer_index")
    if isinstance(pred, str) and pred.strip().isdigit():
        pred = int(pred.strip())
    if not isinstance(pred, int):
        match = re.search(r"final_answer_index\D+(\d+)", stripped, flags=re.I)
        if match:
            pred = int(match.group(1))

    answer_text = str(parsed.get("final_answer") or "").strip()
    if not isinstance(pred, int) and answer_text:
        normalized_answer = answer_text.lower()
        for idx, option in enumerate(options):
            if normalized_answer == option.lower() or normalized_answer in option.lower() or option.lower() in normalized_answer:
                pred = idx
                break

    return {
        **parsed,
        "final_answer_index": pred,
        "final_answer": answer_text or (options[pred] if isinstance(pred, int) and 0 <= pred < len(options) else None),
        "parse_fallback": not bool(parsed),
    }


def call_ollama(args: argparse.Namespace, prompt: str, image_path: Path) -> tuple[float, dict[str, Any]]:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": args.model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "keep_alive": args.keep_alive,
        "options": {
            "temperature": args.temperature,
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        result = json.loads(response.read())
    return time.time() - started, result


def run_sample(sample: dict[str, Any], samples_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    image_path = resolve_image_path(sample["image_path"], samples_path)
    row = {
        "idx": sample["idx"],
        "sample_id": sample["id"],
        "model": args.model,
        "question": sample["question"],
        "options": sample["options"],
        "gold": sample["gold"],
        "image_path": str(image_path),
        "ok": False,
    }
    if not image_path.exists() or image_path.stat().st_size == 0:
        row["error"] = f"missing image: {image_path}"
        return row

    prompt = build_prompt(sample, args.rationale)
    try:
        latency, result = call_ollama(args, prompt, image_path)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        row["error"] = repr(exc)
        return row

    raw_answer = str(result.get("response") or "")
    parsed = parse_prediction(raw_answer, sample["options"])
    pred = parsed.get("final_answer_index")
    valid = isinstance(pred, int) and 0 <= pred < len(sample["options"])
    row.update(
        {
            "ok": True,
            "latency_sec": round(latency, 3),
            "ollama_total_sec": round(float(result.get("total_duration") or 0) / 1e9, 3),
            "prompt_eval_count": result.get("prompt_eval_count"),
            "eval_count": result.get("eval_count"),
            "raw_answer_text": raw_answer,
            "valid_prediction": valid,
            "correct": pred == sample["gold"] if valid else False,
        }
    )
    row.update(parsed)
    return row


def summarize(rows: list[dict[str, Any]], out_jsonl: Path) -> dict[str, Any]:
    valid = [row for row in rows if row.get("ok") and row.get("valid_prediction")]
    correct = [row for row in valid if row.get("correct")]
    latencies = [float(row["latency_sec"]) for row in rows if row.get("latency_sec") is not None]
    return {
        "requested_samples": len(rows),
        "completed": len(valid),
        "errors": sum(1 for row in rows if not row.get("ok")),
        "invalid_predictions": sum(1 for row in rows if row.get("ok") and not row.get("valid_prediction")),
        "accuracy": round(len(correct) / len(valid), 4) if valid else None,
        "avg_latency_sec": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "out_jsonl": str(out_jsonl),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-json", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--num-predict", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--rationale", action="store_true")
    parser.add_argument("--out-jsonl", required=True)
    args = parser.parse_args()

    samples_path = Path(args.samples_json)
    samples = load_samples(samples_path, args.start_index, args.max_samples)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for sample in samples:
            row = run_sample(sample, samples_path, args)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    print(json.dumps(summarize(rows, out_jsonl), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
