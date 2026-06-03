#!/usr/bin/env python3
"""Two-stage VQA: vision-only descriptor subactor plus text matcher.

The descriptor sees the image and clinical stem but not answer options. The
matcher sees only the descriptor, question, and options. This tests whether
separating visual evidence extraction from option selection helps low-B actors.
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


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


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


def build_descriptor_prompt(sample: dict[str, Any]) -> str:
    return (
        "You are a medical visual descriptor. Your job is only to describe the image. "
        "Do not answer the question. Do not infer or mention any answer option. "
        "Do not use web search, filenames, benchmark memory, or dataset lookup. "
        "Return exactly one compact JSON object and no markdown.\n"
        "Describe concrete visible findings: organ/site, specimen type if apparent, color, texture, lesion shape, distribution, size pattern, necrosis/hemorrhage/inflammation/tumor-like features, and any important absence. "
        "If clinical context helps decide what to inspect, use it only to guide observation, not to diagnose.\n"
        "JSON schema: {\"sample_id\": string, \"organ_or_site\": string, \"specimen_type\": string, \"visual_findings\": string[], \"important_absences\": string[], \"uncertainties\": string[]}\n"
        f"Clinical question stem without answer options: {sample['question']}"
    )


def build_matcher_prompt(sample: dict[str, Any], descriptor: dict[str, Any]) -> str:
    options = "\n".join(f"{idx}: {option}" for idx, option in enumerate(sample["options"]))
    descriptor_json = json.dumps(
        {
            "organ_or_site": descriptor.get("organ_or_site"),
            "specimen_type": descriptor.get("specimen_type"),
            "visual_findings": descriptor.get("visual_findings") or [],
            "important_absences": descriptor.get("important_absences") or [],
            "uncertainties": descriptor.get("uncertainties") or [],
        },
        ensure_ascii=False,
    )
    return (
        "You are a medical VQA option matcher. You cannot see the image. "
        "Use only the visual descriptor, clinical question, answer options, and general medical knowledge. "
        "Do not use web search, filenames, benchmark memory, or dataset lookup. "
        "Return exactly one compact JSON object and no markdown.\n"
        "Scoring rule: each option must be judged by whether the descriptor contains direct visual support, direct visual mismatch, or only clinical/general plausibility. "
        "Do not give a high score to an option that is plausible from the stem but lacks matching visual morphology.\n"
        "JSON schema: {\"sample_id\": string, \"option_scores\": [{\"index\": int, \"option\": string, \"descriptor_support\": string, \"descriptor_mismatch\": string, \"score\": int}], \"final_answer_index\": int, \"final_answer\": string, \"confidence\": number, \"rationale\": string}\n"
        f"Question: {sample['question']}\n"
        f"Visual descriptor: {descriptor_json}\n"
        f"Options:\n{options}\n"
        f"final_answer_index must be between 0 and {len(sample['options']) - 1}."
    )


def call_ollama(
    *,
    base_url: str,
    model: str,
    prompt: str,
    image_path: Path | None,
    timeout: int,
    num_ctx: int,
    num_predict: int,
    temperature: float,
    keep_alive: str,
) -> tuple[float, dict[str, Any]]:
    message: dict[str, Any] = {"role": "user", "content": prompt}
    if image_path is not None:
        message["images"] = [base64.b64encode(image_path.read_bytes()).decode("ascii")]
    payload = {
        "model": model,
        "messages": [message],
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    return time.time() - started, result


def response_text(result: dict[str, Any]) -> str:
    message = result.get("message") if isinstance(result, dict) else None
    return str((message or {}).get("content") or result.get("response") or "")


def parse_descriptor(raw_answer: str) -> dict[str, Any]:
    parsed = parse_json_object(raw_answer)
    for key in ("visual_findings", "important_absences", "uncertainties"):
        value = parsed.get(key) or []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            value = []
        parsed[key] = [normalize_text(item) for item in value if normalize_text(item)]
    parsed["organ_or_site"] = normalize_text(parsed.get("organ_or_site"))
    parsed["specimen_type"] = normalize_text(parsed.get("specimen_type"))
    return parsed


def parse_matcher(raw_answer: str, options: list[str]) -> dict[str, Any]:
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

    option_scores = parsed.get("option_scores") or []
    if not isinstance(option_scores, list):
        option_scores = []
    return {
        **parsed,
        "option_scores": option_scores,
        "final_answer_index": pred,
        "final_answer": answer_text or (options[pred] if isinstance(pred, int) and 0 <= pred < len(options) else None),
        "rationale": normalize_text(parsed.get("rationale")),
    }


def run_sample(args: argparse.Namespace, sample: dict[str, Any], samples_path: Path) -> dict[str, Any]:
    image_path = resolve_image_path(sample["image_path"], samples_path)
    try:
        descriptor_latency, descriptor_result = call_ollama(
            base_url=args.base_url,
            model=args.descriptor_model,
            prompt=build_descriptor_prompt(sample),
            image_path=image_path,
            timeout=args.timeout,
            num_ctx=args.num_ctx,
            num_predict=args.descriptor_num_predict,
            temperature=args.temperature,
            keep_alive=args.keep_alive,
        )
        descriptor_ok = True
        descriptor_error = None
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        descriptor_latency = 0.0
        descriptor_result = {}
        descriptor_ok = False
        descriptor_error = repr(exc)

    descriptor_raw = response_text(descriptor_result)
    descriptor = parse_descriptor(descriptor_raw) if descriptor_ok else {}

    try:
        matcher_latency, matcher_result = call_ollama(
            base_url=args.base_url,
            model=args.matcher_model,
            prompt=build_matcher_prompt(sample, descriptor),
            image_path=None,
            timeout=args.timeout,
            num_ctx=args.num_ctx,
            num_predict=args.matcher_num_predict,
            temperature=args.temperature,
            keep_alive=args.keep_alive,
        )
        matcher_ok = True
        matcher_error = None
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        matcher_latency = 0.0
        matcher_result = {}
        matcher_ok = False
        matcher_error = repr(exc)

    matcher_raw = response_text(matcher_result)
    matcher = parse_matcher(matcher_raw, sample["options"]) if matcher_ok else {}
    pred = matcher.get("final_answer_index")
    valid = (
        descriptor_ok
        and matcher_ok
        and isinstance(pred, int)
        and 0 <= pred < len(sample["options"])
        and bool(descriptor.get("visual_findings"))
        and bool(matcher.get("option_scores"))
    )
    return {
        "idx": sample["idx"],
        "sample_id": sample["id"],
        "agent": "gemma4-descriptor-text-matcher",
        "descriptor_model": args.descriptor_model,
        "matcher_model": args.matcher_model,
        "answer_type": "CLOSED",
        "question": sample["question"],
        "options": sample["options"],
        "gold": sample["gold"],
        "gold_text": sample["gold_text"],
        "image_path": str(image_path),
        "descriptor_ok": descriptor_ok,
        "descriptor_latency_sec": round(descriptor_latency, 3),
        "descriptor_raw_answer_text": descriptor_raw,
        "descriptor": descriptor,
        "descriptor_error": descriptor_error,
        "matcher_ok": matcher_ok,
        "matcher_latency_sec": round(matcher_latency, 3),
        "matcher_raw_answer_text": matcher_raw,
        "matcher_response_json": matcher_result,
        "option_scores": matcher.get("option_scores") or [],
        "final_answer_index": pred,
        "final_answer": matcher.get("final_answer"),
        "confidence": matcher.get("confidence"),
        "rationale": matcher.get("rationale"),
        "valid_prediction": valid,
        "correct": pred == sample["gold"] if valid else False,
        "matcher_error": matcher_error,
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
    parser.add_argument("--descriptor-model", default="gemma4:e4b")
    parser.add_argument("--matcher-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--descriptor-num-predict", type=int, default=768)
    parser.add_argument("--matcher-num-predict", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--out-jsonl", required=True)
    args = parser.parse_args()

    samples_path = Path(args.samples_json)
    samples = load_samples(samples_path, args.start_index, args.max_samples)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for sample in samples:
            row = run_sample(args, sample, samples_path)
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
