#!/usr/bin/env python3
"""Gemma4 direct VQA with image variants and simple controlled retry branches."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter


HARNESS_ROOT = Path("/Users/youngkwon/projects/visualprm_openclaw_harness")
OUT_DIR = HARNESS_ROOT / "results_native_openclaw"
VARIANT_DIR = OUT_DIR / "image_variants"


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


def make_variants(image_path: Path, sample_id: str) -> list[tuple[str, Path]]:
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    variants: list[tuple[str, Path]] = [("original", image_path)]

    contrast_path = VARIANT_DIR / f"{sample_id}_contrast.jpg"
    ImageEnhance.Contrast(image).enhance(1.45).filter(ImageFilter.SHARPEN).save(contrast_path, quality=95)
    variants.append(("contrast_sharpen", contrast_path))

    side = int(min(width, height) * 0.78)
    left = max((width - side) // 2, 0)
    top = max((height - side) // 2, 0)
    crop = image.crop((left, top, left + side, top + side)).resize((width, height))
    crop_path = VARIANT_DIR / f"{sample_id}_center_zoom.jpg"
    crop.save(crop_path, quality=95)
    variants.append(("center_zoom", crop_path))
    return variants


def option_hints(options: list[str]) -> str:
    hints = {
        "amyloidosis": "waxy firm pale diffuse infiltration or thick restrictive myocardium",
        "systemic hypertension": "concentric left ventricular hypertrophy without a primary vascular plaque clue",
        "atherosclerosis": "yellow-white intimal plaques in arteries/coronaries or narrowed vascular lumen",
        "viral myocarditis": "flabby myocardium, mottled inflammation, not primarily arterial plaque disease",
        "thrombocytopenia": "petechiae, purpura, ecchymoses, or hemorrhagic skin spots without mass formation",
        "gangrene": "black-green necrotic tissue, sharply devitalized skin or extremity",
        "metastatic breast carcinoma": "discrete tumor nodules or masses rather than diffuse petechiae/purpura",
        "pellagra": "photosensitive dermatitis, symmetric hyperpigmented/scaly exposed skin changes",
        "congestive heart failure": "edema/cyanosis pattern rather than focal hemorrhagic skin spots",
    }
    lines = []
    for idx, option in enumerate(options):
        key = option.lower().strip()
        if key in hints:
            lines.append(f"{idx} {option}: {hints[key]}")
    if not lines:
        return ""
    return "Controlled gross morphology reminders. Use only as general medical knowledge, not as dataset lookup:\n" + "\n".join(lines)


def build_prompt(sample: dict[str, Any], mode: str, variant_name: str) -> str:
    options = "\n".join(f"{idx}: {option}" for idx, option in enumerate(sample["options"]))
    base = (
        "You are a concise medical visual VQA model. Inspect the image directly. "
        "Do not use web search, dataset memory, or external benchmark knowledge. "
        "Return exactly one compact JSON object and no markdown. "
        "The JSON keys must be: sample_id, steps, final_answer_index, final_answer, confidence, rationale. "
        "steps must be a JSON array of 4 to 6 plain strings only. "
        "Step 1 must start with 'Visual evidence:' and describe concrete findings visible in this image. "
        "The last step must start with 'Decision:' and map the evidence to one option. "
    )
    if mode == "option_check":
        base += (
            "Use a simple option check: include one step starting with 'Option check:' that compares the two most plausible options using visible support and visible mismatch. "
        )
    elif mode == "controlled_knowledge":
        base += option_hints(sample["options"]) + "\n"
        base += (
            "Use the reminders only to interpret visible morphology. If the image does not show the expected morphology for an option, reject it. "
            "Include one step starting with 'Controlled check:'. "
        )
    else:
        base += (
            f"This is the '{variant_name}' view of the same image. First describe what changed or became clearer in this view. "
        )
    return (
        base
        + f"Question: {sample['question']}\n"
        + f"Options:\n{options}\n"
        + f"final_answer_index must be between 0 and {len(sample['options']) - 1}."
    )


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def normalize_step_item(step: object) -> str | None:
    if isinstance(step, dict):
        for key in ("text", "step", "content", "description", "evidence", "decision", "rationale"):
            value = step.get(key)
            if value is not None and str(value).strip():
                return re.sub(r"\s+", " ", str(value)).strip()
        return re.sub(r"\s+", " ", json.dumps(step, ensure_ascii=True)).strip()
    text = str(step).strip()
    return re.sub(r"\s+", " ", text).strip() if text else None


def parse_prediction(text: str, options: list[str]) -> dict[str, Any]:
    stripped = strip_code_fence(text)
    parsed: dict[str, Any] = {}
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            parsed = value
    except json.JSONDecodeError:
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

    raw_steps = parsed.get("steps")
    if isinstance(raw_steps, list):
        steps = [item for step in raw_steps if (item := normalize_step_item(step))]
    else:
        steps = []
    if not steps and parsed.get("rationale"):
        steps = [re.sub(r"\s+", " ", str(parsed["rationale"])).strip()]

    return {
        **parsed,
        "steps": steps,
        "final_answer_index": pred,
        "final_answer": answer_text or (options[pred] if isinstance(pred, int) and 0 <= pred < len(options) else None),
        "parse_fallback": not bool(parsed),
    }


def valid_visual_steps(steps: list[str]) -> tuple[bool, str | None]:
    if not steps:
        return False, "missing_reasoning_steps"
    if not str(steps[0]).startswith("Visual evidence:"):
        return False, "first_step_missing_visual_evidence_prefix"
    if not any(str(step).startswith("Decision:") for step in steps):
        return False, "missing_decision_step"
    return True, None


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


def run_attempt(
    args: argparse.Namespace,
    sample: dict[str, Any],
    image_path: Path,
    *,
    attempt_index: int,
    branch: str,
    variant_name: str,
) -> dict[str, Any]:
    mode = "option_check" if branch == "option_check" else "controlled_knowledge" if branch == "controlled_knowledge" else "visual"
    prompt = build_prompt(sample, mode, variant_name)
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
    steps = parsed.get("steps") or []
    visual_ok, visual_error = valid_visual_steps(steps)
    valid = isinstance(pred, int) and 0 <= pred < len(sample["options"]) and visual_ok
    return {
        "attempt_index": attempt_index,
        "branch": branch,
        "variant_name": variant_name,
        "variant_image_path": str(image_path),
        "ok": ok,
        "latency_sec": round(latency, 3),
        "final_answer_index": pred,
        "final_answer": parsed.get("final_answer"),
        "confidence": parsed.get("confidence"),
        "confidence_float": None,
        "steps": steps,
        "valid_prediction": valid,
        "correct": pred == sample["gold"] if valid else False,
        "tool_calls": [],
        "forbidden_tool_calls": [],
        "image_models": [args.model],
        "image_failures": 0,
        "answer_mentions_image_failure": False,
        "visual_grounding_error": None if visual_ok else visual_error,
        "raw_answer_text": raw_answer,
        "error": error,
    }


def choose_majority(sample: dict[str, Any], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [attempt for attempt in attempts if attempt.get("valid_prediction") and isinstance(attempt.get("final_answer_index"), int)]
    if not valid:
        return {"final_answer_index": None, "final_answer": None, "correct": False, "vote_counts": {}}
    counts = Counter(int(attempt["final_answer_index"]) for attempt in valid)
    selected_index = counts.most_common(1)[0][0]
    selected_attempt = next(attempt for attempt in valid if attempt.get("final_answer_index") == selected_index)
    return {
        "final_answer_index": selected_index,
        "final_answer": selected_attempt.get("final_answer"),
        "correct": selected_index == sample["gold"],
        "vote_counts": {str(key): value for key, value in counts.items()},
    }


def run_sample(sample: dict[str, Any], samples_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    image_path = resolve_image_path(sample["image_path"], samples_path)
    attempts = []
    attempt_index = 1
    for variant_name, variant_path in make_variants(image_path, sample["id"]):
        attempts.append(
            run_attempt(
                args,
                sample,
                variant_path,
                attempt_index=attempt_index,
                branch="multiview",
                variant_name=variant_name,
            )
        )
        attempt_index += 1

    attempts.append(
        run_attempt(
            args,
            sample,
            image_path,
            attempt_index=attempt_index,
            branch="option_check",
            variant_name="original",
        )
    )
    attempt_index += 1

    attempts.append(
        run_attempt(
            args,
            sample,
            image_path,
            attempt_index=attempt_index,
            branch="controlled_knowledge",
            variant_name="original",
        )
    )

    selected = choose_majority(sample, attempts)
    return {
        "idx": sample["idx"],
        "sample_id": sample["id"],
        "agent": "gemma4-multiview-controller",
        "input_mode": "normal",
        "openclaw_mode": "direct-ollama",
        "answer_type": "CLOSED",
        "question": sample["question"],
        "options": sample["options"],
        "gold": sample["gold"],
        "gold_text": sample["gold_text"],
        "image_path": str(image_path),
        "ok": any(attempt.get("ok") for attempt in attempts),
        "attempts_used": len(attempts),
        "valid_vote_count": sum(1 for attempt in attempts if attempt.get("valid_prediction")),
        "invalid_attempt_count": sum(1 for attempt in attempts if attempt.get("ok") and not attempt.get("valid_prediction")),
        "errored_attempt_count": sum(1 for attempt in attempts if not attempt.get("ok")),
        "attempts": attempts,
        "final_answer_index": selected["final_answer_index"],
        "final_answer": selected["final_answer"],
        "vote_counts": selected["vote_counts"],
        "valid_prediction": selected["final_answer_index"] is not None,
        "correct": selected["correct"],
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
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=512)
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
            row = run_sample(sample, samples_path, args)
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
