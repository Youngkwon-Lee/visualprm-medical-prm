#!/usr/bin/env python3
"""Rerank OpenClaw candidate answers with MedVisualPRM Branch B.

The Hugging Face repo contains the official Branch B scoring script. This
wrapper downloads/imports that script, loads the PRM once, and scores every
valid OpenClaw attempt as a candidate reasoning step.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


DEFAULT_MODEL_ID = "SNUH-C/medvisualprm-branch-b"
DEFAULT_BASE_MODEL = "google/gemma-4-E4B-it"
OUT_DIR = Path("/Users/youngkwon/projects/visualprm_openclaw_harness/results_native_openclaw")
DEFAULT_SOURCE_JSONL = OUT_DIR / "pathvqa_pathvqa-web_majority_0_10_1778377889.jsonl"
DEFAULT_CACHE_DIR = Path("/Users/youngkwon/projects/visualprm_openclaw_harness/native_openclaw_cache")


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def download_model_package(model_id: str, cache_dir: Path, token: str | None) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    local_dir = cache_dir / model_id.replace("/", "__")
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        return Path(
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                token=token,
                allow_patterns=[
                    "*.json",
                    "*.py",
                "*.pt",
                "*.safetensors",
                "*.jinja",
                "tokenizer*",
                "processor*",
                "README.md",
                ],
            )
        )
    except RepositoryNotFoundError as exc:
        raise RuntimeError(
            f"Cannot access {model_id}. The repo is private/gated or the current HF token "
            "does not include read permission for this repo."
        ) from exc


def import_branch_b_script(model_dir: Path) -> Any:
    script_path = model_dir / "run_branch_b_prm.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Branch B runner not found: {script_path}")

    spec = importlib.util.spec_from_file_location("medvisualprm_branch_b_runner", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_device(requested: str) -> tuple[str, bool]:
    if requested != "auto":
        return requested, requested == "mps"

    import torch

    if torch.cuda.is_available():
        return "cuda", False
    if torch.backends.mps.is_available():
        return "mps", True
    return "cpu", False


def build_prm_args(args: argparse.Namespace, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        base_model=args.base_model,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        device=device,
        bf16=args.bf16,
        threshold=args.threshold,
        max_length=args.max_length,
        trust_remote_code=True,
        print_prompt=args.print_prompt,
    )


def move_to_mps_if_needed(model: Any, classifier: Any, force_mps: bool) -> None:
    if not force_mps:
        return
    import torch

    if not torch.backends.mps.is_available():
        return
    model.to("mps").eval()
    classifier.to("mps").eval()


def candidate_step(attempt: dict[str, Any], options: list[str]) -> str:
    raw_text = str(attempt.get("raw_answer_text") or "")
    parsed = extract_json_object(raw_text)
    answer_index = attempt.get("final_answer_index")
    answer = attempt.get("final_answer") or parsed.get("final_answer")
    if answer is None and isinstance(answer_index, int) and 0 <= answer_index < len(options):
        answer = options[answer_index]

    rationale = parsed.get("rationale")
    if not rationale:
        rationale = raw_text.strip()
    rationale = re.sub(r"\s+", " ", str(rationale)).strip()
    answer_text = f"choice {answer_index}: {answer}" if answer_index is not None else str(answer)
    return f"The candidate selects {answer_text}. Rationale: {rationale}"


def valid_attempts(row: dict[str, Any], max_candidates: int) -> list[dict[str, Any]]:
    attempts = [
        attempt
        for attempt in row.get("attempts", [])
        if attempt.get("ok") and attempt.get("valid_prediction") and attempt.get("final_answer_index") is not None
    ]
    if max_candidates > 0:
        attempts = attempts[:max_candidates]
    return attempts


def score_row(
    row: dict[str, Any],
    *,
    runner: Any,
    processor: Any,
    model: Any,
    classifier: Any,
    token_ids: dict[str, int],
    prm_args: SimpleNamespace,
    max_candidates: int,
    answer_aggregation: str,
) -> dict[str, Any]:
    choices = [str(choice) for choice in row.get("options", [])]
    candidates = []
    for attempt in valid_attempts(row, max_candidates):
        example = {
            "image": row["image_path"],
            "question": row["question"],
            "choices": choices,
            "steps": [candidate_step(attempt, choices)],
        }
        started = time.time()
        result = runner.score_example(example, processor, model, classifier, token_ids, prm_args)
        score = float(result["score"])
        candidates.append(
            {
                "attempt_index": attempt.get("attempt_index"),
                "final_answer_index": attempt.get("final_answer_index"),
                "final_answer": attempt.get("final_answer"),
                "openclaw_confidence": attempt.get("confidence_float"),
                "prm_score": score,
                "latency_sec": round(time.time() - started, 3),
                "step": example["steps"][0],
            }
        )

    selected = select_candidate(candidates, answer_aggregation)
    pred = selected.get("final_answer_index") if selected else None
    valid_prediction = isinstance(pred, int) and 0 <= pred < len(choices)
    return {
        "idx": row.get("idx"),
        "sample_id": row.get("sample_id"),
        "question": row.get("question"),
        "options": choices,
        "gold": row.get("gold"),
        "image_path": row.get("image_path"),
        "majority_final_answer_index": row.get("final_answer_index"),
        "majority_correct": row.get("correct"),
        "prm_final_answer_index": pred,
        "prm_final_answer": selected.get("final_answer") if selected else None,
        "prm_score": selected.get("prm_score") if selected else None,
        "answer_aggregation": answer_aggregation,
        "answer_scores": aggregate_answer_scores(candidates, answer_aggregation),
        "valid_prediction": valid_prediction,
        "correct": pred == row.get("gold") if valid_prediction else False,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def aggregate_answer_scores(candidates: list[dict[str, Any]], answer_aggregation: str) -> dict[str, float]:
    grouped: dict[int, list[float]] = {}
    for candidate in candidates:
        answer_index = candidate.get("final_answer_index")
        if not isinstance(answer_index, int):
            continue
        grouped.setdefault(answer_index, []).append(float(candidate["prm_score"]))

    scores: dict[str, float] = {}
    for answer_index, values in grouped.items():
        if answer_aggregation == "max":
            score = max(values)
        elif answer_aggregation == "mean":
            score = sum(values) / len(values)
        elif answer_aggregation == "sum":
            score = sum(values)
        elif answer_aggregation == "sum_sqrt":
            score = sum(values) / (len(values) ** 0.5)
        else:
            raise ValueError(f"Unknown answer aggregation: {answer_aggregation}")
        scores[str(answer_index)] = float(score)
    return scores


def select_candidate(candidates: list[dict[str, Any]], answer_aggregation: str) -> dict[str, Any] | None:
    if not candidates:
        return None

    answer_scores = aggregate_answer_scores(candidates, answer_aggregation)
    selected_answer = int(max(answer_scores, key=lambda key: answer_scores[key]))
    return max(
        (candidate for candidate in candidates if candidate.get("final_answer_index") == selected_answer),
        key=lambda candidate: float(candidate["prm_score"]),
    )


def summarize(rows: list[dict[str, Any]], out_jsonl: Path) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid_prediction")]
    correct = [row for row in valid if row.get("correct")]
    majority_valid = [row for row in rows if row.get("majority_final_answer_index") is not None]
    majority_correct = [row for row in majority_valid if row.get("majority_correct")]
    return {
        "requested_samples": len(rows),
        "scored_samples": len(valid),
        "invalid_samples": len(rows) - len(valid),
        "prm_accuracy": round(len(correct) / len(valid), 4) if valid else None,
        "majority_accuracy_on_same_rows": round(len(majority_correct) / len(majority_valid), 4)
        if majority_valid
        else None,
        "out_jsonl": str(out_jsonl),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-jsonl", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-candidates-per-sample", type=int, default=0)
    parser.add_argument(
        "--answer-aggregation",
        choices=["max", "mean", "sum", "sum_sqrt"],
        default="max",
        help="How to aggregate PRM scores for candidates that select the same answer.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--print-prompt", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build candidate steps without loading the PRM.")
    args = parser.parse_args()

    if not args.source_jsonl.exists():
        raise FileNotFoundError(args.source_jsonl)

    rows = read_jsonl(args.source_jsonl, args.max_samples)
    out_jsonl = args.out_jsonl or (
        OUT_DIR / f"medvisualprm_rerank_{args.source_jsonl.stem}_{int(time.time())}.jsonl"
    )
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        preview_rows = []
        for row in rows:
            choices = [str(choice) for choice in row.get("options", [])]
            preview_rows.append(
                {
                    "idx": row.get("idx"),
                    "sample_id": row.get("sample_id"),
                    "candidate_steps": [
                        candidate_step(attempt, choices)
                        for attempt in valid_attempts(row, args.max_candidates_per_sample)
                    ],
                }
            )
        out_jsonl.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in preview_rows) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"dry_run": True, "rows": len(preview_rows), "out_jsonl": str(out_jsonl)}, indent=2))
        return 0

    token = args.hf_token or os.environ.get("HF_TOKEN")
    model_dir = download_model_package(args.model_id, args.cache_dir, token)
    if args.download_only:
        print(json.dumps({"model_dir": str(model_dir)}, ensure_ascii=False, indent=2))
        return 0

    runner = import_branch_b_script(model_dir)
    device, force_mps = resolve_device(args.device)
    prm_args = build_prm_args(args, device)
    processor, model, classifier, token_ids = runner.load_model(str(model_dir), prm_args)
    move_to_mps_if_needed(model, classifier, force_mps)

    scored_rows = []
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            scored = score_row(
                row,
                runner=runner,
                processor=processor,
                model=model,
                classifier=classifier,
                token_ids=token_ids,
                prm_args=prm_args,
                max_candidates=args.max_candidates_per_sample,
                answer_aggregation=args.answer_aggregation,
            )
            scored_rows.append(scored)
            f.write(json.dumps(scored, ensure_ascii=False) + "\n")
            f.flush()

    print(json.dumps(summarize(scored_rows, out_jsonl), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1)
