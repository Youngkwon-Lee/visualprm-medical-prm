#!/usr/bin/env python3
"""Evaluate PathVQA with native OpenClaw only, without the VisualPRM/RAG tool."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


REMOTE = "82106@100.83.147.56"
REMOTE_ROOT = "/mnt/d/visualprm"
DEFAULT_AGENT = "pathvqa-native"
HARNESS_ROOT = Path(os.environ.get("VISUALPRM_HARNESS_ROOT", Path(__file__).resolve().parent))
# Do not use OPENCLAW_HOME for harness paths. The OpenClaw CLI interprets that
# variable as a state root and then looks under "$OPENCLAW_HOME/.openclaw".
OPENCLAW_STATE_DIR = Path(
    os.environ.get("VISUALPRM_OPENCLAW_STATE_DIR")
    or os.environ.get("OPENCLAW_STATE_DIR")
    or (Path.home() / ".openclaw")
)
OPENCLAW_CONFIG_PATH = Path(
    os.environ.get("OPENCLAW_CONFIG_PATH", OPENCLAW_STATE_DIR / "openclaw.json")
)
OPENCLAW_WORKSPACE_DIR = Path(os.environ.get("OPENCLAW_WORKSPACE", OPENCLAW_STATE_DIR / "workspace"))
OPENCLAW_AGENTS_DIR = Path(os.environ.get("OPENCLAW_AGENTS_DIR", OPENCLAW_STATE_DIR / "agents"))
WORKSPACE_IMAGE_DIR = Path(os.environ.get("OPENCLAW_IMAGE_WORKSPACE", OPENCLAW_WORKSPACE_DIR / "pathvqa_images"))
OUT_DIR = Path(os.environ.get("VISUALPRM_OUT_DIR", HARNESS_ROOT / "results_native_openclaw"))


DATASET_FILES = {
    "pathvqa": "pathvqa_test_for_app.json",
    "pathvqa_test": "pathvqa_test_for_app.json",
    "pathvqa_full": "pathvqa_for_app.json",
}

LOCAL_SAMPLE_FALLBACKS = [
    OUT_DIR / "pathvqa_pathvqa-web_majority_0_10_1778377889.jsonl",
    OUT_DIR / "pathvqa_pathvqa-web_native_openclaw_0_20_1778332775.jsonl",
    OUT_DIR / "pathvqa_pathvqa-native_native_openclaw_0_20_1778330377.jsonl",
]


def run(
    args: list[str],
    *,
    timeout: int = 60,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, check=False, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args,
            124,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else f"timeout after {timeout}s",
        )


def run_ssh(args: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", REMOTE, *args], timeout=timeout)


def session_file_for(agent: str, session_id: str) -> Path:
    return OPENCLAW_AGENTS_DIR / agent / "sessions" / f"{session_id}.jsonl"


def allowed_tools_for(agent: str, input_mode: str = "normal") -> set[str]:
    if input_mode == "question_only":
        if "web" in agent:
            return {"web_search", "web_fetch"}
        return set()
    if "web" in agent:
        return {"image", "web_search", "web_fetch"}
    return {"image"}


def parse_outer_json(output: str) -> dict | None:
    stripped = output.strip()
    candidates = [stripped]
    for marker in ('\n{', '{"payloads"', '{\n  "payloads"', '{\n    "payloads"'):
        start = output.find(marker)
        if start >= 0:
            candidates.append(output[start + (1 if marker.startswith("\n") else 0):].strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def load_local_samples(samples_json: str, start_index: int, max_samples: int) -> list[dict]:
    path = Path(samples_json)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
    samples = []
    for local_offset, sample in enumerate(rows[start_index : start_index + max_samples], start=start_index):
        samples.append(
            {
                "idx": int(sample.get("idx", local_offset)),
                "id": sample.get("id", sample.get("sample_id", f"local_{local_offset}")),
                "question": sample.get("question", ""),
                "options": [str(x) for x in (sample.get("options") or ["Yes", "No"])],
                "gold": int(sample.get("gold", sample.get("answer_index", 0))),
                "image_url": str(sample.get("image_url", sample.get("image_path", ""))),
                "case_type": str(sample.get("case_type", "")),
                "modality": str(sample.get("modality", "")),
            }
        )
    return samples


def load_samples(dataset: str, start_index: int, max_samples: int, samples_json: str = "") -> list[dict]:
    if samples_json:
        return load_local_samples(samples_json, start_index, max_samples)

    remote_dataset = f"{REMOTE_ROOT}/{DATASET_FILES[dataset]}"
    proc = run_ssh(["wsl", "cat", remote_dataset], timeout=60)
    if proc.returncode != 0:
        cached = load_cached_samples(start_index, max_samples)
        if cached:
            return cached
        raise RuntimeError(f"failed to load samples\nstdout={proc.stdout}\nstderr={proc.stderr}")
    rows = json.loads(proc.stdout.lstrip("\ufeff"))
    samples = []
    for idx in range(start_index, min(len(rows), start_index + max_samples)):
        sample = rows[idx]
        samples.append(
            {
                "idx": idx,
                "id": sample.get("id", f"{dataset}_{idx}"),
                "question": sample.get("question", ""),
                "options": [str(x) for x in (sample.get("options") or [])],
                "gold": int(sample.get("gold", sample.get("answer_index", 0))),
                "image_url": str(sample.get("image_url", "")),
                "case_type": str(sample.get("case_type", "")),
                "modality": str(sample.get("modality", "")),
            }
        )
    return samples


def load_cached_samples(start_index: int, max_samples: int) -> list[dict]:
    by_idx: dict[int, dict] = {}
    for path in LOCAL_SAMPLE_FALLBACKS:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["idx"])
            image_path = row.get("image_path") or ""
            image_name = Path(image_path).name if image_path else f"{row['sample_id']}.jpg"
            by_idx[idx] = {
                "idx": idx,
                "id": row.get("sample_id", f"pathvqa_{idx:06d}"),
                "question": row.get("question", ""),
                "options": [str(x) for x in row.get("options", ["Yes", "No"])],
                "gold": int(row.get("gold", 0)),
                "image_url": f"images/pathvqa/{image_name}",
                "case_type": str(row.get("case_type", "")),
                "modality": str(row.get("modality", "")),
            }
    samples = []
    for idx in range(start_index, start_index + max_samples):
        if idx not in by_idx:
            return []
        samples.append(by_idx[idx])
    return samples


def ensure_workspace_image(sample: dict) -> Path:
    WORKSPACE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    source_path = Path(sample["image_url"])
    if source_path.is_absolute() and source_path.exists() and source_path.stat().st_size > 0:
        if str(source_path).startswith(str(OPENCLAW_WORKSPACE_DIR)):
            return source_path
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample.get("id") or "sample"))
        local_path = WORKSPACE_IMAGE_DIR / f"{safe_id}_{source_path.name}"
        if not local_path.exists() or local_path.stat().st_size == 0:
            shutil.copyfile(source_path, local_path)
        return local_path
    filename = Path(sample["image_url"]).name
    local_path = WORKSPACE_IMAGE_DIR / filename
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    remote_image = f"{REMOTE_ROOT}/{sample['image_url'].lstrip('/')}"
    with local_path.open("wb") as out:
        proc = subprocess.run(
            ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", REMOTE, "wsl", "cat", remote_image],
            check=False,
            stdout=out,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to copy image {remote_image}: {proc.stderr.decode('utf-8', errors='replace')}")
    return local_path


def build_prompt(sample: dict, image_path: Path | None, agent: str, input_mode: str) -> str:
    options = "; ".join(f"{i} {option}" for i, option in enumerate(sample["options"]))
    if input_mode == "question_only":
        tool_instruction = (
            "No image is provided for this leakage-control run. "
            "Do not use the image tool. "
            "You may use OpenClaw web_search and web_fetch for general medical background if useful. "
            "Do not use exec, process, memory_get, memory_search, session_status, SSH, files, VisualPRM, RAG, retrieval backends, or custom batch tools. "
        )
        image_instruction = ""
    elif "web" in agent:
        tool_instruction = (
            "Call OpenClaw's native image tool exactly once to inspect the image; do not call the image tool again. "
            "Use web_search or web_fetch only if the image tool result is unavailable or clearly insufficient. "
            "Do not use exec, process, memory_get, memory_search, session_status, SSH, VisualPRM, RAG, retrieval backends, or custom batch tools. "
        )
        image_instruction = f"Image: {image_path}. "
    else:
        tool_instruction = (
            "Do not use exec, process, web_search, web_fetch, memory_get, memory_search, or session_status. "
            "Do not use any VisualPRM, RAG, retrieval, SSH, or custom batch tool. "
            "Call OpenClaw's native image tool exactly once to inspect the image, then answer; do not call the image tool again. "
        )
        image_instruction = f"Image: {image_path}. "
    return (
        "Native OpenClaw PathVQA evaluation. "
        f"{tool_instruction}"
        f"{image_instruction}"
        f"Question: {sample['question']} "
        f"Options: {options}. "
        "Return exactly one compact JSON object and no markdown with keys: "
        "sample_id, final_answer_index, final_answer, confidence, rationale. "
        f"final_answer_index must be one of these option indices: 0 through {len(sample['options']) - 1}. "
        "If uncertain, choose the closest option rather than returning -1 or an out-of-range index."
    )


def extract_json(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    index_match = re.search(r"final_answer_index\D+(\d+)", text, flags=re.I)
    if not index_match:
        index_match = re.search(r"answer(?: is|:)?\D+([01])\b", text, flags=re.I)
    return {
        "final_answer_index": int(index_match.group(1)) if index_match else None,
        "final_answer": None,
        "confidence": None,
        "rationale": text.strip(),
        "parse_fallback": True,
    }


def parse_openclaw_output(output: str, agent: str, session_id: str) -> tuple[str, dict]:
    raw_text = ""
    if output.strip():
        outer = parse_outer_json(output)
        if outer:
            payloads = outer.get("payloads") or []
            if payloads:
                raw_text = "\n".join(payload.get("text", "") for payload in payloads if payload.get("text"))
        else:
            raw_text = output.strip()

    if not raw_text:
        session_file = session_file_for(agent, session_id)
        for line in session_file.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            message = event.get("message") or {}
            if message.get("role") != "assistant":
                continue
            chunks = [item.get("text", "") for item in message.get("content", []) if item.get("type") == "text"]
            if chunks:
                raw_text = "\n".join(chunks)

    return raw_text, extract_json(raw_text)


def inspect_session_tools(agent: str, session_id: str, input_mode: str = "normal") -> dict:
    session_file = session_file_for(agent, session_id)
    tool_calls: list[str] = []
    image_models: list[str] = []
    image_failures = 0
    if not session_file.exists():
        return {"tool_calls": tool_calls, "image_models": image_models, "image_failures": image_failures}

    for line in session_file.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        message = event.get("message") or {}
        for item in message.get("content", []) or []:
            if item.get("type") == "toolCall":
                tool_calls.append(str(item.get("name")))
        if message.get("role") == "toolResult" and message.get("toolName") == "image":
            if message.get("isError"):
                image_failures += 1
            details = message.get("details") or {}
            if details.get("model"):
                image_models.append(str(details["model"]))

    allowed_tools = allowed_tools_for(agent, input_mode)
    forbidden = [name for name in tool_calls if name not in allowed_tools]
    return {
        "tool_calls": tool_calls,
        "image_call_count": tool_calls.count("image"),
        "forbidden_tool_calls": forbidden,
        "image_models": image_models,
        "image_failures": image_failures,
    }


def confidence_as_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            number = float(match.group(0))
            return number / 100 if number > 1 else number
    return None


def run_attempt(
    sample: dict,
    image_path: Path | None,
    *,
    agent: str,
    input_mode: str,
    timeout: int,
    attempt_index: int,
) -> dict:
    session_id = f"native-openclaw-pathvqa-{sample['idx']}-{attempt_index}-{int(time.time() * 1000)}"
    started = time.time()
    openclaw_env = os.environ.copy()
    openclaw_env.pop("OPENCLAW_HOME", None)
    openclaw_env.setdefault("OPENCLAW_STATE_DIR", str(OPENCLAW_STATE_DIR))
    openclaw_env.setdefault("OPENCLAW_CONFIG_PATH", str(OPENCLAW_CONFIG_PATH))
    thinking_args = []
    if os.environ.get("OPENCLAW_THINKING"):
        thinking_args = ["--thinking", os.environ["OPENCLAW_THINKING"]]
    proc = run(
        [
            "openclaw",
            "agent",
            "--local",
            "--agent",
            agent,
            "--session-id",
            session_id,
            "--message",
            build_prompt(sample, image_path, agent, input_mode),
            *thinking_args,
            "--json",
            "--timeout",
            str(timeout),
        ],
        timeout=timeout + 60,
        env=openclaw_env,
    )

    row = {
        "idx": sample["idx"],
        "sample_id": sample["id"],
        "agent": agent,
        "input_mode": input_mode,
        "attempt_index": attempt_index,
        "question": sample["question"],
        "options": sample["options"],
        "gold": sample["gold"],
        "image_path": str(image_path) if image_path else None,
        "session_id": session_id,
        "ok": proc.returncode == 0,
        "latency_sec": round(time.time() - started, 3),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.returncode != 0:
        row["error"] = f"openclaw exited with code {proc.returncode}"
        row.update(inspect_session_tools(agent, session_id, input_mode))
        return row

    raw_text, parsed = parse_openclaw_output(proc.stdout or proc.stderr, agent, session_id)
    raw_text_lower = raw_text.lower()
    answer_mentions_image_failure = (
        "[tools] image failed" in raw_text_lower
        or "image model failed" in raw_text_lower
        or "all image models failed" in raw_text_lower
    )
    pred = parsed.get("final_answer_index")
    if isinstance(pred, str) and pred.isdigit():
        pred = int(pred)
    valid_prediction = isinstance(pred, int) and 0 <= pred < len(sample["options"])
    row.update(parsed)
    row.update(
        {
            "raw_answer_text": raw_text,
            "answer_mentions_image_failure": answer_mentions_image_failure,
            "final_answer_index": pred,
            "valid_prediction": valid_prediction,
            "correct": pred == sample["gold"] if valid_prediction else False,
            "confidence_float": confidence_as_float(parsed.get("confidence")),
        }
    )
    row.update(inspect_session_tools(agent, session_id, input_mode))
    return row


def compact_attempt(row: dict) -> dict:
    return {
        "attempt_index": row.get("attempt_index"),
        "session_id": row.get("session_id"),
        "ok": row.get("ok"),
        "latency_sec": row.get("latency_sec"),
        "final_answer_index": row.get("final_answer_index"),
        "final_answer": row.get("final_answer"),
        "confidence": row.get("confidence"),
        "confidence_float": row.get("confidence_float"),
        "valid_prediction": row.get("valid_prediction"),
        "correct": row.get("correct"),
        "tool_calls": row.get("tool_calls") or [],
        "forbidden_tool_calls": row.get("forbidden_tool_calls") or [],
        "image_models": row.get("image_models") or [],
        "image_failures": row.get("image_failures") or 0,
        "answer_mentions_image_failure": row.get("answer_mentions_image_failure") or False,
        "raw_answer_text": row.get("raw_answer_text", ""),
        "error": row.get("error"),
    }


def choose_majority(valid_attempts: list[dict]) -> dict:
    counts = Counter(int(row["final_answer_index"]) for row in valid_attempts)
    if not counts:
        return {
            "final_answer_index": None,
            "final_answer": None,
            "confidence": None,
            "confidence_float": None,
            "vote_counts": {},
            "majority_tie": False,
            "selection_reason": "no_valid_votes",
        }

    max_count = max(counts.values())
    tied = sorted(pred for pred, count in counts.items() if count == max_count)
    by_pred: dict[int, list[dict]] = defaultdict(list)
    for row in valid_attempts:
        by_pred[int(row["final_answer_index"])].append(row)

    def avg_conf(pred: int) -> float:
        values = [row.get("confidence_float") for row in by_pred[pred] if row.get("confidence_float") is not None]
        return sum(values) / len(values) if values else -1.0

    selected_pred = max(tied, key=lambda pred: (avg_conf(pred), -tied.index(pred)))
    selected_attempt = max(
        by_pred[selected_pred],
        key=lambda row: row.get("confidence_float") if row.get("confidence_float") is not None else -1.0,
    )
    return {
        "final_answer_index": selected_pred,
        "final_answer": selected_attempt.get("final_answer"),
        "confidence": selected_attempt.get("confidence"),
        "confidence_float": selected_attempt.get("confidence_float"),
        "vote_counts": {str(k): v for k, v in sorted(counts.items())},
        "majority_tie": len(tied) > 1,
        "selection_reason": "majority" if len(tied) == 1 else "confidence_tiebreak",
    }


def run_sample(
    sample: dict,
    *,
    agent: str,
    input_mode: str,
    image_sample: dict | None,
    timeout: int,
    votes: int,
    retry_invalid: int,
) -> dict:
    image_path = None if input_mode == "question_only" else ensure_workspace_image(image_sample or sample)
    started = time.time()
    attempts: list[dict] = []
    max_attempts = votes + retry_invalid
    majority_threshold = votes // 2 + 1

    for attempt_index in range(1, max_attempts + 1):
        attempt = run_attempt(
            sample,
            image_path,
            agent=agent,
            input_mode=input_mode,
            timeout=timeout,
            attempt_index=attempt_index,
        )
        attempts.append(attempt)
        valid_attempts = [row for row in attempts if row.get("ok") and row.get("valid_prediction")]
        vote_counts = Counter(int(row["final_answer_index"]) for row in valid_attempts)
        if vote_counts and max(vote_counts.values()) >= majority_threshold:
            break
        if len(valid_attempts) >= votes:
            break

    valid_attempts = [row for row in attempts if row.get("ok") and row.get("valid_prediction")]
    choice = choose_majority(valid_attempts)
    pred = choice["final_answer_index"]
    tool_calls = [tool for row in attempts for tool in (row.get("tool_calls") or [])]
    forbidden = [tool for row in attempts for tool in (row.get("forbidden_tool_calls") or [])]
    image_models = [model for row in attempts for model in (row.get("image_models") or [])]
    row = {
        "idx": sample["idx"],
        "sample_id": sample["id"],
        "agent": agent,
        "input_mode": input_mode,
        "question": sample["question"],
        "options": sample["options"],
        "gold": sample["gold"],
        "image_source_sample_id": (image_sample or sample).get("id") if image_sample else None,
        "image_source_idx": (image_sample or sample).get("idx") if image_sample else None,
        "image_path": str(image_path) if image_path else None,
        "ok": any(attempt.get("ok") for attempt in attempts),
        "latency_sec": round(time.time() - started, 3),
        "votes_requested": votes,
        "majority_threshold": majority_threshold,
        "retry_invalid": retry_invalid,
        "attempts_used": len(attempts),
        "valid_vote_count": len(valid_attempts),
        "invalid_attempt_count": len([attempt for attempt in attempts if attempt.get("ok") and not attempt.get("valid_prediction")]),
        "errored_attempt_count": len([attempt for attempt in attempts if not attempt.get("ok")]),
        "attempts": [compact_attempt(attempt) for attempt in attempts],
        "tool_calls": tool_calls,
        "image_call_count": tool_calls.count("image"),
        "forbidden_tool_calls": forbidden,
        "image_models": image_models,
        "image_failures": sum(int(attempt.get("image_failures") or 0) for attempt in attempts),
        "answer_mentions_image_failure": any(attempt.get("answer_mentions_image_failure") for attempt in attempts),
        "raw_answer_text": "\n\n--- ATTEMPT ---\n\n".join(
            attempt.get("raw_answer_text", "") for attempt in attempts if attempt.get("raw_answer_text")
        ),
    }
    row.update(choice)
    row.update(
        {
            "valid_prediction": isinstance(pred, int) and 0 <= pred < len(sample["options"]),
            "correct": pred == sample["gold"] if isinstance(pred, int) and 0 <= pred < len(sample["options"]) else False,
        }
    )
    return row


def summarize(rows: list[dict], out_jsonl: Path, agent: str, input_mode: str) -> dict:
    done = [r for r in rows if r.get("ok") and r.get("valid_prediction")]
    correct = [r for r in done if r.get("correct") is True]
    invalid = [r for r in rows if r.get("ok") and not r.get("valid_prediction")]
    forbidden = sum(1 for r in rows if r.get("forbidden_tool_calls"))
    image_failures = sum(int(r.get("image_failures") or 0) for r in rows)
    answer_image_failure_mentions = sum(1 for r in rows if r.get("answer_mentions_image_failure"))
    attempts_used = sum(int(r.get("attempts_used") or 1) for r in rows)
    return {
        "mode": "openclaw_image_web" if "web" in agent else "native_openclaw_only",
        "agent": agent,
        "input_mode": input_mode,
        "requested_samples": len(rows),
        "attempts_used": attempts_used,
        "completed": len(done),
        "invalid_predictions": len(invalid),
        "errors": len([r for r in rows if not r.get("ok")]),
        "accuracy": round(len(correct) / len(done), 4) if done else None,
        "forbidden_tool_violations": forbidden,
        "image_failures": image_failures,
        "answer_image_failure_mentions": answer_image_failure_mentions,
        "out_jsonl": str(out_jsonl),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_FILES), default="pathvqa")
    parser.add_argument("--samples-json", default="")
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--input-mode", choices=["normal", "question_only", "image_shuffle"], default="normal")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=500)
    parser.add_argument("--votes", type=int, default=1)
    parser.add_argument("--retry-invalid", type=int, default=0)
    parser.add_argument("--out-jsonl", default="")
    args = parser.parse_args()
    if args.votes < 1:
        raise ValueError("--votes must be >= 1")
    if args.retry_invalid < 0:
        raise ValueError("--retry-invalid must be >= 0")

    samples = load_samples(args.dataset, args.start_index, args.max_samples, args.samples_json)
    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else (
        OUT_DIR / (
            f"{args.dataset}_{args.agent}_{args.input_mode}_native_openclaw_"
            f"{args.start_index}_{args.max_samples}_{int(time.time())}.jsonl"
        )
    )
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with out_jsonl.open("w", encoding="utf-8") as f:
        for offset, sample in enumerate(samples):
            if args.input_mode == "image_shuffle" and len(samples) > 1:
                image_sample = samples[(offset + 1) % len(samples)]
            elif args.input_mode == "question_only":
                image_sample = None
            else:
                image_sample = sample
            row = run_sample(
                sample,
                agent=args.agent,
                input_mode=args.input_mode,
                image_sample=image_sample,
                timeout=args.timeout,
                votes=args.votes,
                retry_invalid=args.retry_invalid,
            )
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    print(json.dumps(summarize(rows, out_jsonl, args.agent, args.input_mode), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
