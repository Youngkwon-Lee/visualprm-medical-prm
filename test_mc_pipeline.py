#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisualPRM-style medical MC pipeline test.

What this script does:
1. Generate N reasoning candidates per case.
2. For every step prefix in each candidate, sample K continuations.
3. Compute mc_i = correct continuations / total continuations.
4. Save all solutions and all per-step MC scores instead of collapsing to the first one.

This is still an evaluation/test harness, not the main app pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("[ERROR] OPENAI_API_KEY not found in .env")
    sys.exit(1)

client = AsyncOpenAI(api_key=API_KEY)

PAPER_TEMPERATURE = 0.7
PAPER_TOP_P = 0.95


def load_cases(filename: str, limit: int) -> list[dict[str, Any]]:
    with open(ROOT / filename, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    cases = data if isinstance(data, list) else data.get("cases", [])
    return cases[:limit]


def normalize_options(options: Any) -> list[str]:
    if isinstance(options, list):
        return [str(o) for o in options]
    if isinstance(options, dict):
        return [str(v) for _, v in sorted(options.items())]
    return []


def option_letter(index: int | None) -> str:
    if isinstance(index, int) and 0 <= index < 26:
        return chr(65 + index)
    return "?"


def build_generation_prompt(case: dict[str, Any], prefix_steps: list[dict[str, str]] | None = None) -> str:
    options = normalize_options(case["options"])
    options_text = "\n".join(f"{option_letter(i)}. {opt}" for i, opt in enumerate(options))
    prefix_block = ""
    if prefix_steps:
        prefix_text = "\n".join(
            f"{i+1}. {step['title']} :: {step['text']}" for i, step in enumerate(prefix_steps)
        )
        prefix_block = (
            "\nExisting prefix steps:\n"
            f"{prefix_text}\n"
            "Continue after these steps. Do not rewrite them.\n"
        )

    return f"""
You are a medical imaging reasoning model.

Question: {case['question']}
Options:
{options_text}
Correct answer index (hidden for evaluation): {case['gold']}

Generate a medically grounded chain-of-thought with 3 to 5 concise titled steps.
Each step should advance the reasoning.
End with a final answer chosen from the provided options.
{prefix_block}

Return ONLY valid JSON in this format:
{{
  "steps": [
    {{"title": "Step title", "text": "Step explanation"}},
    {{"title": "Step title", "text": "Step explanation"}}
  ],
  "final_answer_index": 0,
  "final_answer_letter": "A"
}}
""".strip()


async def request_json(
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int = 900,
) -> dict[str, Any]:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=60,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content or "{}"
    return json.loads(text)


async def generate_solution(case: dict[str, Any], solution_index: int) -> dict[str, Any]:
    deterministic = solution_index == 0
    result = await request_json(
        build_generation_prompt(case),
        temperature=0.0 if deterministic else PAPER_TEMPERATURE,
        top_p=1.0 if deterministic else PAPER_TOP_P,
    )
    steps = result.get("steps") or []
    if not steps:
        raise ValueError("generation returned no steps")

    normalized_steps = [
        {
            "title": step.get("title") or f"Step {i+1}",
            "text": step.get("text") or "",
        }
        for i, step in enumerate(steps)
    ]
    final_idx = result.get("final_answer_index")
    if not isinstance(final_idx, int):
        letter = str(result.get("final_answer_letter", "")).strip().upper()
        if len(letter) == 1 and "A" <= letter <= "Z":
            final_idx = ord(letter) - 65
        else:
            final_idx = None

    return {
        "solution_index": solution_index + 1,
        "sampling": {
            "temperature": 0.0 if deterministic else PAPER_TEMPERATURE,
            "top_p": 1.0 if deterministic else PAPER_TOP_P,
            "deterministic": deterministic,
        },
        "steps": normalized_steps,
        "final_answer_index": final_idx,
        "final_answer_letter": option_letter(final_idx),
    }


async def sample_continuation(case: dict[str, Any], prefix_steps: list[dict[str, str]]) -> dict[str, Any]:
    return await request_json(
        build_generation_prompt(case, prefix_steps=prefix_steps),
        temperature=PAPER_TEMPERATURE,
        top_p=PAPER_TOP_P,
    )


def continuation_is_correct(case: dict[str, Any], continuation: dict[str, Any]) -> bool:
    gold = int(case["gold"])
    final_idx = continuation.get("final_answer_index")
    if isinstance(final_idx, int):
        return final_idx == gold
    letter = str(continuation.get("final_answer_letter", "")).strip().upper()
    return letter == option_letter(gold)


async def evaluate_prefix(case: dict[str, Any], prefix_steps: list[dict[str, str]], k: int) -> dict[str, Any]:
    successes = 0
    for _ in range(k):
        continuation = await sample_continuation(case, prefix_steps)
        if continuation_is_correct(case, continuation):
            successes += 1
    score = successes / k
    return {
        "success": successes,
        "total": k,
        "mc_score": score,
        "label": "+" if score > 0 else "-",
    }


async def evaluate_solution(case: dict[str, Any], solution: dict[str, Any], k: int) -> dict[str, Any]:
    step_results = []
    for idx in range(len(solution["steps"])):
        prefix = solution["steps"][: idx + 1]
        rollout = await evaluate_prefix(case, prefix, k)
        step_results.append(
            {
                "step": idx + 1,
                "title": solution["steps"][idx]["title"],
                "text": solution["steps"][idx]["text"],
                **rollout,
            }
        )

    final_step = step_results[-1]
    final_correct = solution.get("final_answer_index") == int(case["gold"])
    return {
        **solution,
        "step_results": step_results,
        "solution_mc_score": final_step["mc_score"],
        "solution_label": final_step["label"],
        "final_answer_correct": final_correct,
    }


async def process_case(case: dict[str, Any], num_solutions: int, k: int) -> dict[str, Any]:
    print(f"\n[CASE] {case['id']}")
    print(f"  Q: {case['question'][:80]}")

    evaluated_solutions = []
    for sol_idx in range(num_solutions):
        print(f"  Solution {sol_idx + 1}/{num_solutions}")
        solution = await generate_solution(case, sol_idx)
        print(f"    Generated {len(solution['steps'])} steps, final={solution['final_answer_letter']}")
        evaluated = await evaluate_solution(case, solution, k)
        print(
            f"    [OK] final-step mc={evaluated['solution_mc_score']:.4f} "
            f"({evaluated['step_results'][-1]['success']}/{k}), label={evaluated['solution_label']}"
        )
        evaluated_solutions.append(evaluated)

    correct_solutions = sum(1 for s in evaluated_solutions if s["solution_label"] == "+")
    mean_solution_score = sum(s["solution_mc_score"] for s in evaluated_solutions) / len(evaluated_solutions)
    best_solution_score = max(s["solution_mc_score"] for s in evaluated_solutions)

    return {
        "id": case["id"],
        "case_type": case.get("case_type", "Unknown"),
        "modality": case.get("modality", ""),
        "question": case["question"],
        "options": normalize_options(case["options"]),
        "gold": case["gold"],
        "image_url": case.get("image_url", ""),
        "solutions": evaluated_solutions,
        "case_summary": {
            "solutions_generated": num_solutions,
            "correct_solutions": correct_solutions,
            "mean_solution_mc_score": mean_solution_score,
            "best_solution_mc_score": best_solution_score,
            "primary_solution_mc_score": evaluated_solutions[0]["solution_mc_score"],
            "primary_solution_label": evaluated_solutions[0]["solution_label"],
        },
    }


async def process_dataset(
    dataset_name: str,
    input_filename: str,
    output_filename: str,
    *,
    num_cases: int,
    num_solutions: int,
    k: int,
) -> dict[str, Any]:
    print(f"\n{'=' * 72}")
    print(f"[PROCESSING] {dataset_name}")
    print(f"  cases={num_cases}, solutions={num_solutions}, rollout_k={k}")
    print(f"{'=' * 72}")

    cases = load_cases(input_filename, num_cases)
    started = time.time()
    processed_cases: list[dict[str, Any]] = []

    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}]", end=" ", flush=True)
        try:
            processed_cases.append(await process_case(case, num_solutions, k))
        except Exception as exc:
            print(f"\n  [ERROR] {case.get('id')}: {exc}")

    payload = {
        "dataset": dataset_name,
        "input_file": input_filename,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "num_cases": num_cases,
            "num_solutions": num_solutions,
            "rollout_k": k,
            "model": "gpt-4o-mini",
            "paper_sampling_temperature": PAPER_TEMPERATURE,
            "paper_sampling_top_p": PAPER_TOP_P,
        },
        "cases": processed_cases,
        "summary": build_dataset_summary(processed_cases),
        "runtime_seconds": round(time.time() - started, 2),
    }

    with open(ROOT / output_filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Saved {len(processed_cases)} cases to {output_filename}")
    return payload


def build_dataset_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {
            "total_cases": 0,
            "avg_primary_solution_mc_score": 0.0,
            "avg_best_solution_mc_score": 0.0,
            "avg_correct_solutions": 0.0,
            "primary_label_positive_pct": 0.0,
        }

    primary_scores = [c["case_summary"]["primary_solution_mc_score"] for c in cases]
    best_scores = [c["case_summary"]["best_solution_mc_score"] for c in cases]
    correct_solution_counts = [c["case_summary"]["correct_solutions"] for c in cases]
    primary_positive = [c["case_summary"]["primary_solution_label"] == "+" for c in cases]

    return {
        "total_cases": len(cases),
        "avg_primary_solution_mc_score": sum(primary_scores) / len(primary_scores),
        "avg_best_solution_mc_score": sum(best_scores) / len(best_scores),
        "avg_correct_solutions": sum(correct_solution_counts) / len(correct_solution_counts),
        "primary_label_positive_pct": sum(primary_positive) / len(primary_positive) * 100,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a VisualPRM-style MC test on medical VQA datasets.")
    parser.add_argument("--num-cases", type=int, default=20)
    parser.add_argument("--num-solutions", type=int, default=4)
    parser.add_argument("--rollout-k", type=int, default=16)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--datasets", nargs="*", default=None, help="Subset of datasets to run: pathvqa omnimedvqa pmcvqa vqarad")
    args = parser.parse_args()

    run_map = {
        "pathvqa": ("PathVQA", "pathvqa_for_app.json", "pathvqa_test_results.json"),
        "omnimedvqa": ("OmniMedVQA", "omnimedvqa_for_app_test.json", "omnimedvqa_test_results.json"),
        "pmcvqa": ("PMC-VQA", "pmcvqa_for_app_test.json", "pmcvqa_test_results.json"),
        "vqarad": ("VQA-RAD", "vqarad_for_app.json", "vqarad_test_results.json"),
    }
    if args.datasets:
        runs = [run_map[name.lower()] for name in args.datasets if name.lower() in run_map]
    else:
        runs = list(run_map.values())

    final = {}
    for dataset_name, input_filename, output_filename in runs:
        output_path = ROOT / output_filename
        stem = output_path.stem
        suffix = args.output_suffix or f"_{args.num_cases}c_{args.num_solutions}s_{args.rollout_k}k"
        versioned_output = f"{stem}{suffix}{output_path.suffix}"
        payload = await process_dataset(
            dataset_name,
            input_filename,
            versioned_output,
            num_cases=args.num_cases,
            num_solutions=args.num_solutions,
            k=args.rollout_k,
        )
        final[dataset_name] = payload["summary"]

    print(f"\n{'=' * 72}")
    print("[FINAL SUMMARY]")
    print(f"{'=' * 72}")
    for name, summary in final.items():
        print(f"\n{name}")
        print(f"  cases: {summary['total_cases']}")
        print(f"  avg primary mc: {summary['avg_primary_solution_mc_score']:.4f}")
        print(f"  avg best mc:    {summary['avg_best_solution_mc_score']:.4f}")
        print(f"  avg + sols:     {summary['avg_correct_solutions']:.2f}/{args.num_solutions}")
        print(f"  primary + pct:  {summary['primary_label_positive_pct']:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
