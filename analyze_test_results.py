#!/usr/bin/env python3
"""Analyze VisualPRM-style MC pipeline test results."""

from __future__ import annotations

import io
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent


def load_results(filename: str) -> dict[str, Any] | None:
    path = ROOT / filename
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def compute_metrics(payload: dict[str, Any] | list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not payload:
        return None

    if isinstance(payload, list):
        return compute_legacy_metrics(payload)

    cases = payload.get("cases", [])
    if not cases:
        return None

    primary_scores = [case["case_summary"]["primary_solution_mc_score"] for case in cases]
    best_scores = [case["case_summary"]["best_solution_mc_score"] for case in cases]
    correct_solution_counts = [case["case_summary"]["correct_solutions"] for case in cases]
    primary_labels = [case["case_summary"]["primary_solution_label"] for case in cases]

    per_solution_scores = []
    per_step_scores = []
    for case in cases:
        for solution in case.get("solutions", []):
            per_solution_scores.append(solution.get("solution_mc_score", 0.0))
            for step in solution.get("step_results", []):
                per_step_scores.append(step.get("mc_score", 0.0))

    return {
        "total_cases": len(cases),
        "config": payload.get("config", {}),
        "avg_primary_mc": sum(primary_scores) / len(primary_scores),
        "avg_best_mc": sum(best_scores) / len(best_scores),
        "primary_positive_pct": sum(1 for x in primary_labels if x == "+") / len(primary_labels) * 100,
        "avg_correct_solutions": sum(correct_solution_counts) / len(correct_solution_counts),
        "primary_distribution": Counter(primary_scores),
        "solution_distribution": Counter(per_solution_scores),
        "avg_step_mc": (sum(per_step_scores) / len(per_step_scores)) if per_step_scores else 0.0,
    }


def compute_legacy_metrics(cases: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not cases:
        return None

    primary_scores = [case["mc_results"]["mc_score"] for case in cases]
    primary_labels = [case["final_label"] for case in cases]
    correct_solution_counts = [case["mc_results"]["correct_solutions"] for case in cases]

    return {
        "total_cases": len(cases),
        "config": {"legacy_format": True},
        "avg_primary_mc": sum(primary_scores) / len(primary_scores),
        "avg_best_mc": sum(primary_scores) / len(primary_scores),
        "primary_positive_pct": sum(1 for x in primary_labels if x == "+") / len(primary_labels) * 100,
        "avg_correct_solutions": sum(correct_solution_counts) / len(correct_solution_counts),
        "primary_distribution": Counter(primary_scores),
        "solution_distribution": Counter(primary_scores),
        "avg_step_mc": sum(primary_scores) / len(primary_scores),
    }


def print_dataset_report(name: str, metrics: dict[str, Any]) -> None:
    config = metrics["config"]
    num_solutions = config.get("num_solutions", 4)
    print(f"\n{name} ({metrics['total_cases']} cases)")
    print("-" * 72)
    if config.get("legacy_format"):
        print("  Config: legacy result format")
    else:
        print(f"  Config: solutions={config.get('num_solutions')} rollout_k={config.get('rollout_k')} model={config.get('model')}")
    print(f"  Avg primary MC:        {metrics['avg_primary_mc']:.4f}")
    print(f"  Avg best MC:           {metrics['avg_best_mc']:.4f}")
    print(f"  Primary '+' labels:    {metrics['primary_positive_pct']:.1f}%")
    print(f"  Avg correct solutions: {metrics['avg_correct_solutions']:.2f}/{num_solutions}")
    print(f"  Avg step MC:           {metrics['avg_step_mc']:.4f}")
    print(f"  Primary distribution:  {dict(sorted(metrics['primary_distribution'].items()))}")
    print(f"  Solution distribution: {dict(sorted(metrics['solution_distribution'].items()))}")


def print_report() -> None:
    print("\n" + "=" * 72)
    print("VISUALPRM-STYLE MC PIPELINE ANALYSIS")
    print("=" * 72)

    pmc_metrics = compute_metrics(load_results("pmcvqa_test_results.json"))
    rad_metrics = compute_metrics(load_results("vqarad_test_results.json"))

    if pmc_metrics:
        print_dataset_report("PMC-VQA", pmc_metrics)
    else:
        print("\nPMC-VQA: no valid results found")

    if rad_metrics:
        print_dataset_report("VQA-RAD", rad_metrics)
    else:
        print("\nVQA-RAD: no valid results found")

    if pmc_metrics and rad_metrics:
        print("\nComparison")
        print("-" * 72)
        print(f"  Avg primary MC diff (RAD-PMC): {rad_metrics['avg_primary_mc'] - pmc_metrics['avg_primary_mc']:+.4f}")
        print(f"  Avg best MC diff (RAD-PMC):    {rad_metrics['avg_best_mc'] - pmc_metrics['avg_best_mc']:+.4f}")
        print(f"  '+' label diff (RAD-PMC):      {rad_metrics['primary_positive_pct'] - pmc_metrics['primary_positive_pct']:+.1f}%")


if __name__ == "__main__":
    os.chdir(ROOT)
    print_report()
