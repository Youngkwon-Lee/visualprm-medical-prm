#!/usr/bin/env python3
"""Generate a small VLM batch through the local /generate-steps backend."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any


def load_cases(path: Path, limit: int) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data if isinstance(data, list) else data.get("cases", [])
    return cases[:limit]


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a VLM sample batch via backend")
    parser.add_argument("--input", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8764")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    cases = load_cases(Path(args.input), args.limit)
    image_root = Path(args.image_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, case in enumerate(cases, start=1):
            image_url = str(image_root / case["image_url"])
            payload = {
                "question": case["question"],
                "options": case["options"],
                "gold": case["gold"],
                "dataset": case.get("dataset", "pathvqa"),
                "case_type": case.get("case_type", "Medical"),
                "modality": case.get("modality", ""),
                "image_url": image_url,
                "temperature": 0.0,
                "top_p": 1.0,
            }
            result = post_json(f"{args.backend_url}/generate-steps", payload, args.timeout)
            row = {
                "case_id": case.get("id"),
                "question": case.get("question"),
                "options": case.get("options"),
                "gold_index": case.get("gold"),
                "gold_letter": chr(65 + int(case.get("gold", 0))),
                "image_url": image_url,
                "steps": result.get("steps", []),
                "final_answer_letter": result.get("final_answer_letter"),
                "final_answer_index": result.get("final_answer_index"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{idx}/{len(cases)}] {row['case_id']} -> {row['final_answer_letter']}")


if __name__ == "__main__":
    main()
