#!/usr/bin/env python3
"""Prepare small MedicalVisualProcessBench subsets from the shared Drive folder.

This converts the Drive label JSON files into the local format accepted by
run_openclaw_pathvqa_native.py and downloads only the images needed for the
selected samples.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "medical_visual_process_bench"
RAW_DIR = DATA_ROOT / "raw"
OUT_DIR = DATA_ROOT / "openclaw"
IMAGE_DIR = DATA_ROOT / "images"


@dataclass(frozen=True)
class DatasetConfig:
    label_file: str
    image_folder_id: str


DATASETS: dict[str, DatasetConfig] = {
    "r_pathvqa_closed": DatasetConfig(
        label_file="r_pathvqa_closed_label.json",
        image_folder_id="1DHoM7XGXvVAyDhc7OnkKfBPlb8kRTRW-",
    ),
    "r_pathvqa_open": DatasetConfig(
        label_file="r_pathvqa_open_rationale.json",
        image_folder_id="1DHoM7XGXvVAyDhc7OnkKfBPlb8kRTRW-",
    ),
    "r_rad_closed": DatasetConfig(
        label_file="r_rad_closed_rationale.json",
        image_folder_id="1yKTJawFce8Y96cHlZVjzuqcyh2UwHKBI",
    ),
    "r_rad_open": DatasetConfig(
        label_file="r_rad_open_rationale.json",
        image_folder_id="1yKTJawFce8Y96cHlZVjzuqcyh2UwHKBI",
    ),
    "utah_webpath_closed": DatasetConfig(
        label_file="utah_webpath_closed_rationale.json",
        image_folder_id="1Lri9R9aLA3krGBuCuOaXyhkSeFssOuGb",
    ),
}

DATASET_GROUPS: dict[str, list[str]] = {
    "closed": ["r_pathvqa_closed", "r_rad_closed", "utah_webpath_closed"],
    "full": ["r_pathvqa_closed", "r_pathvqa_open", "r_rad_closed", "r_rad_open", "utah_webpath_closed"],
}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def import_gdown_folder_parser() -> Any:
    try:
        from importlib import import_module

        return import_module("gdown.download_folder")
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required for parsing public Google Drive folders. "
            "Install it with: python -m pip install gdown"
        ) from exc


@lru_cache(maxsize=None)
def collect_drive_files(folder_id: str) -> dict[str, str]:
    parser = import_gdown_folder_parser()
    session = requests.Session()
    by_name: dict[str, str] = {}

    def walk(current_id: str) -> None:
        _folder_name, children = parser._parse_embedded_folder_view(session, current_id)
        for child_id, child_name, child_type in children:
            if child_type == parser._GoogleDriveFile.TYPE_FOLDER:
                walk(child_id)
            else:
                by_name[child_name] = child_id

    walk(folder_id)
    log(f"[drive] folder {folder_id}: indexed {len(by_name)} files")
    return by_name


def download_drive_file(file_id: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type.lower():
            raise RuntimeError(f"Drive returned HTML instead of file bytes for {file_id}")
        tmp_path = out_path.with_name(f"{out_path.name}.{file_id}.tmp")
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        tmp_path.replace(out_path)


def download_many(files: list[tuple[str, Path]], max_workers: int) -> None:
    total = len(files)
    if total == 0:
        return
    if max_workers <= 1:
        for index, (file_id, out_path) in enumerate(files, start=1):
            download_drive_file(file_id, out_path)
            if index == 1 or index == total or index % 25 == 0:
                log(f"[download] {index}/{total}")
        return
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_drive_file, file_id, out_path) for file_id, out_path in files]
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed == 1 or completed == total or completed % 25 == 0:
                log(f"[download] {completed}/{total}")


def load_label_rows(path: Path) -> list[tuple[str, dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return sorted(data.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]))
    if isinstance(data, list):
        return [(str(index + 1), row) for index, row in enumerate(data)]
    raise TypeError(f"Unsupported label JSON shape in {path}: {type(data).__name__}")


def convert_row(dataset: str, key: str, row: dict[str, Any], image_path: Path) -> dict[str, Any]:
    choices = [str(choice) for choice in row.get("choices", [])]
    answer_type = str(row.get("answer_type", "CLOSED")).upper()
    answer = row.get("answer")
    base = {
        "idx": int(key) - 1 if key.isdigit() else key,
        "id": f"{dataset}_{key}",
        "question": str(row.get("question", "")),
        "image_path": str(image_path),
        "image_url": str(image_path),
        "answer_type": answer_type,
        "image_organ": row.get("image_organ", ""),
        "source_image": row.get("image"),
        "source_key": key,
    }
    if answer_type == "OPEN":
        if isinstance(answer, int):
            if answer < 0 or answer >= len(choices):
                raise ValueError(f"{dataset}:{key} answer index {answer} is out of range for {choices!r}")
            gold_text = choices[answer]
        else:
            gold_text = str(answer or "").strip()
        if not gold_text:
            raise ValueError(f"{dataset}:{key} has empty open answer")
        return {
            **base,
            "options": [],
            "gold": None,
            "gold_text": gold_text,
            "source_choices": choices,
        }

    if isinstance(answer, bool):
        gold = int(answer)
    elif isinstance(answer, int):
        gold = answer
    elif isinstance(answer, str) and answer.isdigit():
        gold = int(answer)
    else:
        raise ValueError(f"{dataset}:{key} has non-index answer {answer!r}; closed-end data is expected")
    if not choices:
        raise ValueError(f"{dataset}:{key} has no choices")
    if gold < 0 or gold >= len(choices):
        raise ValueError(f"{dataset}:{key} answer index {gold} is out of range for {choices!r}")
    return {
        **base,
        "options": choices,
        "gold": gold,
        "gold_text": choices[gold],
    }


def prepare_dataset(dataset: str, max_samples: int, start_index: int, max_workers: int) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset {dataset!r}. Choose one of: {', '.join(DATASETS)}")
    config = DATASETS[dataset]
    label_path = RAW_DIR / config.label_file
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label JSON: {label_path}")

    rows = load_label_rows(label_path)
    selected = rows[start_index:] if max_samples <= 0 else rows[start_index : start_index + max_samples]
    log(
        f"[prepare] dataset={dataset} start={start_index} "
        f"requested={'all' if max_samples <= 0 else max_samples} selected={len(selected)}"
    )
    image_ids = collect_drive_files(config.image_folder_id)

    converted: list[dict[str, Any]] = []
    missing_images: list[str] = []
    download_job_map: dict[Path, str] = {}
    selected_rows: list[tuple[str, dict[str, Any], Path]] = []
    for key, row in selected:
        image_name = str(row.get("image", "")).strip()
        if not image_name:
            raise ValueError(f"{dataset}:{key} is missing image name")
        file_id = image_ids.get(image_name)
        if not file_id:
            missing_images.append(image_name)
            continue
        image_path = IMAGE_DIR / dataset / image_name
        existing_file_id = download_job_map.get(image_path)
        if existing_file_id and existing_file_id != file_id:
            raise ValueError(f"{dataset}:{key} maps {image_path} to conflicting Drive ids {existing_file_id} vs {file_id}")
        download_job_map[image_path] = file_id
        selected_rows.append((key, row, image_path))

    download_jobs = [(file_id, image_path) for image_path, file_id in download_job_map.items()]
    log(
        f"[prepare] dataset={dataset} sample_refs={len(selected_rows)} "
        f"unique_images={len(download_jobs)} workers={max_workers}"
    )
    download_many(download_jobs, max_workers=max_workers)
    for key, row, image_path in selected_rows:
        converted.append(convert_row(dataset, key, row, image_path.resolve()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{dataset}_{start_index}_{max_samples}_for_openclaw.json"
    out_path.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[prepare] dataset={dataset} wrote={len(converted)} out={out_path}")
    return {
        "dataset": dataset,
        "label_path": str(label_path),
        "out_json": str(out_path),
        "requested": len(selected),
        "converted": len(converted),
        "missing_images": missing_images,
        "image_dir": str((IMAGE_DIR / dataset).resolve()),
        "available_drive_images": len(image_ids),
    }


def prepare_many(datasets: list[str], max_samples: int, start_index: int, max_workers: int) -> dict[str, Any]:
    log(f"[prepare] group start datasets={','.join(datasets)}")
    summaries = [prepare_dataset(dataset, max_samples, start_index, max_workers) for dataset in datasets]
    merged_rows: list[dict[str, Any]] = []
    for item in summaries:
        merged_rows.extend(json.loads(Path(item["out_json"]).read_text(encoding="utf-8")))
    group_name = next(
        (name for name, members in DATASET_GROUPS.items() if members == datasets),
        "group",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_path = OUT_DIR / f"{group_name}_{start_index}_{max_samples}_for_openclaw.json"
    merged_path.write_text(json.dumps(merged_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[prepare] group={group_name} wrote={len(merged_rows)} out={merged_path}")
    return {
        "datasets": datasets,
        "group_name": group_name,
        "start_index": start_index,
        "max_samples": max_samples,
        "total_requested": sum(int(item["requested"]) for item in summaries),
        "total_converted": sum(int(item["converted"]) for item in summaries),
        "missing_images": {item["dataset"]: item["missing_images"] for item in summaries if item["missing_images"]},
        "out_json": str(merged_path),
        "outputs": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS))
    parser.add_argument("--dataset-group", choices=sorted(DATASET_GROUPS))
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if bool(args.dataset) == bool(args.dataset_group):
        raise ValueError("Choose exactly one of --dataset or --dataset-group")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.dataset_group:
        summary = prepare_many(DATASET_GROUPS[args.dataset_group], args.max_samples, args.start_index, args.workers)
        has_missing = bool(summary["missing_images"])
    else:
        summary = prepare_dataset(args.dataset, args.max_samples, args.start_index, args.workers)
        has_missing = bool(summary["missing_images"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if has_missing:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
