"""
데이터 로더: OmniMedVQA / PathVQA / VQA-RAD → 통일 raw 포맷
HuggingFace datasets 라이브러리로 다운로드 후 표준 dict 리스트로 변환.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# HF 캐시를 D 드라이브에 저장 (C 드라이브 보호) — import 전에 설정
HF_CACHE_DIR = str(Path(__file__).parent / ".hf_cache")
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset, concatenate_datasets
from huggingface_hub import hf_hub_download
from tqdm import tqdm

DATASET_CONFIGS = {
    "omnimedvqa": {
        "hf_id": "foreverbeliever/OmniMedVQA",
        "splits": ["train"],
        "streaming": False,
    },
    "pathvqa": {
        "hf_id": "flaviagiammarino/path-vqa",
        "splits": ["train"],  # train만 사용 (valid/test 제외)
    },
    "pmcvqa": {
        "hf_id": "RadGenome/PMC-VQA",
        "splits": ["train"],
        "streaming": False,
    },
    "vqarad": {
        "hf_id": "flaviagiammarino/vqa-rad",
        "splits": ["train"],  # train만 사용
    },
}


def _find_omnimedvqa_root() -> Path | None:
    """Find a locally extracted OmniMedVQA root that contains QA_information."""
    candidates = [
        Path(__file__).parent / "OmniMedVQA",
        Path(__file__).parent / "sample" / "OmniMedVQA",
        Path(__file__).parent / "sample",
    ]

    for candidate in candidates:
        if (candidate / "QA_information").exists():
            return candidate
    return None


def load_omnimedvqa(_ds=None) -> list[dict]:
    """OmniMedVQA → 통일 포맷. Local extracted QA JSON files are required."""
    root = _find_omnimedvqa_root()
    if root is None:
        raise FileNotFoundError(
            "OmniMedVQA requires a locally extracted dataset folder containing "
            "'QA_information'. The Hugging Face repo provides a large zip archive, "
            "so load_dataset() alone does not yield usable QA rows in this environment."
        )

    qa_dir = root / "QA_information"
    json_files = sorted(qa_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No OmniMedVQA QA JSON files found under: {qa_dir}")

    items = []
    skipped = 0

    for json_path in tqdm(json_files, desc="OmniMedVQA files"):
        try:
            try:
                rows = json.loads(json_path.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                rows = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except Exception:
            skipped += 1
            continue

        if isinstance(rows, dict):
            rows = [rows]

        for row in rows:
            try:
                question = str(row.get("question", "")).strip()
                answer = str(row.get("gt_answer", "")).strip()
                options = [
                    str(row.get("option_A", "")).strip(),
                    str(row.get("option_B", "")).strip(),
                    str(row.get("option_C", "")).strip(),
                    str(row.get("option_D", "")).strip(),
                ]
                image_rel = str(row.get("image_path", "")).strip()

                if not question or not answer or not any(options):
                    skipped += 1
                    continue

                gold = _answer_to_index(answer, options)
            except Exception:
                skipped += 1
                continue

            items.append({
                "id_orig": str(row.get("question_id", "")),
                "dataset": "omnimedvqa",
                "question": question,
                "options": options,
                "gold": gold,
                "answer_raw": answer,
                "case_type": str(row.get("question_type", "Medical Image")),
                "modality": str(row.get("modality_type", "")),
                "image_path": image_rel,
                "image": None,
            })

    if skipped:
        print(f"  ⚠ OmniMedVQA: {skipped} items skipped")
    return items


def load_pathvqa(ds) -> list[dict]:
    """PathVQA → 통일 포맷. closed-end (yes/no)만 필터."""
    items = []
    skipped = 0
    for row in tqdm(ds, desc="PathVQA"):
        try:
            answer = str(row.get("answer", "")).strip().lower()
            if answer not in ("yes", "no"):
                continue
            gold = 0 if answer == "yes" else 1
        except Exception:
            skipped += 1
            continue
        items.append({
            "id_orig": str(row.get("image", "pathvqa")),
            "dataset": "pathvqa",
            "question": str(row.get("question", "")),
            "options": ["Yes", "No"],
            "gold": gold,
            "answer_raw": answer,
            "case_type": "Pathology",
            "modality": "H&E",
            "image": row.get("image"),
        })
    if skipped:
        print(f"  ⚠ PathVQA: {skipped} items skipped (broken images)")
    return items


def _pmcvqa_choice_text(value: str) -> str:
    text = str(value or "").strip()
    return re.sub(r"^[A-D]\s*:\s*", "", text).strip()


def load_pmcvqa(_ds=None, limit: int | None = None) -> list[dict]:
    """PMC-VQA -> unified format using the public train.csv file."""
    csv_path = Path(
        hf_hub_download(
            repo_id=DATASET_CONFIGS["pmcvqa"]["hf_id"],
            repo_type="dataset",
            filename="train.csv",
            cache_dir=HF_CACHE_DIR,
        )
    )

    items = []
    skipped = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in tqdm(reader, desc="PMC-VQA"):
            question = str(row.get("Question", "")).strip()
            figure_path = str(row.get("Figure_path", "")).strip()
            options = [
                _pmcvqa_choice_text(row.get("Choice A", "")),
                _pmcvqa_choice_text(row.get("Choice B", "")),
                _pmcvqa_choice_text(row.get("Choice C", "")),
                _pmcvqa_choice_text(row.get("Choice D", "")),
            ]
            answer = str(row.get("Answer", "")).strip()
            answer_label = str(row.get("Answer_label", "")).strip().upper()

            if not question or not figure_path or not any(options):
                skipped += 1
                continue

            items.append({
                "id_orig": figure_path,
                "dataset": "pmcvqa",
                "question": question,
                "options": options,
                "gold": _answer_to_index(answer_label or answer, options),
                "answer_raw": answer,
                "case_type": "PMC Figure",
                "modality": "Mixed Figure",
                "image_remote_id": figure_path,
                "image": None,
            })
            if limit and len(items) >= limit:
                break

    if skipped:
        print(f"  ??PMC-VQA: {skipped} items skipped")
    return items


def load_vqarad(ds) -> list[dict]:
    """VQA-RAD → 통일 포맷. CLOSED 타입만 필터."""
    items = []
    skipped = 0
    for row in tqdm(ds, desc="VQA-RAD"):
        try:
            answer = str(row.get("answer", "")).strip().lower()
            if answer not in ("yes", "no"):
                continue
        except Exception:
            skipped += 1
            continue
        gold = 0 if answer == "yes" else 1
        items.append({
            "id_orig": str(row.get("image", "vqarad")),
            "dataset": "vqarad",
            "question": str(row.get("question", "")),
            "options": ["Yes", "No"],
            "gold": gold,
            "answer_raw": answer,
            "case_type": "Radiology",
            "modality": "Mixed",
            "image": row.get("image"),
        })
    if skipped:
        print(f"  ⚠ VQA-RAD: {skipped} items skipped (broken images)")
    return items


def _answer_to_index(answer: str, options: list[str]) -> int:
    """답변을 선택지 인덱스(0-based)로 변환."""
    # "A","B","C","D" 매핑
    letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    ans_upper = answer.strip().upper()
    if ans_upper in letter_map:
        return letter_map[ans_upper]
    # 선택지 텍스트와 직접 매칭
    for i, opt in enumerate(options):
        if opt.strip().lower() == answer.strip().lower():
            return i
    # 매칭 실패 → 0 (기본값)
    return 0


def load_dataset_by_name(name: str, limit: int | None = None) -> list[dict]:
    """데이터셋 이름으로 로드. splits=None이면 전체, 지정 시 해당 split만."""
    cfg = DATASET_CONFIGS.get(name)
    if not cfg:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASET_CONFIGS.keys())}")

    print(f"Loading {name} from HuggingFace ({cfg['hf_id']})...")

    # OmniMedVQA 특별 처리: HF repo is a zip archive, so prefer local extracted QA files.
    is_omnimedvqa = "omnimedvqa" in cfg["hf_id"].lower()
    use_streaming = cfg.get("streaming", False)

    if is_omnimedvqa:
        items = load_omnimedvqa()
        if limit:
            items = items[:limit]
            print(f"  → Limited to {limit} samples")
        print(f"  → {len(items)} items loaded (after filtering)")
        return items

    if name == "pmcvqa":
        items = load_pmcvqa(limit=limit)
        print(f"  ??{len(items)} items loaded (after filtering)")
        return items

    splits = cfg["splits"]
    if use_streaming:
        split_name = splits[0] if splits else "train"
        if is_omnimedvqa:
            print(f"  → Non-streaming mode (image decoding issues in dataset)")
            # Load without streaming to better handle the dataset
            try:
                ds_dict = load_dataset(cfg["hf_id"])
                ds = ds_dict.get(split_name, ds_dict.get("train"))
                print(f"  → Dataset loaded with {len(ds)} samples")
            except Exception as e:
                print(f"  → Error: {e}")
                raise
        else:
            print(f"  → Streaming mode (split: {split_name})")
            ds = load_dataset(cfg["hf_id"], split=split_name, streaming=True)
        if limit:
            if is_omnimedvqa:
                ds = ds.select(range(min(limit, len(ds))))
            else:
                ds = ds.take(limit)
            print(f"  → Limited to {limit} samples")
    elif splits is None:
        ds_dict = load_dataset(cfg["hf_id"])
        available = list(ds_dict.keys())
        print(f"  → Splits available: {available}")
        ds = concatenate_datasets([ds_dict[s] for s in available])
        print(f"  → Total: {len(ds)} samples (all splits)")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
            print(f"  → Limited to {limit} samples")
    elif len(splits) == 1:
        ds = load_dataset(cfg["hf_id"], split=splits[0])
        print(f"  → Using split: {splits[0]} ({len(ds)} samples)")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
            print(f"  → Limited to {limit} samples")
    else:
        ds_dict = load_dataset(cfg["hf_id"])
        ds = concatenate_datasets([ds_dict[s] for s in splits])
        print(f"  → Using splits: {splits} ({len(ds)} samples)")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
            print(f"  → Limited to {limit} samples")

    loader_map = {
        "omnimedvqa": load_omnimedvqa,
        "pathvqa": load_pathvqa,
        "pmcvqa": load_pmcvqa,
        "vqarad": load_vqarad,
    }
    try:
        items = loader_map[name](ds)
    except Exception as e:
        print(f"  ✗ Error loading dataset: {type(e).__name__}: {str(e)[:100]}")
        print(f"  (Note: OmniMedVQA has corrupted images in HuggingFace)")
        items = []
    print(f"  → {len(items)} items loaded (after filtering)")
    return items


def save_raw_json(items: list[dict], output_path: str):
    """PIL Image 제외하고 raw JSON 저장."""
    serializable = []
    for item in items:
        d = {k: v for k, v in item.items() if k != "image"}
        d["has_image"] = item.get("image") is not None
        serializable.append(d)

    # Use utf-8-sig to ensure Windows compatibility
    with open(output_path, "w", encoding="utf-8-sig") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(serializable)} items → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Load medical VQA datasets from HuggingFace")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to load")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: {dataset}_raw.json)")
    args = parser.parse_args()

    output = args.output or f"{args.dataset}_raw.json"
    items = load_dataset_by_name(args.dataset, limit=args.limit)
    save_raw_json(items, output)


if __name__ == "__main__":
    main()
