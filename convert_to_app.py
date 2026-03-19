"""
Convert raw medical VQA samples into the app-friendly JSON format.
"""

import argparse
import json
import re
from pathlib import Path
from shutil import copy2

import requests
from PIL import Image
from tqdm import tqdm

from data_loader import DATASET_CONFIGS, _find_omnimedvqa_root, load_dataset_by_name

BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images"


def to_app_format(item: dict, idx: int) -> dict:
    dataset = item["dataset"]
    item_id = f"{dataset}_{idx:06d}"
    return {
        "id": item_id,
        "case_type": item.get("case_type", "Unknown"),
        "modality": item.get("modality", ""),
        "question": item["question"],
        "options": item["options"],
        "gold": item["gold"],
        "image_url": f"images/{dataset}/{item_id}.jpg",
        "sol_num": 1,
        "final_label": "+",
        "steps": [
            {
                "step": 1,
                "title": "Step 1",
                "text": "placeholder - replace with generated reasoning",
                "label": "+",
                "mc_score": 0.0,
            }
        ],
    }


def save_image(img: Image.Image, path: Path):
    if img is None:
        return
    if img.mode != "RGB":
        img = img.convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), "JPEG", quality=85)


def copy_local_image(image_rel_path: str, destination: Path):
    omni_root = _find_omnimedvqa_root()
    if omni_root is None:
        raise FileNotFoundError("OmniMedVQA root with QA_information was not found.")

    source = omni_root / image_rel_path
    if not source.exists():
        raise FileNotFoundError(f"Source image not found: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    copy2(source, destination)


def _pmc_figure_url(figure_name: str) -> str:
    match = re.match(r"^(PMC\d+)_(.+)\.(jpg|jpeg|png)$", figure_name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported PMC figure name: {figure_name}")

    article_id, figure_id, _ext = match.groups()
    article_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{article_id}/"
    html = requests.get(article_url, timeout=30).text
    figure_match = re.search(
        rf'id="{re.escape(figure_id)}".*?<img[^>]+src="([^"]+)"',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not figure_match:
        raise FileNotFoundError(f"Could not map {figure_name} on {article_url}")

    image_url = figure_match.group(1)
    if image_url.startswith("/"):
        image_url = f"https://pmc.ncbi.nlm.nih.gov{image_url}"
    return image_url


def download_pmcvqa_figure(figure_name: str, destination: Path):
    image_url = _pmc_figure_url(figure_name)
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)


def convert_dataset(dataset_name: str, limit: int | None = None, output: str | None = None):
    img_dir = IMAGES_DIR / dataset_name
    img_dir.mkdir(parents=True, exist_ok=True)

    items = load_dataset_by_name(dataset_name, limit=limit)
    app_data = []
    saved_images = 0

    for idx, item in enumerate(tqdm(items, desc=f"Converting {dataset_name}")):
        app_item = to_app_format(item, idx)
        img_path = img_dir / f"{app_item['id']}.jpg"

        if item.get("image") is not None:
            try:
                save_image(item["image"], img_path)
                saved_images += 1
            except Exception as exc:
                print(f"  Warning: Failed to save image for {app_item['id']}: {exc}")
                app_item["image_url"] = ""
        elif item.get("image_path"):
            try:
                copy_local_image(item["image_path"], img_path)
                saved_images += 1
            except Exception as exc:
                print(f"  Warning: Failed to copy image for {app_item['id']}: {exc}")
                app_item["image_url"] = ""
        elif item.get("image_remote_id"):
            try:
                download_pmcvqa_figure(item["image_remote_id"], img_path)
                saved_images += 1
            except Exception as exc:
                print(f"  Warning: Failed to download image for {app_item['id']}: {exc}")
                app_item["image_url"] = ""

        app_data.append(app_item)

    output_path = output or f"{dataset_name}_for_app.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(app_data, handle, ensure_ascii=False, indent=2)

    print("\nDone!")
    print(f"  JSON: {output_path} ({len(app_data)} items)")
    print(f"  Images: {img_dir}/ ({saved_images} saved)")


def main():
    parser = argparse.ArgumentParser(description="Convert medical VQA data to app.html format")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    convert_dataset(args.dataset, limit=args.limit, output=args.output)


if __name__ == "__main__":
    main()
