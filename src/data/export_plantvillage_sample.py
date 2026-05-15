from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = AI_ROOT / "data" / "interim" / "plantvillage_sample"
INVALID_WINDOWS_CHARS = re.compile(r'[<>:"/\\|?*]+')


def clean_windows_name(value: str) -> str:
    cleaned = INVALID_WINDOWS_CHARS.sub("_", value).strip(" .")
    return cleaned or "unknown_class"


def load_plantvillage():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets\nInstall with: python -m pip install -r requirements.txt"
        ) from exc

    try:
        return load_dataset("mohanty/PlantVillage", "color"), "color"
    except Exception as exc:
        if "BuilderConfig 'color' not found" not in str(exc):
            raise
        print("Requested PlantVillage config 'color' is unavailable; falling back to config 'default'.")
        return load_dataset("mohanty/PlantVillage", "default"), "default"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a small PlantVillage sample for inspection.")
    parser.add_argument("--max-per-class", type=int, default=100)
    args = parser.parse_args()

    dataset, used_config = load_plantvillage()
    split_name = "train" if "train" in dataset else next(iter(dataset.keys()))
    split = dataset[split_name]
    if "image" not in split.features or "label" not in split.features:
        raise SystemExit(
            "PlantVillage dataset loaded, but this Hugging Face builder does not expose image/label fields. "
            "It exposes: "
            f"{list(split.features.keys())}. Cannot export images from this builder automatically."
        )
    label_names = list(split.features["label"].names)
    counts: Counter[str] = Counter()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for index, item in enumerate(split):
        label_name = label_names[item["label"]]
        if counts[label_name] >= args.max_per_class:
            continue
        class_dir = OUTPUT_ROOT / clean_windows_name(label_name)
        class_dir.mkdir(parents=True, exist_ok=True)
        image = item["image"].convert("RGB")
        output_path = class_dir / f"{counts[label_name]:04d}_{index}.jpg"
        image.save(output_path, format="JPEG", quality=95)
        counts[label_name] += 1
        if len(counts) == len(label_names) and all(counts[name] >= args.max_per_class for name in label_names):
            break

    print(f"Exported PlantVillage sample to: {OUTPUT_ROOT}")
    print(f"Used PlantVillage config: {used_config}")
    for label_name, count in sorted(counts.items()):
        print(f"{label_name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
