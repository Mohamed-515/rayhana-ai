from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


AI_ROOT = Path(__file__).resolve().parents[2]
BASIL_DIR = AI_ROOT / "data" / "raw" / "basil_roboflow_yolov8"
PROCESSED_BASIL_DIR = AI_ROOT / "data" / "processed" / "basil_yolo_detection"
PLANTDOC_DIR = AI_ROOT / "data" / "raw" / "plantdoc"
REPORT_PATH = AI_ROOT / "reports" / "dataset_inspection_report.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def yolo_label_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in root.rglob("*.txt")
        if path.is_file() and any(parent.name.lower() == "labels" for parent in path.parents)
    ]


def find_data_yaml(root: Path) -> Path | None:
    candidates = sorted(root.rglob("data.yaml")) + sorted(root.rglob("data.yml"))
    return candidates[0] if candidates else None


def load_yolo_class_names(data_yaml: Path | None) -> list[str]:
    if data_yaml is None:
        return []
    with data_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item) if str(item).isdigit() else str(item))]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def count_yolo_annotations(root: Path, class_names: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for label_path in yolo_label_files(root):
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if not parts or not parts[0].isdigit():
                continue
            class_id = parts[0]
            label = class_id
            if class_id.isdigit():
                index = int(class_id)
                if 0 <= index < len(class_names):
                    label = class_names[index]
            counts[label] += 1
    return dict(sorted(counts.items()))


def inspect_yolo_dataset(root: Path) -> dict[str, Any]:
    data_yaml = find_data_yaml(root)
    class_names = load_yolo_class_names(data_yaml)
    split_summary: dict[str, dict[str, int | bool]] = {}

    for split in ("train", "valid", "val", "test"):
        split_dir = root / split
        split_summary[split] = {
            "exists": split_dir.exists(),
            "image_count": len(image_files(split_dir)),
            "label_txt_count": len(yolo_label_files(split_dir)),
        }

    return {
        "root": str(root),
        "exists": root.exists(),
        "data_yaml": str(data_yaml) if data_yaml else None,
        "data_yaml_exists": data_yaml is not None,
        "splits": split_summary,
        "total_images": len(image_files(root)),
        "total_label_txt_files": len(yolo_label_files(root)),
        "class_names": class_names,
        "annotation_counts_per_class": count_yolo_annotations(root, class_names),
    }


def inspect_basil_yolo() -> dict[str, Any]:
    return inspect_yolo_dataset(BASIL_DIR)


def inspect_processed_basil_yolo() -> dict[str, Any]:
    return inspect_yolo_dataset(PROCESSED_BASIL_DIR)


def inspect_plantdoc() -> dict[str, Any]:
    images = image_files(PLANTDOC_DIR)
    direct_class_counts: dict[str, int] = {}
    split_class_counts: dict[str, dict[str, int]] = {}
    if PLANTDOC_DIR.exists():
        for child in sorted(PLANTDOC_DIR.iterdir()):
            if child.is_dir():
                count = len(image_files(child))
                if count:
                    direct_class_counts[child.name] = count
                for split_dir in sorted(child.iterdir()):
                    if split_dir.is_dir() and split_dir.name.lower() in {"train", "test", "valid", "val"}:
                        split_class_counts[split_dir.name] = {
                            class_dir.name: len(image_files(class_dir))
                            for class_dir in sorted(split_dir.iterdir())
                            if class_dir.is_dir() and len(image_files(class_dir)) > 0
                        }

    top_level_entries = []
    if PLANTDOC_DIR.exists():
        top_level_entries = [
            {"name": child.name, "type": "dir" if child.is_dir() else "file"}
            for child in sorted(PLANTDOC_DIR.iterdir())
        ]

    return {
        "root": str(PLANTDOC_DIR),
        "exists": PLANTDOC_DIR.exists(),
        "top_level_entries": top_level_entries,
        "total_images": len(images),
        "class_folder_image_counts": direct_class_counts,
        "split_class_folder_image_counts": split_class_counts,
    }


def inspect_plantvillage_hf() -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        return {
            "available": False,
            "error": "Missing dependency: datasets",
            "install_command": "python -m pip install -r requirements.txt",
            "details": str(exc),
        }

    used_config = "color"
    fallback_note = None
    try:
        dataset = load_dataset("mohanty/PlantVillage", "color")
    except Exception as exc:
        if "BuilderConfig 'color' not found" not in str(exc):
            return {
                "available": False,
                "error": "Could not load Hugging Face dataset metadata/sample.",
                "install_command": "python -m pip install -r requirements.txt",
                "details": str(exc),
            }
        try:
            dataset = load_dataset("mohanty/PlantVillage", "default")
            used_config = "default"
            fallback_note = "Requested config 'color' was unavailable locally; used Hugging Face config 'default'."
        except Exception as fallback_exc:
            return {
                "available": False,
                "error": "Could not load Hugging Face dataset metadata/sample.",
                "install_command": "python -m pip install -r requirements.txt",
                "details": str(fallback_exc),
                "original_color_error": str(exc),
            }

    split_sizes = {split: len(dataset[split]) for split in dataset.keys()}
    first_split = next(iter(dataset.keys()), None)
    class_names: list[str] = []
    sample_distribution: dict[str, int] = {}

    if first_split:
        features = dataset[first_split].features
        label_feature = features.get("label")
        if label_feature is not None and hasattr(label_feature, "names"):
            class_names = list(label_feature.names)
        counter: Counter[str] = Counter()
        sample_count = min(1000, len(dataset[first_split]))
        for item in dataset[first_split].select(range(sample_count)):
            label = item.get("label")
            if label is not None:
                label_name = class_names[label] if isinstance(label, int) and label < len(class_names) else str(label)
            elif isinstance(item.get("text"), str):
                parts = item["text"].replace("\\", "/").split("/")
                label_name = parts[-2] if len(parts) >= 2 else "unknown"
            else:
                label_name = "unknown"
            counter[label_name] += 1
        sample_distribution = dict(sorted(counter.items()))
        if not class_names and "text" in features:
            label_values = set()
            for item in dataset[first_split]:
                text_path = item.get("text")
                if isinstance(text_path, str):
                    parts = text_path.replace("\\", "/").split("/")
                    if len(parts) >= 2:
                        label_values.add(parts[-2])
            class_names = sorted(label_values)

    return {
        "available": True,
        "dataset_name": "mohanty/PlantVillage",
        "requested_config": "color",
        "used_config": used_config,
        "fallback_note": fallback_note,
        "features": {split: list(dataset[split].features.keys()) for split in dataset.keys()},
        "splits": list(dataset.keys()),
        "split_sizes": split_sizes,
        "class_names": class_names,
        "sample_distribution_first_1000": sample_distribution,
    }


def main() -> int:
    report = {
        "basil_roboflow_yolov8": inspect_basil_yolo(),
        "processed_basil_yolo_detection": inspect_processed_basil_yolo(),
        "plantdoc": inspect_plantdoc(),
        "plantvillage_hf": inspect_plantvillage_hf(),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved inspection report to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
