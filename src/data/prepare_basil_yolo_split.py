from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AI_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = AI_ROOT / "data" / "raw" / "basil_roboflow_yolov8"
RAW_IMAGES_DIR = RAW_ROOT / "train" / "images"
RAW_LABELS_DIR = RAW_ROOT / "train" / "labels"
PROCESSED_ROOT = AI_ROOT / "data" / "processed" / "basil_yolo_detection"
REPORT_PATH = AI_ROOT / "reports" / "basil_yolo_split_report.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES = ["fusarium", "healthy", "powdery"]
VALID_CLASS_IDS = set(range(len(CLASS_NAMES)))
SPLIT_RATIOS = {"train": 0.7, "valid": 0.2, "test": 0.1}
RANDOM_SEED = 42


@dataclass(frozen=True)
class YoloPair:
    stem: str
    image_path: Path
    label_path: Path


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def list_images(root: Path) -> dict[str, Path]:
    images = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    return {path.stem: path for path in images}


def list_labels(root: Path) -> dict[str, Path]:
    labels = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".txt")
    return {path.stem: path for path in labels}


def parse_label_file(label_path: Path) -> tuple[Counter[str], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    invalid_rows: list[dict[str, Any]] = []

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            invalid_rows.append({"file": str(label_path), "line": line_number, "reason": "Expected at least 5 YOLO fields."})
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            invalid_rows.append({"file": str(label_path), "line": line_number, "reason": "Class ID is not an integer."})
            continue

        if class_id not in VALID_CLASS_IDS:
            invalid_rows.append({"file": str(label_path), "line": line_number, "reason": f"Class ID {class_id} is outside 0-2."})
            continue

        try:
            [float(value) for value in parts[1:5]]
        except ValueError:
            invalid_rows.append({"file": str(label_path), "line": line_number, "reason": "Bounding box values are not numeric."})
            continue

        counts[CLASS_NAMES[class_id]] += 1

    return counts, invalid_rows


def count_annotations(pairs: list[YoloPair]) -> dict[str, int]:
    counts: Counter[str] = Counter({class_name: 0 for class_name in CLASS_NAMES})
    for pair in pairs:
        label_counts, _ = parse_label_file(pair.label_path)
        counts.update(label_counts)
    return dict(counts)


def split_pairs(pairs: list[YoloPair]) -> dict[str, list[YoloPair]]:
    shuffled = sorted(pairs, key=lambda pair: pair.stem)
    random.Random(RANDOM_SEED).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * SPLIT_RATIOS["train"])
    valid_count = int(total * SPLIT_RATIOS["valid"])

    return {
        "train": shuffled[:train_count],
        "valid": shuffled[train_count : train_count + valid_count],
        "test": shuffled[train_count + valid_count :],
    }


def reset_processed_root() -> None:
    if PROCESSED_ROOT.exists():
        processed_parent = AI_ROOT / "data" / "processed"
        if not is_relative_to(PROCESSED_ROOT, processed_parent):
            raise RuntimeError(f"Refusing to clear unexpected path: {PROCESSED_ROOT}")
        shutil.rmtree(PROCESSED_ROOT)

    for split in ("train", "valid", "test"):
        (PROCESSED_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (PROCESSED_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_split_files(splits: dict[str, list[YoloPair]]) -> None:
    for split_name, pairs in splits.items():
        image_dir = PROCESSED_ROOT / split_name / "images"
        label_dir = PROCESSED_ROOT / split_name / "labels"
        for pair in pairs:
            shutil.copy2(pair.image_path, image_dir / pair.image_path.name)
            shutil.copy2(pair.label_path, label_dir / pair.label_path.name)


def write_data_yaml() -> None:
    data_yaml = "\n".join(
        [
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "nc: 3",
            "names:",
            "  - fusarium",
            "  - healthy",
            "  - powdery",
            "",
        ]
    )
    (PROCESSED_ROOT / "data.yaml").write_text(data_yaml, encoding="utf-8")


def build_report(
    *,
    total_images: int,
    splits: dict[str, list[YoloPair]],
    missing_labels: list[str],
    missing_images: list[str],
    invalid_rows: list[dict[str, Any]],
    prepared: bool,
) -> dict[str, Any]:
    return {
        "prepared": prepared,
        "source_root": str(RAW_ROOT),
        "output_root": str(PROCESSED_ROOT),
        "random_seed": RANDOM_SEED,
        "split_ratios": SPLIT_RATIOS,
        "classes": CLASS_NAMES,
        "total_images": total_images,
        "split_counts": {split: len(pairs) for split, pairs in splits.items()},
        "annotation_counts_per_class_per_split": {
            split: count_annotations(pairs) for split, pairs in splits.items()
        },
        "missing_label_count": len(missing_labels),
        "missing_image_count": len(missing_images),
        "invalid_label_count": len(invalid_rows),
        "missing_labels": missing_labels,
        "missing_images": missing_images,
        "invalid_labels": invalid_rows,
    }


def main() -> int:
    if not RAW_IMAGES_DIR.exists() or not RAW_LABELS_DIR.exists():
        raise SystemExit(f"Missing raw YOLO folders under: {RAW_ROOT}")

    images = list_images(RAW_IMAGES_DIR)
    labels = list_labels(RAW_LABELS_DIR)
    missing_labels = sorted(stem for stem in images if stem not in labels)
    missing_images = sorted(stem for stem in labels if stem not in images)

    invalid_rows: list[dict[str, Any]] = []
    for label_path in labels.values():
        _, label_invalid_rows = parse_label_file(label_path)
        invalid_rows.extend(label_invalid_rows)

    matched_stems = sorted(set(images) & set(labels))
    pairs = [YoloPair(stem=stem, image_path=images[stem], label_path=labels[stem]) for stem in matched_stems]
    splits = split_pairs(pairs) if not missing_labels and not missing_images and not invalid_rows else {
        "train": [],
        "valid": [],
        "test": [],
    }

    prepared = not missing_labels and not missing_images and not invalid_rows
    if prepared:
        reset_processed_root()
        copy_split_files(splits)
        write_data_yaml()

    report = build_report(
        total_images=len(images),
        splits=splits,
        missing_labels=missing_labels,
        missing_images=missing_images,
        invalid_rows=invalid_rows,
        prepared=prepared,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved split report to: {REPORT_PATH}")

    if not prepared:
        raise SystemExit("Raw Basil YOLO validation failed. Processed dataset was not created.")

    print(f"Processed Basil YOLO dataset ready at: {PROCESSED_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
