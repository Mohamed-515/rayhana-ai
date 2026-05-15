from __future__ import annotations

import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


AI_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = AI_ROOT / "data" / "processed" / "basil_yolo_detection"
OUTPUT_ROOT = AI_ROOT / "reports" / "figures" / "basil_yolo_samples"
CLASS_NAMES = ["fusarium", "healthy", "powdery"]
COLORS = {
    "fusarium": (220, 53, 69),
    "healthy": (25, 135, 84),
    "powdery": (13, 110, 253),
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RANDOM_SEED = 42
SAMPLE_COUNT = 12


def list_processed_images() -> list[tuple[str, Path]]:
    images: list[tuple[str, Path]] = []
    for split in ("train", "valid", "test"):
        image_dir = DATASET_ROOT / split / "images"
        if image_dir.exists():
            images.extend(
                (split, path)
                for path in sorted(image_dir.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
    return images


def label_path_for(split: str, image_path: Path) -> Path:
    return DATASET_ROOT / split / "labels" / f"{image_path.stem}.txt"


def load_annotations(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    annotations: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return annotations

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = [float(value) for value in parts[1:5]]
        except ValueError:
            continue
        if 0 <= class_id < len(CLASS_NAMES):
            annotations.append((class_id, x_center, y_center, width, height))
    return annotations


def draw_annotations(split: str, image_path: Path, output_path: Path) -> None:
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    image_width, image_height = canvas.size

    for class_id, x_center, y_center, box_width, box_height in load_annotations(label_path_for(split, image_path)):
        class_name = CLASS_NAMES[class_id]
        color = COLORS[class_name]
        left = max(0, int((x_center - box_width / 2) * image_width))
        top = max(0, int((y_center - box_height / 2) * image_height))
        right = min(image_width - 1, int((x_center + box_width / 2) * image_width))
        bottom = min(image_height - 1, int((y_center + box_height / 2) * image_height))

        line_width = max(2, image_width // 240)
        draw.rectangle((left, top, right, bottom), outline=color, width=line_width)

        label = f"{class_name}"
        text_bbox = draw.textbbox((left, top), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_top = max(0, top - text_height - 6)
        draw.rectangle((left, label_top, left + text_width + 8, label_top + text_height + 6), fill=color)
        draw.text((left + 4, label_top + 3), label, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def reset_output_root() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    if not DATASET_ROOT.exists():
        raise SystemExit(
            f"Processed dataset not found at {DATASET_ROOT}. Run: python -m src.data.prepare_basil_yolo_split"
        )

    images = list_processed_images()
    if not images:
        raise SystemExit(f"No processed images found in {DATASET_ROOT}")

    random_source = random.Random(RANDOM_SEED)
    samples = images if len(images) <= SAMPLE_COUNT else random_source.sample(images, SAMPLE_COUNT)

    reset_output_root()
    for index, (split, image_path) in enumerate(samples, start=1):
        output_path = OUTPUT_ROOT / f"{index:02d}_{split}_{image_path.stem}.jpg"
        draw_annotations(split, image_path, output_path)
        print(f"Saved sample: {output_path}")

    print(f"Saved {len(samples)} annotated samples to: {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
