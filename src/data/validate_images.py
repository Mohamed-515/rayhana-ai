from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from PIL import Image, UnidentifiedImageError
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit(
        "Missing dependency for image validation.\nInstall with: python -m pip install -r requirements.txt"
    ) from exc


AI_ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOTS = [AI_ROOT / "data" / "raw", AI_ROOT / "data" / "interim"]
REPORT_PATH = AI_ROOT / "reports" / "image_validation_report.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images() -> list[Path]:
    images: list[Path] = []
    for root in TARGET_ROOTS:
        if root.exists():
            images.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    return sorted(images)


def validate_image(path: Path) -> dict[str, Any] | None:
    try:
        with Image.open(path) as image:
            image.verify()
        return None
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return {"path": str(path), "error": str(exc)}


def main() -> int:
    images = iter_images()
    corrupted = []
    for path in tqdm(images, desc="Validating images"):
        issue = validate_image(path)
        if issue:
            corrupted.append(issue)

    report = {
        "target_roots": [str(root) for root in TARGET_ROOTS],
        "total_images_checked": len(images),
        "corrupted_count": len(corrupted),
        "corrupted_images": corrupted,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved image validation report to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
