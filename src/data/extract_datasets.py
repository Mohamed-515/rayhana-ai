from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(r"D:\Rayhana Project")
AI_ROOT = PROJECT_ROOT / "rayhana-ai"
RAW_ROOT = AI_ROOT / "data" / "raw"
SKIP_CODE_ARCHIVE_KEYWORDS = (
    "rayhana-app",
    "rayhana_app",
    "rayhana-backend",
    "backend-main",
    "app-main",
    "stitch_rayhana",
)


def clean_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return "_".join(part for part in safe.split("_") if part).lower()


def detect_destination(zip_path: Path) -> Path | None:
    lower_name = zip_path.name.lower()
    if any(keyword in lower_name for keyword in SKIP_CODE_ARCHIVE_KEYWORDS):
        return None
    if any(keyword in lower_name for keyword in ("basil", "roboflow", "yolo")):
        return RAW_ROOT / "basil_roboflow_yolov8"
    if "plantdoc" in lower_name or "plantdec" in lower_name:
        return RAW_ROOT / "plantdoc"
    return RAW_ROOT / f"unknown_{clean_name(zip_path.stem)}"


def destination_has_files(destination: Path) -> bool:
    return destination.exists() and any(destination.iterdir())


def extract_zip(zip_path: Path, destination: Path, force: bool) -> str:
    destination.mkdir(parents=True, exist_ok=True)
    if destination_has_files(destination):
        if not force:
            return f"SKIPPED existing extraction: {zip_path.name} -> {destination}"
        shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)
    return f"EXTRACTED: {zip_path.name} -> {destination}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract local Rayhana AI dataset ZIP files.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing extracted files.")
    args = parser.parse_args()

    zip_paths = sorted(PROJECT_ROOT.glob("*.zip"))
    if not zip_paths:
        print(f"No ZIP files found in {PROJECT_ROOT}")
        return 0

    print(f"Searching for ZIP files in: {PROJECT_ROOT}")
    for zip_path in zip_paths:
        destination = detect_destination(zip_path)
        if destination is None:
            print(f"SKIPPED non-AI project archive: {zip_path.name}")
            continue
        try:
            print(extract_zip(zip_path, destination, args.force))
        except zipfile.BadZipFile:
            print(f"ERROR invalid ZIP file: {zip_path}")
        except OSError as exc:
            print(f"ERROR extracting {zip_path.name}: {exc}")

    print("Original ZIP files were preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
