# Rayhana AI

Rayhana AI is the machine learning workspace for the Rayhana smart basil care app. This folder is intentionally separate from the Flutter app and FastAPI backend so dataset preparation, experiments, model artifacts, and reports can evolve without disturbing production application code.

## Dataset Strategy

- **Basil Roboflow YOLOv8**: basil-specific disease localization. This is the Phase 1 detection dataset because it can teach a YOLO model where disease symptoms appear on basil leaves.
- **PlantDoc**: real-world robustness. This dataset helps test how models behave with less controlled, more natural plant images.
- **PlantVillage Hugging Face**: large controlled plant disease knowledge/classification. This is useful for broad disease classification experiments and later care-knowledge support.

We do not merge all datasets immediately because they are not the same task. Roboflow YOLO data is object detection data with bounding-box labels, while PlantVillage and most PlantDoc usage are classification-style datasets. Mixing them before defining a shared label/task strategy would create noisy training data.

## Phases

### Phase 1: Basil YOLOv8 Detection

Train a YOLOv8 baseline on the basil Roboflow dataset. The goal is basil-specific disease localization for scan results inside Rayhana.

### Phase 2: Classification Comparison

Use PlantVillage and PlantDoc for classification comparison, robustness checks, or a supporting image classification model. These experiments should stay separate from the YOLO detection pipeline unless a clear multi-task design is chosen.

### Phase 3: RAG Care Explanations

Build a vector database for care explanations, disease guidance, and basil treatment recommendations. The RAG layer should explain model outputs in practical, user-friendly care language.

## Colab Training Plan

1. Push only the code workspace to GitHub. Dataset folders, model weights, and reports are ignored by Git.
2. Open `notebooks/02_yolov8_training_colab.ipynb` in Google Colab.
3. Select a GPU runtime.
4. Let the notebook clone this repository into Colab.
5. Download the Basil Roboflow YOLOv8 dataset inside Colab.
6. Run the dataset preparation cell to create `data/processed/basil_yolo_detection`.
7. Train the YOLOv8n baseline.
8. Evaluate and export `best.pt`.

Do not train locally unless the machine is intentionally prepared for GPU training.

## Local Setup

```powershell
cd "D:\Rayhana Project\rayhana-ai"
python -m pip install -r requirements.txt
```

## Local Commands

Extract local dataset ZIPs from `D:\Rayhana Project`:

```powershell
python -m src.data.extract_datasets
```

Inspect all configured datasets:

```powershell
python -m src.data.inspect_all_datasets
```

Validate images without deleting anything:

```powershell
python -m src.data.validate_images
```

Export a small PlantVillage inspection sample:

```powershell
python -m src.data.export_plantvillage_sample
```

## Preparing Basil YOLO Dataset

Create a reproducible 70/20/10 processed split for the basil-only YOLO detection model after the Roboflow export has been downloaded to `data/raw/basil_roboflow_yolov8`:

```powershell
python -m src.data.prepare_basil_yolo_split
```

Generate annotated sample images for visual QA:

```powershell
python -m src.data.visualize_yolo_annotations
```

The processed dataset is written to `data/processed/basil_yolo_detection`, and the split report is written to `reports/basil_yolo_split_report.json`. These paths are local experiment outputs and should not be pushed to GitHub.

## Notes

- Original ZIP files are preserved.
- Extraction does not overwrite existing files unless `--force` is passed.
- GitHub should contain code, notebooks, configs, and documentation only.
- PlantVillage is loaded from Hugging Face on demand and is not exported in full by default.
- Reports are written to `reports/`.
