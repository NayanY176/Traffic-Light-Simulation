from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from ultralytics import YOLO
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

from src.config import project_root
from src.device import select_device
from src.evaluate import evaluate_yolo, relative_improvement, write_metrics_table
from src.logging_utils import get_logger

logger = get_logger(__name__)


def train_yolov5_baseline(
    data_yaml: Path,
    epochs: int = 30,
    imgsz: int = 960,
    batch: int = 8,
    device: Optional[str] = None,
    name: str = "baseline_yolov5s",
) -> Path:
    model = YOLO("yolov5su.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=select_device(device),
        project=str(project_root() / "runs" / "detect"),
        name=name,
        exist_ok=True,
        amp=True,
        patience=15,
        seed=42,
    )
    return Path(results.save_dir) / "weights" / "best.pt"


def train_yolov8_variant(
    model_name: str,
    data_yaml: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: Optional[str],
    run_name: str,
) -> Path:
    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=select_device(device),
        project=str(project_root() / "runs" / "detect"),
        name=run_name,
        exist_ok=True,
        amp=True,
        patience=20,
        seed=42,
        cos_lr=True,
        optimizer="AdamW",
        lr0=0.001,
    )
    return Path(results.save_dir) / "weights" / "best.pt"


def faster_rcnn_smoke_forward(device: Optional[str] = None) -> str:
    dev = select_device(device)
    torch_device = torch.device("cuda:0" if str(dev).isdigit() and torch.cuda.is_available() else "cpu")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(torch_device)
    model.eval()
    dummy = [torch.rand(3, 640, 640, device=torch_device)]
    with torch.inference_mode():
        _ = model(dummy)
    logger.info("Faster R-CNN ResNet50 forward pass OK on %s", torch_device)
    return str(torch_device)


def compare_models(
    custom_weights: Path,
    baseline_weights: List[Tuple[str, Path]],
    data_yaml: Path,
    imgsz: int = 960,
    device: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    out_dir = output_dir or (project_root() / "outputs" / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    custom = evaluate_yolo(custom_weights, data_yaml, split="test", imgsz=imgsz, device=device)
    custom_row = {"name": "custom_yolov8", **{k: v for k, v in custom.items() if k != "per_class_ap50"}}
    custom_row["per_class_ap50"] = json.dumps(custom["per_class_ap50"])
    rows.append(custom_row)

    baseline_maps: List[float] = []
    for name, weights in baseline_weights:
        metrics = evaluate_yolo(weights, data_yaml, split="test", imgsz=imgsz, device=device)
        row = {"name": name, **{k: v for k, v in metrics.items() if k != "per_class_ap50"}}
        row["per_class_ap50"] = json.dumps(metrics["per_class_ap50"])
        rows.append(row)
        baseline_maps.append(metrics["map50"])

    best_baseline = max(baseline_maps) if baseline_maps else 0.0
    improvement = relative_improvement(custom["map50"], best_baseline)
    summary = {
        "custom_map50": custom["map50"],
        "best_baseline_map50": best_baseline,
        "relative_map50_improvement_pct": improvement,
        "rows": rows,
    }
    write_metrics_table(rows, out_dir / "comparison.csv", out_dir / "comparison.json")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Relative mAP@0.5 improvement vs best baseline: %.2f%%",
        improvement,
    )
    return summary
