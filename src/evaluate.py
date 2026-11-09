from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from ultralytics import YOLO

from src.config import project_root
from src.device import select_device
from src.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_yolo(
    weights: Path | str,
    data_yaml: Path | str,
    split: str = "test",
    imgsz: int = 960,
    device: Optional[str] = None,
    conf: float = 0.001,
    iou: float = 0.6,
) -> Dict[str, Any]:
    model = YOLO(str(weights))
    data_path = Path(data_yaml)
    if not data_path.is_absolute():
        data_path = project_root() / data_path
    metrics = model.val(
        data=str(data_path),
        split=split,
        imgsz=imgsz,
        device=select_device(device),
        conf=conf,
        iou=iou,
        plots=True,
    )
    names = metrics.names if isinstance(metrics.names, dict) else {i: n for i, n in enumerate(metrics.names)}
    per_class: Dict[str, float] = {}
    if getattr(metrics.box, "ap50", None) is not None:
        ap50 = metrics.box.ap50
        for idx, value in enumerate(ap50):
            per_class[str(names.get(idx, idx))] = float(value)
    result = {
        "model": str(weights),
        "split": split,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class_ap50": per_class,
    }
    logger.info(
        "Eval %s | mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f",
        weights,
        result["map50"],
        result["map50_95"],
        result["precision"],
        result["recall"],
    )
    return result


def relative_improvement(custom_map: float, baseline_map: float) -> float:
    if baseline_map <= 0:
        return 0.0
    return (custom_map - baseline_map) / baseline_map * 100.0


def write_metrics_table(
    rows: List[Dict[str, Any]],
    output_csv: Path,
    output_json: Path,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logger.info("Wrote %s and %s", output_csv, output_json)
    return frame
