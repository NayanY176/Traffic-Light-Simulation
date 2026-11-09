from __future__ import annotations

from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from src.device import select_device
from src.logging_utils import get_logger

logger = get_logger(__name__)


def export_onnx(weights: Path | str, imgsz: int = 960) -> Path:
    model = YOLO(str(weights))
    path = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=12)
    out = Path(path)
    logger.info("Exported ONNX -> %s", out)
    return out


def export_tensorrt_fp16(
    weights: Path | str,
    imgsz: int = 960,
    device: Optional[str] = None,
) -> Path:
    dev = select_device(device)
    model = YOLO(str(weights))
    path = model.export(format="engine", imgsz=imgsz, half=True, device=dev)
    out = Path(path)
    logger.info("Exported TensorRT FP16 -> %s", out)
    return out
