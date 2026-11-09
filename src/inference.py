from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from src.device import device_label, select_device
from src.logging_utils import get_logger
from src.postprocess import CLASS_COLORS_BGR

logger = get_logger(__name__)


class FpsMeter:
    def __init__(self, window: int = 30) -> None:
        self._deltas: Deque[float] = deque(maxlen=window)
        self._last: Optional[float] = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            self._deltas.append(now - self._last)
        self._last = now
        if not self._deltas:
            return 0.0
        mean = sum(self._deltas) / len(self._deltas)
        return 0.0 if mean <= 0 else 1.0 / mean


def draw_detections(
    frame: np.ndarray,
    result,
    conf_threshold: float = 0.25,
) -> Tuple[np.ndarray, int]:
    out = frame.copy()
    count = 0
    if result.boxes is None:
        return out, 0
    names = result.names
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = CLASS_COLORS_BGR.get(cls_id, (255, 180, 0))
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        text = f"{label} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, text, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        count += 1
    return out, count


def overlay_hud(frame: np.ndarray, fps: float, device: str, detections: int) -> np.ndarray:
    lines = [
        f"FPS: {fps:.1f}" if fps > 0 else "FPS: --",
        f"Device: {device_label(device)}",
        f"Lights: {detections}",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1)
        y += 28
    return frame


def run_image(
    model: YOLO,
    image_path: Path,
    device: str,
    conf: float,
    imgsz: int,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    result = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    annotated, n = draw_detections(frame, result, conf)
    annotated = overlay_hud(annotated, fps=0.0, device=device, detections=n)
    out = save_path or (Path("outputs") / f"det_{image_path.name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), annotated)
    logger.info("Saved %s (%s detections)", out, n)
    if show:
        cv2.imshow("Traffic Light Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return out


def run_stream(
    model: YOLO,
    source: str,
    device: str,
    conf: float,
    imgsz: int,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    cap_source: object = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    writer = None
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))
    meter = FpsMeter()
    logger.info("Streaming %s on %s", source, device_label(device))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        annotated, n = draw_detections(frame, result, conf)
        fps = meter.tick()
        annotated = overlay_hud(annotated, fps=fps, device=device, detections=n)
        if writer is not None:
            writer.write(annotated)
        if show:
            cv2.imshow("Traffic Light Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def load_model(weights: str, device: Optional[str] = None) -> tuple[YOLO, str]:
    dev = select_device(device)
    model = YOLO(weights)
    return model, dev
