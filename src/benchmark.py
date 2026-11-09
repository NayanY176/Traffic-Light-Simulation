from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from src.config import project_root
from src.device import device_label, select_device, synchronize
from src.logging_utils import get_logger

logger = get_logger(__name__)


def _latency_ms(model: YOLO, frame: np.ndarray, device: str, imgsz: int) -> float:
    synchronize(device)
    t0 = time.perf_counter()
    model.predict(frame, imgsz=imgsz, device=device, verbose=False)
    synchronize(device)
    return (time.perf_counter() - t0) * 1000.0


def benchmark_backend(
    weights: str,
    backend: str,
    imgsz: int = 960,
    width: int = 1920,
    height: int = 1080,
    warmup: int = 20,
    iters: int = 100,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    dev = select_device(device)
    model_path = weights
    export_dir = project_root() / "weights" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    if backend == "pytorch":
        model = YOLO(weights)
    elif backend == "onnx":
        base = YOLO(weights)
        exported = base.export(format="onnx", imgsz=imgsz, half=False, simplify=True)
        model_path = str(exported)
        model = YOLO(model_path)
    elif backend == "tensorrt":
        if not (str(dev).isdigit() and __import__("torch").cuda.is_available()):
            raise RuntimeError("TensorRT FP16 export/benchmark requires CUDA")
        base = YOLO(weights)
        exported = base.export(format="engine", imgsz=imgsz, half=True, device=dev)
        model_path = str(exported)
        model = YOLO(model_path)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    for _ in range(warmup):
        model.predict(frame, imgsz=imgsz, device=dev, verbose=False)
    synchronize(dev)

    latencies: List[float] = []
    for _ in range(iters):
        latencies.append(_latency_ms(model, frame, dev, imgsz))

    mean_ms = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    result = {
        "backend": backend,
        "weights": model_path,
        "device": device_label(dev),
        "resolution": f"{width}x{height}",
        "imgsz": imgsz,
        "warmup": warmup,
        "iters": iters,
        "mean_latency_ms": mean_ms,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "fps": fps,
    }
    logger.info(
        "%s | %.1f FPS | mean=%.2fms p50=%.2fms p95=%.2fms",
        backend,
        fps,
        mean_ms,
        p50,
        p95,
    )
    return result


def run_backend_comparison(
    weights: str,
    imgsz: int = 960,
    device: Optional[str] = None,
    include_tensorrt: bool = True,
    warmup: int = 20,
    iters: int = 100,
) -> Dict[str, Any]:
    backends = ["pytorch", "onnx"]
    results: List[Dict[str, Any]] = []
    for backend in backends:
        try:
            results.append(
                benchmark_backend(
                    weights=weights,
                    backend=backend,
                    imgsz=imgsz,
                    device=device,
                    warmup=warmup,
                    iters=iters,
                )
            )
        except Exception as exc:
            logger.warning("Backend %s failed: %s", backend, exc)
            results.append({"backend": backend, "error": str(exc)})
    if include_tensorrt:
        try:
            results.append(
                benchmark_backend(
                    weights=weights,
                    backend="tensorrt",
                    imgsz=imgsz,
                    device=device,
                    warmup=warmup,
                    iters=iters,
                )
            )
        except Exception as exc:
            logger.warning("Backend tensorrt failed: %s", exc)
            results.append({"backend": "tensorrt", "error": str(exc)})

    out_dir = project_root() / "outputs" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"results": results}
    out_path = out_dir / "backend_comparison.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    return payload
