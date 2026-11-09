#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import benchmark_backend, run_backend_comparison
from src.config import project_root
from src.logging_utils import configure_root, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--backend", default="pytorch", choices=["pytorch", "onnx", "tensorrt", "all"])
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--include-tensorrt", action="store_true")
    return p.parse_args()


def main() -> None:
    configure_root()
    args = parse_args()
    if args.backend == "all":
        payload = run_backend_comparison(
            weights=args.weights,
            imgsz=args.imgsz,
            device=args.device,
            include_tensorrt=args.include_tensorrt,
            warmup=args.warmup,
            iters=args.iters,
        )
        print(json.dumps(payload, indent=2))
        return
    result = benchmark_backend(
        weights=args.weights,
        backend=args.backend,
        imgsz=args.imgsz,
        width=args.width,
        height=args.height,
        warmup=args.warmup,
        iters=args.iters,
        device=args.device,
    )
    out = project_root() / "outputs" / "benchmark" / f"{args.backend}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
