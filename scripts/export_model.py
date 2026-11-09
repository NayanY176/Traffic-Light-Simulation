#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.export import export_onnx, export_tensorrt_fp16
from src.logging_utils import configure_root, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--format", default="onnx", choices=["onnx", "engine", "both"])
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    configure_root()
    args = parse_args()
    if args.format in {"onnx", "both"}:
        path = export_onnx(args.weights, args.imgsz)
        logger.info("ONNX: %s", path)
    if args.format in {"engine", "both"}:
        path = export_tensorrt_fp16(args.weights, args.imgsz, args.device)
        logger.info("TensorRT: %s", path)


if __name__ == "__main__":
    main()
