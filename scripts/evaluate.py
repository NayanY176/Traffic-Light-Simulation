#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import project_root
from src.evaluate import evaluate_yolo, write_metrics_table
from src.logging_utils import configure_root, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="data/processed/traffic_lights/data.yaml")
    p.add_argument("--split", default="test")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="outputs/eval")
    return p.parse_args()


def main() -> None:
    configure_root()
    args = parse_args()
    root = project_root()
    metrics = evaluate_yolo(
        weights=args.weights,
        data_yaml=root / args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
    )
    out = root / args.out
    write_metrics_table([metrics], out / "metrics.csv", out / "metrics.json")
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
