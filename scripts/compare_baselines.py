#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import compare_models, faster_rcnn_smoke_forward, train_yolov5_baseline, train_yolov8_variant
from src.config import project_root
from src.logging_utils import configure_root, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/traffic_lights/data.yaml")
    p.add_argument("--custom-weights", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--fasterrcnn-smoke", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    configure_root()
    args = parse_args()
    root = project_root()
    data = root / args.data
    if not data.exists():
        raise SystemExit(f"Missing dataset yaml: {data}")

    if args.fasterrcnn_smoke:
        faster_rcnn_smoke_forward(args.device)

    if args.skip_train:
        if not args.custom_weights:
            raise SystemExit("--custom-weights required with --skip-train")
        custom = Path(args.custom_weights)
        baselines = []
        y5 = root / "runs" / "detect" / "baseline_yolov5s" / "weights" / "best.pt"
        y8s = root / "runs" / "detect" / "baseline_yolov8s" / "weights" / "best.pt"
        if y5.exists():
            baselines.append(("baseline_yolov5s", y5))
        if y8s.exists():
            baselines.append(("baseline_yolov8s", y8s))
        if not baselines:
            raise SystemExit("No baseline weights found under runs/detect")
    else:
        custom = train_yolov8_variant(
            "yolov8n.pt",
            data,
            args.epochs,
            args.imgsz,
            args.batch,
            args.device,
            "custom_yolov8n",
        )
        if args.custom_weights:
            custom = Path(args.custom_weights)
        y5 = train_yolov5_baseline(data, args.epochs, args.imgsz, args.batch, args.device)
        y8s = train_yolov8_variant(
            "yolov8s.pt",
            data,
            args.epochs,
            args.imgsz,
            args.batch,
            args.device,
            "baseline_yolov8s",
        )
        baselines = [("baseline_yolov5s", y5), ("baseline_yolov8s", y8s)]

    summary = compare_models(custom, baselines, data, imgsz=args.imgsz, device=args.device)
    logger.info("Comparison summary: %s", summary)


if __name__ == "__main__":
    main()
