#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.logging_utils import configure_root, get_logger
from src.train import load_train_configs, train_yolov8

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--hyp", default="configs/hyp.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    configure_root()
    args = parse_args()
    train_cfg, hyp_cfg = load_train_configs(args.config, args.hyp)
    overrides = {
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "name": args.name,
        "resume": args.resume or None,
    }
    best = train_yolov8(train_cfg, hyp_cfg, overrides)
    logger.info("Training finished: %s", best)


if __name__ == "__main__":
    main()
