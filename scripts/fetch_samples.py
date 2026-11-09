#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.download import copy_sample_images
from src.logging_utils import configure_root, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dest", default="samples")
    p.add_argument("--raw", default="data/raw/traffic_lights")
    p.add_argument("--limit", type=int, default=6)
    return p.parse_args()


def main() -> None:
    configure_root()
    args = parse_args()
    raw = ROOT / args.raw
    if not (raw / "images" / "train").exists():
        raise SystemExit(
            "Raw dataset missing. Run: python scripts/download_dataset.py"
        )
    copied = copy_sample_images(raw, ROOT / args.dest, limit=args.limit)
    logger.info("Copied %s real traffic-light frames into %s", len(copied), args.dest)


if __name__ == "__main__":
    main()
