#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_yaml, project_root
from src.data.splits import materialize_processed_dataset
from src.logging_utils import configure_root, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dataset.yaml")
    p.add_argument("--raw", default="data/raw/traffic_lights")
    p.add_argument("--processed", default="data/processed/traffic_lights")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _resolve_raw_root(raw: Path) -> Path:
    marker = raw / "DOWNLOAD_PATH.txt"
    if marker.exists():
        return Path(marker.read_text(encoding="utf-8").strip())
    candidates = list(raw.rglob("data.yaml"))
    if candidates:
        return candidates[0].parent
    return raw


def main() -> None:
    configure_root()
    args = parse_args()
    root = project_root()
    cfg = load_yaml(root / args.config)
    aliases = cfg.get("source", {}).get("class_aliases", {})
    raw_root = _resolve_raw_root(root / args.raw)
    processed = root / args.processed
    counts = materialize_processed_dataset(
        raw_root=raw_root,
        processed_root=processed,
        aliases=aliases,
        seed=args.seed,
    )
    logger.info("Prepared dataset at %s counts=%s", processed, counts)


if __name__ == "__main__":
    main()
