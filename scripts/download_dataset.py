#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_yaml, project_root
from src.data.download import (
    copy_sample_images,
    download_huggingface_subset,
    download_roboflow_dataset,
    download_zip,
)
from src.logging_utils import configure_root, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dataset.yaml")
    p.add_argument("--api-key", default=None)
    p.add_argument("--zip-url", default=None)
    p.add_argument("--provider", default=None, choices=["huggingface", "roboflow", None])
    p.add_argument("--dest", default="data/raw/traffic_lights")
    p.add_argument("--train-count", type=int, default=None)
    p.add_argument("--val-count", type=int, default=None)
    p.add_argument("--test-count", type=int, default=None)
    return p.parse_args()


def main() -> None:
    configure_root()
    args = parse_args()
    root = project_root()
    cfg = load_yaml(root / args.config)
    dest = root / args.dest
    source = cfg.get("source", {})
    provider = args.provider or source.get("provider", "huggingface")

    if args.zip_url:
        path = download_zip(args.zip_url, dest)
    elif provider == "huggingface":
        path = download_huggingface_subset(
            repo_id=source["repo_id"],
            dest_dir=dest,
            train_count=args.train_count or int(source.get("train_count", 120)),
            val_count=args.val_count or int(source.get("val_count", 40)),
            test_count=args.test_count or int(source.get("test_count", 20)),
        )
        copied = copy_sample_images(path, root / "samples", limit=6)
        logger.info("Copied demo samples: %s", [str(p.name) for p in copied])
    else:
        rf = source.get("roboflow_fallback", source)
        path = download_roboflow_dataset(
            workspace=rf["workspace"],
            project=rf["project"],
            version=int(rf["version"]),
            dest_dir=dest,
            api_key=args.api_key,
            fmt=rf.get("format", "yolov8"),
        )

    marker = dest / "DOWNLOAD_PATH.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(path), encoding="utf-8")
    logger.info("Raw dataset ready at %s", path)


if __name__ == "__main__":
    main()
