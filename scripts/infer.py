#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import load_model, run_image, run_stream
from src.logging_utils import configure_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--source", default="samples")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default=None)
    p.add_argument("--save", default=None)
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main() -> None:
    configure_root()
    args = parse_args()
    model, device = load_model(args.weights, args.device)
    source = args.source
    path = Path(source)
    show = args.show and not args.no_show
    if path.exists() and path.is_dir():
        images = sorted(
            [p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        )
        if not images:
            raise SystemExit(f"No images in {path}")
        out_dir = Path(args.save) if args.save else Path("outputs") / "infer"
        out_dir.mkdir(parents=True, exist_ok=True)
        for image in images:
            run_image(
                model,
                image,
                device,
                args.conf,
                args.imgsz,
                save_path=out_dir / f"det_{image.name}",
                show=show,
            )
        return
    if path.exists() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        save = Path(args.save) if args.save else Path("outputs") / f"det_{path.name}"
        run_image(model, path, device, args.conf, args.imgsz, save_path=save, show=show or not args.no_show)
        return
    save = Path(args.save) if args.save else None
    run_stream(model, source, device, args.conf, args.imgsz, save_path=save, show=not args.no_show)


if __name__ == "__main__":
    main()
