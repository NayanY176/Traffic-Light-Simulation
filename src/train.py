from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

from src.config import load_yaml, merge_dict, project_root
from src.device import select_device
from src.logging_utils import get_logger
from src.seed import set_seed

logger = get_logger(__name__)


def train_yolov8(
    train_cfg: Dict[str, Any],
    hyp_cfg: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Path:
    cfg = merge_dict(train_cfg, overrides or {})
    set_seed(int(cfg.get("seed", 42)))
    device = select_device(cfg.get("device"))
    data = cfg.get("data", "configs/dataset.yaml")
    data_path = Path(data)
    if not data_path.is_absolute():
        data_path = project_root() / data_path
    processed = project_root() / "data" / "processed" / "traffic_lights" / "data.yaml"
    if processed.exists():
        data_path = processed

    model_name = cfg.get("model", "yolov8n.pt")
    logger.info("Training %s on %s device=%s", model_name, data_path, device)
    model = YOLO(model_name)

    train_kwargs: Dict[str, Any] = {
        "data": str(data_path),
        "epochs": int(cfg.get("epochs", 50)),
        "imgsz": int(cfg.get("imgsz", 960)),
        "batch": int(cfg.get("batch", 8)),
        "device": device,
        "workers": int(cfg.get("workers", 8)),
        "patience": int(cfg.get("patience", 20)),
        "optimizer": cfg.get("optimizer", "AdamW"),
        "lr0": float(cfg.get("lr0", 0.001)),
        "lrf": float(cfg.get("lrf", 0.01)),
        "momentum": float(cfg.get("momentum", 0.937)),
        "weight_decay": float(cfg.get("weight_decay", 0.0005)),
        "warmup_epochs": float(cfg.get("warmup_epochs", 3.0)),
        "cos_lr": bool(cfg.get("cos_lr", True)),
        "amp": bool(cfg.get("amp", True)),
        "project": str(project_root() / cfg.get("project", "runs/detect")),
        "name": cfg.get("name", "traffic_lights_yolov8"),
        "exist_ok": bool(cfg.get("exist_ok", True)),
        "pretrained": bool(cfg.get("pretrained", True)),
        "close_mosaic": int(cfg.get("close_mosaic", 10)),
        "save": bool(cfg.get("save", True)),
        "save_period": int(cfg.get("save_period", 10)),
        "plots": bool(cfg.get("plots", True)),
        "val": bool(cfg.get("val", True)),
        "seed": int(cfg.get("seed", 42)),
        "fraction": float(cfg.get("fraction", 1.0)),
        "resume": bool(cfg.get("resume", False)),
    }
    if hyp_cfg:
        train_kwargs.update(hyp_cfg)
    if cfg.get("freeze") is not None:
        train_kwargs["freeze"] = cfg["freeze"]

    results = model.train(**train_kwargs)
    best = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("Best checkpoint: %s", best)
    return best


def load_train_configs(
    train_yaml: Path | str,
    hyp_yaml: Path | str | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    root = project_root()
    train_cfg = load_yaml(root / train_yaml if not Path(train_yaml).is_absolute() else train_yaml)
    hyp_cfg: Dict[str, Any] = {}
    if hyp_yaml:
        hyp_path = root / hyp_yaml if not Path(hyp_yaml).is_absolute() else Path(hyp_yaml)
        hyp_cfg = load_yaml(hyp_path)
    return train_cfg, hyp_cfg
