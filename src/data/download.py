from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Sequence

import requests
import yaml

from src.logging_utils import get_logger

logger = get_logger(__name__)


def load_dataset_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def download_roboflow_dataset(
    workspace: str,
    project: str,
    version: int,
    dest_dir: Path,
    api_key: Optional[str] = None,
    fmt: str = "yolov8",
) -> Path:
    key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is required. Create a free key at https://roboflow.com "
            "and export ROBOFLOW_API_KEY=..."
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    from roboflow import Roboflow

    logger.info("Downloading Roboflow dataset %s/%s v%s", workspace, project, version)
    rf = Roboflow(api_key=key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download(fmt, location=str(dest_dir / "roboflow_dl"))
    location = Path(dataset.location)
    logger.info("Downloaded to %s", location)
    return location


def download_zip(url: str, dest_dir: Path, extract: bool = True) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "dataset.zip"
    logger.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with zip_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    if not extract:
        return zip_path
    extract_dir = dest_dir / "extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    logger.info("Extracted to %s", extract_dir)
    return extract_dir


def download_huggingface_subset(
    repo_id: str,
    dest_dir: Path,
    train_count: int = 120,
    val_count: int = 40,
    test_count: int = 20,
) -> Path:
    from huggingface_hub import hf_hub_download, list_repo_files

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    for split in ("train", "val", "test"):
        (dest_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    files = list_repo_files(repo_id, repo_type="dataset")
    train_imgs = sorted(f for f in files if f.startswith("images/train/") and f.endswith(".jpg"))
    val_imgs = sorted(f for f in files if f.startswith("images/val/") and f.endswith(".jpg"))
    if not train_imgs:
        raise RuntimeError(f"No train images found in {repo_id}")
    if not val_imgs:
        raise RuntimeError(f"No val images found in {repo_id}")

    selected = {
        "train": train_imgs[:train_count],
        "val": val_imgs[:val_count],
        "test": val_imgs[val_count : val_count + test_count]
        if len(val_imgs) > val_count
        else val_imgs[: max(1, min(test_count, len(val_imgs) // 4))],
    }

    def pull(rel: str, dest: Path) -> None:
        src = hf_hub_download(repo_id, rel, repo_type="dataset")
        shutil.copy2(src, dest)

    for split, rels in selected.items():
        for rel in rels:
            name = Path(rel).name
            stem = Path(name).stem
            label_rel = f"labels/{'val' if 'val' in rel or split != 'train' else 'train'}/{stem}.txt"
            if split == "train":
                label_rel = f"labels/train/{stem}.txt"
            else:
                label_rel = f"labels/val/{stem}.txt"
            pull(rel, dest_dir / "images" / split / name)
            pull(label_rel, dest_dir / "labels" / split / f"{stem}.txt")
        logger.info("Downloaded %s %s images", len(rels), split)

    path_str = dest_dir.resolve().as_posix()
    lines = [
        f'path: "{path_str}"',
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "nc: 4",
        "names:",
        "  0: red",
        "  1: yellow",
        "  2: green",
        '  3: "off"',
    ]
    (dest_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Hugging Face subset ready at %s", dest_dir)
    return dest_dir


def find_data_yaml(root: Path) -> Path:
    candidates = list(root.rglob("data.yaml")) + list(root.rglob("dataset.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No data.yaml found under {root}")
    return candidates[0]


def copy_sample_images(raw_root: Path, sample_dir: Path, limit: int = 6) -> Sequence[Path]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    for stale in sample_dir.glob("traffic_light_*"):
        stale.unlink()
    images = sorted((raw_root / "images" / "train").glob("*.jpg"))[:limit]
    copied = []
    for idx, img in enumerate(images):
        dest = sample_dir / f"traffic_light_real_{idx:02d}.jpg"
        shutil.copy2(img, dest)
        copied.append(dest)
    return copied
