from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import yaml

from src.data.convert import CanonicalClasses, remap_label_file
from src.data.download import find_data_yaml
from src.logging_utils import get_logger

logger = get_logger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_yolo_layout(root: Path) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for split in ("train", "val", "test"):
        img = root / "images" / split
        lbl = root / "labels" / split
        img.mkdir(parents=True, exist_ok=True)
        lbl.mkdir(parents=True, exist_ok=True)
        paths[f"images_{split}"] = img
        paths[f"labels_{split}"] = lbl
    return paths


def read_source_names(raw_root: Path) -> List[str]:
    yaml_path = find_data_yaml(raw_root)
    with yaml_path.open("r", encoding="utf-8") as fh:
        meta = yaml.safe_load(fh)
    names = meta.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, list):
        return list(names)
    return ["red", "yellow", "green", "off"]


def _pairs_from_images_labels(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    if not images_dir.exists():
        return []
    pairs: List[Tuple[Path, Path]] = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        pairs.append((img, labels_dir / f"{img.stem}.txt"))
    return pairs


def _pairs_from_split_folder(folder: Path) -> List[Tuple[Path, Path]]:
    return _pairs_from_images_labels(folder / "images", folder / "labels")


def discover_splits(raw_root: Path) -> Dict[str, List[Tuple[Path, Path]]]:
    out: Dict[str, List[Tuple[Path, Path]]] = {}

    flat = {
        "train": _pairs_from_images_labels(raw_root / "images" / "train", raw_root / "labels" / "train"),
        "val": _pairs_from_images_labels(raw_root / "images" / "val", raw_root / "labels" / "val"),
        "test": _pairs_from_images_labels(raw_root / "images" / "test", raw_root / "labels" / "test"),
    }
    if any(flat.values()):
        return {k: v for k, v in flat.items() if v}

    yaml_path = find_data_yaml(raw_root)
    base = yaml_path.parent
    mapping = {
        "train": base / "train",
        "val": base / "valid" if (base / "valid").exists() else base / "val",
        "test": base / "test",
    }
    for split, folder in mapping.items():
        if folder.exists():
            pairs = _pairs_from_split_folder(folder)
            if pairs:
                out[split] = pairs
    return out


def materialize_processed_dataset(
    raw_root: Path,
    processed_root: Path,
    aliases: Mapping[str, str],
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, int]:
    if processed_root.exists():
        shutil.rmtree(processed_root)
    ensure_yolo_layout(processed_root)
    source_names = read_source_names(raw_root)
    splits = discover_splits(raw_root)
    canonical = CanonicalClasses()
    counts = {"train": 0, "val": 0, "test": 0}

    def write_pair(split: str, img: Path, label: Path) -> None:
        img_dst = processed_root / "images" / split / img.name
        lbl_dst = processed_root / "labels" / split / f"{img.stem}.txt"
        shutil.copy2(img, img_dst)
        remap_label_file(label, lbl_dst, source_names, aliases, canonical)
        counts[split] += 1

    if "train" in splits and "val" in splits:
        for split, pairs in splits.items():
            if split not in counts:
                continue
            for img, label in pairs:
                write_pair(split, img, label)
        if counts["test"] == 0 and counts["val"] > 1:
            val_imgs = list((processed_root / "images" / "val").glob("*"))
            rng = random.Random(seed)
            rng.shuffle(val_imgs)
            move_n = max(1, len(val_imgs) // 3)
            for img in val_imgs[:move_n]:
                lbl = processed_root / "labels" / "val" / f"{img.stem}.txt"
                img.rename(processed_root / "images" / "test" / img.name)
                if lbl.exists():
                    lbl.rename(processed_root / "labels" / "test" / f"{img.stem}.txt")
                counts["val"] -= 1
                counts["test"] += 1
    else:
        all_pairs: List[Tuple[Path, Path]] = []
        for pairs in splits.values():
            all_pairs.extend(pairs)
        rng = random.Random(seed)
        rng.shuffle(all_pairs)
        n = len(all_pairs)
        n_test = max(1, int(n * test_ratio)) if n else 0
        n_val = max(1, int(n * val_ratio)) if n else 0
        buckets = {
            "test": all_pairs[:n_test],
            "val": all_pairs[n_test : n_test + n_val],
            "train": all_pairs[n_test + n_val :],
        }
        for split, pairs in buckets.items():
            for img, label in pairs:
                write_pair(split, img, label)

    path_str = processed_root.resolve().as_posix()
    lines = [
        f'path: "{path_str}"',
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(canonical.names)}",
        "names:",
    ]
    for idx, name in enumerate(canonical.names):
        safe = f'"{name}"' if name.lower() in {"off", "on", "yes", "no", "true", "false"} else name
        lines.append(f"  {idx}: {safe}")
    (processed_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Processed dataset counts: %s", counts)
    return counts
