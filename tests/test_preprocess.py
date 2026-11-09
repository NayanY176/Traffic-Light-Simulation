from __future__ import annotations

from pathlib import Path

import yaml

from src.data.splits import ensure_yolo_layout, materialize_processed_dataset


def _write_mini_roboflow(root: Path) -> None:
    for split, folder in (("train", "train"), ("valid", "valid"), ("test", "test")):
        img_dir = root / folder / "images"
        lbl_dir = root / folder / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        img = img_dir / f"{split}_0.jpg"
        img.write_bytes(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
        )
        (lbl_dir / f"{split}_0.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    meta = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": ["red", "yellow", "green"],
    }
    with (root / "data.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh)


def test_ensure_yolo_layout(tmp_path: Path) -> None:
    paths = ensure_yolo_layout(tmp_path / "processed")
    assert paths["images_train"].exists()
    assert paths["labels_val"].exists()


def test_materialize_processed_dataset(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    _write_mini_roboflow(raw)
    aliases = {"red": "red", "yellow": "yellow", "green": "green"}
    counts = materialize_processed_dataset(raw, processed, aliases, seed=0)
    assert counts["train"] >= 1
    assert counts["val"] >= 1
    assert (processed / "data.yaml").exists()
    label = next((processed / "labels" / "train").glob("*.txt"))
    assert label.read_text(encoding="utf-8").startswith("0 ")
