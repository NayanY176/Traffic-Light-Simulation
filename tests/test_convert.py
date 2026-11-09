from __future__ import annotations

from pathlib import Path

from src.data.convert import (
    CanonicalClasses,
    format_yolo_line,
    parse_yolo_line,
    remap_label_file,
    remap_label_lines,
    resolve_class_id,
    xyxy_to_yolo,
    yolo_to_xyxy,
)


def test_resolve_class_id_aliases() -> None:
    aliases = {"Red": "red", "amber": "yellow", "GREEN": "green", "inactive": "off"}
    assert resolve_class_id("Red", aliases) == 0
    assert resolve_class_id("amber", aliases) == 1
    assert resolve_class_id("GREEN", aliases) == 2
    assert resolve_class_id("inactive", aliases) == 3
    assert resolve_class_id("unknown", aliases) is None


def test_parse_and_format_roundtrip() -> None:
    line = format_yolo_line(2, 0.5, 0.4, 0.1, 0.2)
    parsed = parse_yolo_line(line)
    assert parsed == (2, 0.5, 0.4, 0.1, 0.2)


def test_xyxy_yolo_roundtrip() -> None:
    xc, yc, w, h = xyxy_to_yolo(100, 50, 200, 150, 400, 300)
    x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, 400, 300)
    assert abs(x1 - 100) < 1e-6
    assert abs(y1 - 50) < 1e-6
    assert abs(x2 - 200) < 1e-6
    assert abs(y2 - 150) < 1e-6


def test_remap_label_lines_and_file(tmp_path: Path) -> None:
    aliases = {"red": "red", "yellow": "yellow", "green": "green", "blue": "off"}
    source_names = ["red", "yellow", "green", "blue"]
    lines = ["0 0.1 0.2 0.3 0.4", "3 0.5 0.5 0.2 0.2", "9 0.1 0.1 0.1 0.1"]
    remapped = remap_label_lines(lines, source_names, aliases, CanonicalClasses())
    assert len(remapped) == 2
    assert remapped[0].startswith("0 ")
    assert remapped[1].startswith("3 ")

    src = tmp_path / "a.txt"
    dst = tmp_path / "b.txt"
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    count = remap_label_file(src, dst, source_names, aliases)
    assert count == 2
    assert dst.exists()
