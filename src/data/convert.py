from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CANONICAL_ORDER: Tuple[str, ...] = ("red", "yellow", "green", "off")


@dataclass(frozen=True)
class CanonicalClasses:
    names: Tuple[str, ...] = CANONICAL_ORDER

    @property
    def name_to_id(self) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(self.names)}

    @property
    def id_to_name(self) -> Dict[int, str]:
        return {idx: name for idx, name in enumerate(self.names)}


def resolve_class_id(
    raw_name: str,
    aliases: Mapping[str, str],
    canonical: CanonicalClasses | None = None,
) -> Optional[int]:
    classes = canonical or CanonicalClasses()
    key = raw_name.strip()
    mapped = aliases.get(key, aliases.get(key.lower(), key.lower()))
    return classes.name_to_id.get(mapped)


def parse_yolo_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = int(float(parts[0]))
    xc, yc, w, h = map(float, parts[1:5])
    return cls_id, xc, yc, w, h


def format_yolo_line(cls_id: int, xc: float, yc: float, w: float, h: float) -> str:
    return f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def remap_label_lines(
    lines: Iterable[str],
    source_names: Sequence[str],
    aliases: Mapping[str, str],
    canonical: CanonicalClasses | None = None,
) -> List[str]:
    classes = canonical or CanonicalClasses()
    out: List[str] = []
    for line in lines:
        parsed = parse_yolo_line(line)
        if parsed is None:
            continue
        src_id, xc, yc, w, h = parsed
        if src_id < 0 or src_id >= len(source_names):
            continue
        dst_id = resolve_class_id(source_names[src_id], aliases, classes)
        if dst_id is None:
            continue
        out.append(format_yolo_line(dst_id, xc, yc, w, h))
    return out


def remap_label_file(
    src_path: Path,
    dst_path: Path,
    source_names: Sequence[str],
    aliases: Mapping[str, str],
    canonical: CanonicalClasses | None = None,
) -> int:
    text = src_path.read_text(encoding="utf-8") if src_path.exists() else ""
    lines = remap_label_lines(text.splitlines(), source_names, aliases, canonical)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def xyxy_to_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: float,
    img_h: float,
) -> Tuple[float, float, float, float]:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    return xc, yc, bw / img_w, bh / img_h


def yolo_to_xyxy(
    xc: float,
    yc: float,
    w: float,
    h: float,
    img_w: float,
    img_h: float,
) -> Tuple[float, float, float, float]:
    bw = w * img_w
    bh = h * img_h
    cx = xc * img_w
    cy = yc * img_h
    return cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0
