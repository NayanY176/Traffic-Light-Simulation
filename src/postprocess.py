from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


CLASS_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 255),
    1: (0, 220, 255),
    2: (0, 200, 0),
    3: (140, 140, 140),
}


def filter_by_confidence(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    conf_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(scores) == 0:
        return boxes_xyxy, scores, class_ids
    mask = scores >= conf_threshold
    return boxes_xyxy[mask], scores[mask], class_ids[mask]


def nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> List[int]:
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]
    return keep


def scale_boxes(
    boxes_xyxy: np.ndarray,
    gain: float,
    pad: Tuple[float, float],
) -> np.ndarray:
    if len(boxes_xyxy) == 0:
        return boxes_xyxy
    out = boxes_xyxy.copy()
    out[:, [0, 2]] -= pad[0]
    out[:, [1, 3]] -= pad[1]
    out[:, :4] /= gain
    return out


def detections_to_dicts(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for box, score, cls_id in zip(boxes_xyxy, scores, class_ids):
        cid = int(cls_id)
        name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        rows.append(
            {
                "xyxy": [float(v) for v in box.tolist()],
                "confidence": float(score),
                "class_id": cid,
                "class_name": name,
            }
        )
    return rows
