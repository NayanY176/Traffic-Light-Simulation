from __future__ import annotations

import numpy as np

from src.postprocess import (
    detections_to_dicts,
    filter_by_confidence,
    nms_xyxy,
    scale_boxes,
)


def test_filter_by_confidence() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 5, 5]], dtype=float)
    scores = np.array([0.9, 0.1], dtype=float)
    classes = np.array([0, 1])
    b, s, c = filter_by_confidence(boxes, scores, classes, 0.25)
    assert len(s) == 1
    assert int(c[0]) == 0


def test_nms_xyxy() -> None:
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [50, 50, 60, 60],
        ],
        dtype=float,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=float)
    keep = nms_xyxy(boxes, scores, iou_threshold=0.5)
    assert 0 in keep
    assert 2 in keep
    assert len(keep) == 2


def test_scale_boxes_and_dicts() -> None:
    boxes = np.array([[10.0, 20.0, 30.0, 40.0]])
    scaled = scale_boxes(boxes, gain=2.0, pad=(2.0, 4.0))
    assert np.allclose(scaled, np.array([[4.0, 8.0, 14.0, 18.0]]))
    rows = detections_to_dicts(
        np.array([[1.0, 2.0, 3.0, 4.0]]),
        np.array([0.88]),
        np.array([2]),
        ["red", "yellow", "green", "off"],
    )
    assert rows[0]["class_name"] == "green"
    assert rows[0]["confidence"] == 0.88
