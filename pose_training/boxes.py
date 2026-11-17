"""Class-agnostic person-box cleanup before top-down pose.

The YOLO person detector (and rtmlib's RTMDet) can leave overlapping or nested
boxes that the default IoU-NMS misses: a spurious partial box that sits mostly
*inside* a real person's box shares only moderate IoU with it, so IoU-NMS keeps
both, and each gets posed -> the same person gets several stacked skeletons.

``clean_detection_boxes`` removes those with a greedy, confidence-ordered pass
that suppresses a box when, against any higher-confidence kept box, either the
IoU exceeds ``iou_thr`` OR the box is mostly contained (intersection-over-its-
own-area exceeds ``ioa_thr``). Two genuinely distinct overlapping people
(parent + child) keep low containment, so both survive.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def box_iou_ioa(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Return (IoU, IoA) where IoA is intersection over the area of box ``a``."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0, 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    iou = inter / union if union > 0.0 else 0.0
    ioa = inter / area_a if area_a > 0.0 else 0.0
    return iou, ioa


def clean_detection_boxes(
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    iou_thr: float = 0.6,
    ioa_thr: float = 0.7,
) -> List[int]:
    """Greedy class-agnostic NMS using IoU and containment.

    Args:
        boxes_xyxy: (n, 4) person boxes.
        confs: (n,) confidences.
        iou_thr: Suppress a box whose IoU with a kept box exceeds this.
        ioa_thr: Suppress a box that is at least this fraction contained within
            a kept (higher-confidence) box.

    Returns:
        Indices of kept boxes, highest confidence first.
    """
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    confs = np.asarray(confs, dtype=float).reshape(-1)
    order = [int(i) for i in np.argsort(confs)[::-1]]
    kept: List[int] = []
    for index in order:
        suppressed = False
        for kept_index in kept:
            iou, ioa = box_iou_ioa(boxes_xyxy[index], boxes_xyxy[kept_index])
            if iou > iou_thr or ioa > ioa_thr:
                suppressed = True
                break
        if not suppressed:
            kept.append(index)
    return kept
