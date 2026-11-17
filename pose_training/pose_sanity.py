"""Geometric sanity cleaning for (especially top-down / fused) keypoints.

Top-down pose on overlapping person boxes can grab the wrong person's joints
(e.g. one person's box steals another's face), producing anatomically
impossible bones and two people sharing a keypoint. These checks make the
labels robust:

1. Anatomical distance: demote any keypoint implausibly far from the person's
   own body center (relative to body scale).
2. Cross-instance de-duplication: when two people claim the same point, keep it
   for the anatomically-consistent owner and demote it for the other.
3. Box tightening: refit each box to its surviving keypoints so overlapping
   boxes separate and the box matches the real person.

All functions operate on the shared person-record schema (``bbox_xyxy``,
``box_conf``, ``keypoints`` = 17 x [x, y, conf]). Demotion sets a keypoint to
[0, 0, 0] so the downstream filters drop it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import hypot
from typing import Any, Dict, List, Optional, Tuple

HEAD_KEYPOINTS = {0, 1, 2, 3, 4}
TORSO_KEYPOINTS = (5, 6, 11, 12)


@dataclass
class CleanConfig:
    """Thresholds for geometric keypoint cleaning."""

    conf_floor: float = 0.2
    # Max distance of a HEAD keypoint from the body center, as a multiple of
    # body scale. Only head keypoints are distance-checked (the face-steal
    # case); limbs legitimately extend far and must never be distance-demoted.
    head_max_ratio: float = 2.2
    # Two keypoints within this fraction of body scale are "the same point".
    overlap_frac: float = 0.12
    tighten_margin: float = 0.12
    min_box_px: float = 10.0
    # Demote a keypoint this far (fraction of image size) outside the frame.
    offimage_margin_frac: float = 0.02
    # Left/right segments of the same limb whose length ratio exceeds this are
    # implausibly asymmetric; the longer side's distal joint is demoted.
    max_asym_ratio: float = 3.0
    # Two persons sharing at least this fraction of (closely-coinciding)
    # keypoints are the same detection; the lower-confidence one is dropped.
    nms_overlap_frac: float = 0.1
    nms_sim_thresh: float = 0.6
    # A duplicate must ALSO have near-identical boxes, so two distinct but
    # overlapping people (parent + child) are never merged.
    nms_box_iou: float = 0.8


# Symmetric limb segments as (proximal, distal) for left and right sides.
SYMMETRIC_SEGMENTS = (
    ((5, 7), (6, 8)),  # upper arm
    ((7, 9), (8, 10)),  # forearm
    ((11, 13), (12, 14)),  # thigh
    ((13, 15), (14, 16)),  # shin
)


def _present(
    keypoints: List[List[float]], conf_floor: float
) -> Dict[int, Tuple[float, float]]:
    return {k: (kp[0], kp[1]) for k, kp in enumerate(keypoints) if kp[2] >= conf_floor}


def body_scale_and_center(
    keypoints: List[List[float]], conf_floor: float
) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """Return a robust body scale (px) and torso center for a person.

    Scale is the largest of shoulder width and torso height among the
    confident keypoints; center is the mean of the present torso keypoints.
    Returns (None, None) when there is too little to estimate.
    """
    present = _present(keypoints, conf_floor)
    scales: List[float] = []
    if 5 in present and 6 in present:
        scales.append(
            hypot(present[5][0] - present[6][0], present[5][1] - present[6][1])
        )
    if 11 in present and 12 in present:
        scales.append(
            hypot(present[11][0] - present[12][0], present[11][1] - present[12][1])
        )

    def _mid(a: int, b: int) -> Optional[Tuple[float, float]]:
        if a in present and b in present:
            return (
                (present[a][0] + present[b][0]) / 2,
                (present[a][1] + present[b][1]) / 2,
            )
        return None

    shoulder_mid = _mid(5, 6)
    hip_mid = _mid(11, 12)
    if shoulder_mid and hip_mid:
        scales.append(hypot(shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1]))

    torso_pts = [present[k] for k in TORSO_KEYPOINTS if k in present]
    center = None
    if torso_pts:
        center = (
            sum(p[0] for p in torso_pts) / len(torso_pts),
            sum(p[1] for p in torso_pts) / len(torso_pts),
        )
    elif present:
        pts = list(present.values())
        center = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    scale = max(scales) if scales else None
    return scale, center


def _robust_scale(
    person: Dict[str, Any], cfg: CleanConfig
) -> Tuple[float, Optional[Tuple[float, float]]]:
    """Body scale floored at half the box diagonal, plus the body center.

    Flooring at the box prevents the scale from collapsing when only a couple
    of torso keypoints are confident, which would otherwise make the off-body
    check demote legitimate far-reaching limbs.
    """
    measured, center = body_scale_and_center(person["keypoints"], cfg.conf_floor)
    box = person["bbox_xyxy"]
    box_diag = hypot(box[2] - box[0], box[3] - box[1])
    return max(measured or 0.0, 0.5 * box_diag), center


def demote_offbody_keypoints(
    person: Dict[str, Any], cfg: CleanConfig
) -> Dict[str, Any]:
    """Demote HEAD keypoints implausibly far from the body center.

    Only head keypoints are distance-checked: this targets the face-steal case
    where one person's box grabbed another's face. Limbs legitimately reach far
    and the scale estimate is too noisy to gate them, so they are left alone.
    """
    person = copy.deepcopy(person)
    scale, center = _robust_scale(person, cfg)
    if center is None or scale <= 0:
        return person
    for k in HEAD_KEYPOINTS:
        kp = person["keypoints"][k]
        if kp[2] < cfg.conf_floor:
            continue
        if hypot(kp[0] - center[0], kp[1] - center[1]) / scale > cfg.head_max_ratio:
            person["keypoints"][k] = [0.0, 0.0, 0.0]
    return person


def resolve_cross_instance_conflicts(
    persons: List[Dict[str, Any]], cfg: CleanConfig
) -> List[Dict[str, Any]]:
    """Demote keypoints that two people share, keeping the nearer owner's."""
    persons = [copy.deepcopy(p) for p in persons]
    meta = [_robust_scale(p, cfg) for p in persons]
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            scale_i, center_i = meta[i]
            scale_j, center_j = meta[j]
            if not scale_i or not scale_j or center_i is None or center_j is None:
                continue
            tol = cfg.overlap_frac * max(scale_i, scale_j)
            # Only de-conflict HEAD keypoints. Two distinct people in close
            # contact can have legitimately-near limb keypoints, so cross-
            # checking limbs would demote real joints; faces never coincide.
            for k in sorted(HEAD_KEYPOINTS):
                ki = persons[i]["keypoints"][k]
                kj = persons[j]["keypoints"][k]
                if ki[2] < cfg.conf_floor or kj[2] < cfg.conf_floor:
                    continue
                if hypot(ki[0] - kj[0], ki[1] - kj[1]) > tol:
                    continue
                point = ((ki[0] + kj[0]) / 2, (ki[1] + kj[1]) / 2)
                di = hypot(point[0] - center_i[0], point[1] - center_i[1])
                dj = hypot(point[0] - center_j[0], point[1] - center_j[1])
                loser = j if di <= dj else i
                persons[loser]["keypoints"][k] = [0.0, 0.0, 0.0]
    return persons


def demote_offimage_keypoints(
    person: Dict[str, Any], image_w: int, image_h: int, cfg: CleanConfig
) -> Dict[str, Any]:
    """Demote keypoints that fall well outside the image frame."""
    person = copy.deepcopy(person)
    mx = cfg.offimage_margin_frac * image_w
    my = cfg.offimage_margin_frac * image_h
    for k, kp in enumerate(person["keypoints"]):
        if kp[2] < cfg.conf_floor:
            continue
        if not (-mx <= kp[0] <= image_w + mx and -my <= kp[1] <= image_h + my):
            person["keypoints"][k] = [0.0, 0.0, 0.0]
    return person


def _segment_length(
    keypoints: List[List[float]], a: int, b: int, conf_floor: float
) -> Optional[float]:
    ka, kb = keypoints[a], keypoints[b]
    if ka[2] < conf_floor or kb[2] < conf_floor:
        return None
    return hypot(ka[0] - kb[0], ka[1] - kb[1])


def demote_asymmetric_limbs(person: Dict[str, Any], cfg: CleanConfig) -> Dict[str, Any]:
    """Demote a distal joint when its limb segment is implausibly longer than
    the mirror segment on the other side of the body."""
    person = copy.deepcopy(person)
    kpts = person["keypoints"]
    for (lp, ld), (rp, rd) in SYMMETRIC_SEGMENTS:
        left_len = _segment_length(kpts, lp, ld, cfg.conf_floor)
        right_len = _segment_length(kpts, rp, rd, cfg.conf_floor)
        if left_len is None or right_len is None:
            continue
        small = min(left_len, right_len)
        large = max(left_len, right_len)
        if small <= 1e-6 or large / small <= cfg.max_asym_ratio:
            continue
        loser_distal = ld if left_len >= right_len else rd
        person["keypoints"][loser_distal] = [0.0, 0.0, 0.0]
    return person


def _pose_similarity(a: Dict[str, Any], b: Dict[str, Any], cfg: CleanConfig) -> float:
    """Fraction of mutually-confident keypoints that nearly coincide."""
    scale_a, _ = body_scale_and_center(a["keypoints"], cfg.conf_floor)
    scale_b, _ = body_scale_and_center(b["keypoints"], cfg.conf_floor)
    scale = max(scale_a or 0.0, scale_b or 0.0)
    if scale <= 0:
        return 0.0
    tol = cfg.nms_overlap_frac * scale
    shared = 0
    close = 0
    for ka, kb in zip(a["keypoints"], b["keypoints"]):
        if ka[2] < cfg.conf_floor or kb[2] < cfg.conf_floor:
            continue
        shared += 1
        if hypot(ka[0] - kb[0], ka[1] - kb[1]) <= tol:
            close += 1
    return close / shared if shared else 0.0


def _box_iou(a: List[float], b: List[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _confident_keypoint_count(person: Dict[str, Any], conf_floor: float) -> int:
    return sum(1 for kp in person["keypoints"] if kp[2] >= conf_floor)


def pose_nms(persons: List[Dict[str, Any]], cfg: CleanConfig) -> List[Dict[str, Any]]:
    """Drop duplicate instances, keeping the one with the fuller, surer pose.

    A duplicate must share most keypoints AND have a near-identical box, so two
    distinct overlapping people are never merged. Among duplicates the instance
    with more confident keypoints wins (box confidence breaks ties), so a fuller
    skeleton is never dropped in favour of a sparser high-score box.
    """
    order = sorted(
        range(len(persons)),
        key=lambda i: (
            _confident_keypoint_count(persons[i], cfg.conf_floor),
            persons[i]["box_conf"],
        ),
        reverse=True,
    )
    keep: List[int] = []
    for index in order:
        duplicate = any(
            _box_iou(persons[index]["bbox_xyxy"], persons[kept]["bbox_xyxy"])
            >= cfg.nms_box_iou
            and _pose_similarity(persons[index], persons[kept], cfg)
            >= cfg.nms_sim_thresh
            for kept in keep
        )
        if not duplicate:
            keep.append(index)
    return [persons[i] for i in sorted(keep)]


def tighten_box(
    person: Dict[str, Any], cfg: CleanConfig, image_w: int, image_h: int
) -> Dict[str, Any]:
    """Refit the person box to its confident keypoints, with a small margin."""
    person = copy.deepcopy(person)
    present = [(kp[0], kp[1]) for kp in person["keypoints"] if kp[2] >= cfg.conf_floor]
    if len(present) < 2:
        return person
    xs = [p[0] for p in present]
    ys = [p[1] for p in present]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    mx = cfg.tighten_margin * max(x2 - x1, cfg.min_box_px)
    my = cfg.tighten_margin * max(y2 - y1, cfg.min_box_px)
    person["bbox_xyxy"] = [
        max(0.0, x1 - mx),
        max(0.0, y1 - my),
        min(float(image_w), x2 + mx),
        min(float(image_h), y2 + my),
    ]
    return person


def clean_frame(
    persons: List[Dict[str, Any]], image_w: int, image_h: int, cfg: CleanConfig
) -> List[Dict[str, Any]]:
    """Run the full cleaning pipeline on one frame's persons."""
    persons = pose_nms(persons, cfg)
    persons = resolve_cross_instance_conflicts(persons, cfg)
    cleaned: List[Dict[str, Any]] = []
    for person in persons:
        person = demote_offimage_keypoints(person, image_w, image_h, cfg)
        person = demote_offbody_keypoints(person, cfg)
        person = tighten_box(person, cfg, image_w, image_h)
        cleaned.append(person)
    return cleaned
