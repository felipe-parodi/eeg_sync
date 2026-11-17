"""Fuse yolo11x-pose and ViTPose pseudo-labels into one stronger label set.

The two teachers are different architectures (bottom-up YOLO vs top-down
ViTPose) with partially independent errors, so combining them per keypoint can
produce labels better than either alone. ViTPose is treated as the primary
keypoint source (it is the stronger pose model) and x-pose as an independent
verifier: where they agree, confidence is boosted and positions averaged; where
they disagree, the ViTPose position is kept but its confidence is penalized so
the downstream geometry / >=N-keypoint filters can drop the uncertain joints.

Boxes are unioned (cap 4 persons/frame) and de-duplicated by IoU. Output uses
the same ``pseudo_labels_raw.json`` schema, so ``dataset.py`` consumes it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Any, Dict, List, Optional

from .pose_sanity import CleanConfig, clean_frame

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XPOSE = REPO_ROOT / "pose_training" / "data" / "pseudo_labels_raw.json"
DEFAULT_VITPOSE = REPO_ROOT / "pose_training" / "data" / "pseudo_labels_vitpose.json"
DEFAULT_OUTPUT = REPO_ROOT / "pose_training" / "data" / "pseudo_labels_fused.json"


@dataclass
class FuseConfig:
    """Configuration for label fusion."""

    match_iou: float = 0.5
    max_persons: int = 4
    conf_floor: float = 0.3
    agree_frac: float = 0.05
    agree_boost: float = 1.1
    disagree_penalty: float = 0.6


def box_iou(a: List[float], b: List[float]) -> float:
    """IoU between two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _box_diag(box: List[float]) -> float:
    return hypot(box[2] - box[0], box[3] - box[1]) + 1e-6


def fuse_keypoint(
    vit: List[float], xp: List[float], box_diag: float, cfg: FuseConfig
) -> List[float]:
    """Fuse one keypoint from ViTPose (primary) and x-pose (verifier).

    Args:
        vit: ViTPose keypoint [x, y, score].
        xp: x-pose keypoint [x, y, conf].
        box_diag: Person box diagonal in pixels (for scale-relative agreement).
        cfg: Fusion thresholds.

    Returns:
        Fused keypoint [x, y, conf].
    """
    vx, vy, vc = vit
    px, py, pc = xp
    v_ok = vc >= cfg.conf_floor
    p_ok = pc >= cfg.conf_floor
    if v_ok and p_ok:
        dist = hypot(vx - px, vy - py) / box_diag
        if dist <= cfg.agree_frac:
            weight = vc + pc
            x = (vx * vc + px * pc) / weight
            y = (vy * vc + py * pc) / weight
            return [x, y, min(1.0, max(vc, pc) * cfg.agree_boost)]
        return [vx, vy, vc * cfg.disagree_penalty]
    if v_ok:
        return [vx, vy, vc]
    if p_ok:
        return [px, py, pc]
    return [0.0, 0.0, 0.0]


def fuse_keypoints(
    vit_kpts: Optional[List[List[float]]],
    xp_kpts: Optional[List[List[float]]],
    box_diag: float,
    cfg: FuseConfig,
) -> List[List[float]]:
    """Fuse all 17 keypoints; if one source is missing, return the other."""
    if vit_kpts is None:
        return xp_kpts or []
    if xp_kpts is None:
        return vit_kpts
    return [fuse_keypoint(v, p, box_diag, cfg) for v, p in zip(vit_kpts, xp_kpts)]


def _merge_box(a: List[float], b: List[float]) -> List[float]:
    return [(a[i] + b[i]) / 2.0 for i in range(4)]


def fuse_frame(
    xpose_persons: List[Dict[str, Any]],
    vitpose_persons: List[Dict[str, Any]],
    cfg: FuseConfig,
) -> List[Dict[str, Any]]:
    """Match persons across the two teachers and fuse them.

    Returns up to ``cfg.max_persons`` fused person records, highest box
    confidence first.
    """
    used_xp: set = set()
    fused: List[Dict[str, Any]] = []
    for vp in vitpose_persons:
        best_index, best_iou = -1, cfg.match_iou
        for index, xp in enumerate(xpose_persons):
            if index in used_xp:
                continue
            iou = box_iou(vp["bbox_xyxy"], xp["bbox_xyxy"])
            if iou >= best_iou:
                best_index, best_iou = index, iou
        if best_index >= 0:
            used_xp.add(best_index)
            xp = xpose_persons[best_index]
            box = _merge_box(vp["bbox_xyxy"], xp["bbox_xyxy"])
            fused.append(
                {
                    "bbox_xyxy": box,
                    "box_conf": max(vp["box_conf"], xp["box_conf"]),
                    "keypoints": fuse_keypoints(
                        vp["keypoints"], xp["keypoints"], _box_diag(box), cfg
                    ),
                }
            )
        else:
            fused.append(vp)
    for index, xp in enumerate(xpose_persons):
        if index not in used_xp:
            fused.append(xp)
    fused.sort(key=lambda person: person["box_conf"], reverse=True)
    return fused[: cfg.max_persons]


def fuse_label_files(
    xpose_payload: Dict[str, Any],
    vitpose_payload: Dict[str, Any],
    cfg: FuseConfig,
    clean_cfg: Optional[CleanConfig] = None,
    apply_clean: bool = True,
) -> Dict[str, Any]:
    """Fuse two full pseudo-label payloads keyed by frame path.

    When ``apply_clean`` is set, each fused frame is run through the geometric
    sanity cleaner (cross-instance de-duplication, off-body keypoint demotion,
    box tightening) so stolen/overlapping keypoints are removed.
    """
    vit_by_path = {f["frame_path"]: f for f in vitpose_payload["frames"]}
    clean = clean_cfg if clean_cfg is not None else CleanConfig()
    frames: List[Dict[str, Any]] = []
    for xframe in xpose_payload["frames"]:
        vframe = vit_by_path.get(xframe["frame_path"])
        vit_persons = vframe["persons"] if vframe else []
        fused_persons = fuse_frame(xframe["persons"], vit_persons, cfg)
        if apply_clean and xframe.get("width") and xframe.get("height"):
            fused_persons = clean_frame(
                fused_persons, int(xframe["width"]), int(xframe["height"]), clean
            )
        frames.append({**xframe, "persons": fused_persons})
    return {
        "teacher": "fused:yolo11x-pose+vitpose-huge",
        "fuse_config": vars(cfg),
        "cleaned": apply_clean,
        "frames": frames,
    }


def summarize(payload: Dict[str, Any]) -> str:
    """Short summary of the fused label set."""
    frames = payload["frames"]
    instances = sum(len(f["persons"]) for f in frames)
    empty = sum(1 for f in frames if not f["persons"])
    counts: Dict[int, int] = {}
    for f in frames:
        counts[len(f["persons"])] = counts.get(len(f["persons"]), 0) + 1
    return (
        f"Frames: {len(frames)}; instances: {instances}; empty: {empty}; "
        f"persons-per-frame: {dict(sorted(counts.items()))}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse yolo11x-pose and ViTPose pseudo-labels."
    )
    parser.add_argument("--xpose", default=str(DEFAULT_XPOSE), type=str)
    parser.add_argument("--vitpose", default=str(DEFAULT_VITPOSE), type=str)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), type=str)
    parser.add_argument("--max-persons", default=4, type=int)
    parser.add_argument("--agree-frac", default=0.05, type=float)
    parser.add_argument("--disagree-penalty", default=0.6, type=float)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = FuseConfig(
        max_persons=args.max_persons,
        agree_frac=args.agree_frac,
        disagree_penalty=args.disagree_penalty,
    )
    xpose_payload = json.loads(Path(args.xpose).read_text(encoding="utf-8"))
    vitpose_payload = json.loads(Path(args.vitpose).read_text(encoding="utf-8"))
    fused = fuse_label_files(xpose_payload, vitpose_payload, cfg)
    Path(args.output).write_text(json.dumps(fused, indent=2), encoding="utf-8")
    print(f"Wrote fused labels: {args.output}")
    print(summarize(fused))


if __name__ == "__main__":
    main()
