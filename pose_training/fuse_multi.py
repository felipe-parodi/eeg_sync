"""3-way teacher fusion: box-to-box matching + per-keypoint majority vote.

Each physical person is matched across the three teachers (yolo11x-pose,
ViTPose, RTMPose) by box IoU, then every keypoint is decided by a vote: where
>=2 independent teachers agree (within a fraction of body size) the consensus
is kept with boosted confidence; a lone or disagreeing keypoint is kept only at
reduced confidence so the downstream filters can drop it. Because the vote does
the denoising, the fragile geometry cleaning becomes a light final pass.

Output uses the shared ``pseudo_labels_raw.json`` schema, so ``dataset.py``
consumes it unchanged.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .pose_sanity import CleanConfig, clean_frame

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "pose_training" / "data"
DEFAULT_XPOSE = DATA / "pseudo_labels_raw.json"
DEFAULT_VITPOSE = DATA / "pseudo_labels_vitpose.json"
DEFAULT_RTMPOSE = DATA / "pseudo_labels_rtmpose.json"
DEFAULT_SAPIENS = DATA / "pseudo_labels_sapiens.json"
DEFAULT_OUTPUT = DATA / "pseudo_labels_fused4.json"


@dataclass
class FuseMultiConfig:
    """Configuration for N-way teacher fusion."""

    match_iou: float = 0.4
    max_persons: int = 4
    conf_floor: float = 0.3
    agree_frac: float = 0.07
    agree_boost: float = 1.1
    single_vote_conf: float = 0.6
    disagree_conf: float = 0.5
    clean: bool = True


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
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


def cluster_persons(
    persons_by_teacher: List[List[Dict[str, Any]]], match_iou: float
) -> List[List[Optional[Dict[str, Any]]]]:
    """Group persons that refer to the same physical body across teachers.

    Returns a list of clusters; each cluster is a list with one slot per
    teacher holding that teacher's person (or None if it did not detect them).
    Greedy by box confidence; at most one person per teacher per cluster.
    """
    n = len(persons_by_teacher)
    items = [
        (t, person) for t, plist in enumerate(persons_by_teacher) for person in plist
    ]
    items.sort(key=lambda it: it[1]["box_conf"], reverse=True)
    clusters: List[Dict[str, Any]] = []
    for teacher, person in items:
        placed = False
        for cluster in clusters:
            if teacher in cluster["members"]:
                continue
            if box_iou(person["bbox_xyxy"], cluster["rep_box"]) >= match_iou:
                cluster["members"][teacher] = person
                placed = True
                break
        if not placed:
            clusters.append(
                {"rep_box": person["bbox_xyxy"], "members": {teacher: person}}
            )
    return [[cluster["members"].get(t) for t in range(n)] for cluster in clusters]


def vote_keypoint(
    candidates: List[List[float]], agree_dist: float, cfg: FuseMultiConfig
) -> List[float]:
    """Decide one keypoint by majority vote across teacher candidates.

    Args:
        candidates: List of [x, y, conf] from teachers that are confident here.
        agree_dist: Distance (px) within which two candidates "agree".
        cfg: Fusion config.

    Returns:
        Fused [x, y, conf]. Consensus of an agreeing majority is boosted; a lone
        or fully-disagreeing keypoint is kept at reduced confidence.
    """
    if not candidates:
        return [0.0, 0.0, 0.0]
    if len(candidates) == 1:
        x, y, conf = candidates[0]
        return [x, y, conf * cfg.single_vote_conf]
    best: List[List[float]] = []
    for anchor in candidates:
        agreeing = [
            c
            for c in candidates
            if hypot(c[0] - anchor[0], c[1] - anchor[1]) <= agree_dist
        ]
        if len(agreeing) > len(best):
            best = agreeing
    if len(best) >= 2:
        weight = sum(c[2] for c in best)
        x = sum(c[0] * c[2] for c in best) / weight
        y = sum(c[1] * c[2] for c in best) / weight
        conf = min(1.0, (weight / len(best)) * cfg.agree_boost)
        return [x, y, conf]
    strongest = max(candidates, key=lambda c: c[2])
    return [strongest[0], strongest[1], strongest[2] * cfg.disagree_conf]


def fuse_cluster(
    cluster: List[Optional[Dict[str, Any]]], cfg: FuseMultiConfig
) -> Dict[str, Any]:
    """Fuse one matched cluster into a single person via per-keypoint voting."""
    present = [person for person in cluster if person is not None]
    box = [
        sum(person["bbox_xyxy"][i] for person in present) / len(present)
        for i in range(4)
    ]
    box_diag = hypot(box[2] - box[0], box[3] - box[1]) + 1e-6
    agree_dist = cfg.agree_frac * box_diag
    keypoints: List[List[float]] = []
    for k in range(17):
        candidates = [
            person["keypoints"][k]
            for person in present
            if person["keypoints"][k][2] >= cfg.conf_floor
        ]
        keypoints.append(vote_keypoint(candidates, agree_dist, cfg))
    return {
        "bbox_xyxy": box,
        "box_conf": sum(person["box_conf"] for person in present) / len(present),
        "keypoints": keypoints,
        "n_teachers": len(present),
    }


def fuse_frame(
    persons_by_teacher: List[List[Dict[str, Any]]],
    width: int,
    height: int,
    cfg: FuseMultiConfig,
) -> List[Dict[str, Any]]:
    """Cluster, vote-fuse, cap, and lightly clean one frame's persons."""
    clusters = cluster_persons(persons_by_teacher, cfg.match_iou)
    fused = [fuse_cluster(cluster, cfg) for cluster in clusters]
    # Prefer people more teachers agreed on, then higher confidence.
    fused.sort(key=lambda p: (p["n_teachers"], p["box_conf"]), reverse=True)
    fused = fused[: cfg.max_persons]
    if cfg.clean:
        fused = clean_frame(fused, width, height, CleanConfig())
    return fused


def fuse_label_files(
    payloads: List[Dict[str, Any]], cfg: FuseMultiConfig
) -> Dict[str, Any]:
    """Fuse the per-teacher payloads (keyed by frame path) into one set."""
    by_path = [{f["frame_path"]: f for f in payload["frames"]} for payload in payloads]
    base = payloads[0]["frames"]
    frames: List[Dict[str, Any]] = []
    for frame in base:
        path = frame["frame_path"]
        persons_by_teacher = [
            indexed.get(path, {"persons": []})["persons"] for indexed in by_path
        ]
        fused = fuse_frame(
            persons_by_teacher, int(frame["width"]), int(frame["height"]), cfg
        )
        frames.append({**frame, "persons": fused})
    return {
        "teacher": f"fused{len(payloads)}",
        "fuse_config": vars(cfg),
        "frames": frames,
    }


def summarize(payload: Dict[str, Any]) -> str:
    frames = payload["frames"]
    instances = sum(len(f["persons"]) for f in frames)
    by_votes: Dict[int, int] = {}
    for f in frames:
        for p in f["persons"]:
            by_votes[p.get("n_teachers", 0)] = (
                by_votes.get(p.get("n_teachers", 0), 0) + 1
            )
    return (
        f"Frames: {len(frames)}; instances: {instances}; "
        f"instances by #teachers-agreeing: {dict(sorted(by_votes.items()))}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="N-way teacher fusion (any teacher file that exists is used)."
    )
    parser.add_argument("--xpose", default=str(DEFAULT_XPOSE), type=str)
    parser.add_argument("--vitpose", default=str(DEFAULT_VITPOSE), type=str)
    parser.add_argument("--rtmpose", default=str(DEFAULT_RTMPOSE), type=str)
    parser.add_argument("--sapiens", default=str(DEFAULT_SAPIENS), type=str)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), type=str)
    parser.add_argument("--match-iou", default=0.4, type=float)
    parser.add_argument("--max-persons", default=4, type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = FuseMultiConfig(match_iou=args.match_iou, max_persons=args.max_persons)
    teacher_paths = [
        p
        for p in (args.xpose, args.vitpose, args.rtmpose, args.sapiens)
        if Path(p).exists()
    ]
    payloads = [json.loads(Path(p).read_text(encoding="utf-8")) for p in teacher_paths]
    fused = fuse_label_files(payloads, cfg)
    Path(args.output).write_text(json.dumps(fused, indent=2), encoding="utf-8")
    print(f"Wrote fused labels ({len(payloads)} teachers): {args.output}")
    print(summarize(fused))


if __name__ == "__main__":
    main()
