"""Teacher pseudo-labeling with yolo11x-pose.

Runs the largest COCO pose model over the sampled frames, optionally with
horizontal-flip / multi-scale test-time augmentation and a higher inference
resolution than the portal uses, to produce the best automatic labels the
teacher can give. Up to ``max_persons`` instances are kept per frame (parent
and child). Output is a single JSON file consumed by the Phase 3 QC and
dataset-export stage. No manual correction happens here.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from video_inference.weights import resolve_weights

from .boxes import clean_detection_boxes
from .coco import FLIP_IDX

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "pose_training" / "data" / "frame_manifest.csv"
DEFAULT_OUTPUT = REPO_ROOT / "pose_training" / "data" / "pseudo_labels_raw.json"
DEFAULT_TEACHER = "yolo11x-pose.pt"


@dataclass
class PseudoLabelConfig:
    """Configuration for the teacher pseudo-labeling stage."""

    manifest_path: Path = DEFAULT_MANIFEST
    output_path: Path = DEFAULT_OUTPUT
    teacher_weights: str = DEFAULT_TEACHER
    device: str = "0"
    imgsz: int = 960
    conf: float = 0.25
    iou: float = 0.7
    tta: bool = True
    tta_match_iou: float = 0.5
    max_persons: int = 2
    batch_size: int = 16


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor or array-like to a numpy array on CPU."""
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def extract_persons(
    boxes_xyxy: np.ndarray,
    boxes_conf: np.ndarray,
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    max_persons: int,
) -> List[Dict[str, Any]]:
    """Build up to ``max_persons`` person records from one frame's detections.

    Args:
        boxes_xyxy: (n, 4) person boxes in pixels.
        boxes_conf: (n,) box confidences.
        kpts_xy: (n, 17, 2) keypoint pixel coordinates.
        kpts_conf: (n, 17) per-keypoint confidences.
        max_persons: Maximum number of instances to keep, by box confidence.

    Returns:
        List of person dicts, highest confidence first. Each has ``bbox_xyxy``
        (list of 4), ``box_conf`` (float), and ``keypoints`` (17 x [x, y, conf]).
    """
    if boxes_xyxy.shape[0] == 0:
        return []
    # Remove overlapping/contained duplicate boxes before keeping the top-K, so
    # one person is not exported as several stacked detections.
    order = clean_detection_boxes(boxes_xyxy, boxes_conf)[:max_persons]
    persons: List[Dict[str, Any]] = []
    for index in order:
        keypoints = [
            [float(x), float(y), float(conf)]
            for (x, y), conf in zip(kpts_xy[index], kpts_conf[index])
        ]
        persons.append(
            {
                "bbox_xyxy": [float(v) for v in boxes_xyxy[index]],
                "box_conf": float(boxes_conf[index]),
                "keypoints": keypoints,
            }
        )
    return persons


def flip_persons_back(
    persons: List[Dict[str, Any]], width: int
) -> List[Dict[str, Any]]:
    """Map persons detected on a horizontally-flipped image back to originals.

    A horizontal mirror sends pixel x to ``width - 1 - x`` and swaps every
    left/right keypoint (COCO ``FLIP_IDX``). The box corners swap sides too.

    Args:
        persons: Person records as returned by :func:`extract_persons`, but
            measured on the mirrored image.
        width: Original image width in pixels.

    Returns:
        New person records in original-image coordinates.
    """
    mirrored: List[Dict[str, Any]] = []
    for person in persons:
        x1, y1, x2, y2 = person["bbox_xyxy"]
        new_box = [width - 1 - x2, y1, width - 1 - x1, y2]
        keypoints = person["keypoints"]
        reordered = [keypoints[src] for src in FLIP_IDX]
        new_kpts = [[width - 1 - x, y, conf] for x, y, conf in reordered]
        mirrored.append(
            {
                "bbox_xyxy": [float(v) for v in new_box],
                "box_conf": float(person["box_conf"]),
                "keypoints": new_kpts,
            }
        )
    return mirrored


def _bbox_iou(a: List[float], b: List[float]) -> float:
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


def _average_persons(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Confidence-weight the keypoints of two matched person records."""
    keypoints = []
    for (ax, ay, ac), (bx, by, bc) in zip(a["keypoints"], b["keypoints"]):
        total = ac + bc
        if total <= 0.0:
            keypoints.append([(ax + bx) / 2.0, (ay + by) / 2.0, 0.0])
        else:
            keypoints.append(
                [
                    (ax * ac + bx * bc) / total,
                    (ay * ac + by * bc) / total,
                    (ac + bc) / 2.0,
                ]
            )
    box = [(a["bbox_xyxy"][i] + b["bbox_xyxy"][i]) / 2.0 for i in range(4)]
    return {
        "bbox_xyxy": box,
        "box_conf": (a["box_conf"] + b["box_conf"]) / 2.0,
        "keypoints": keypoints,
    }


def merge_persons(
    persons_a: List[Dict[str, Any]],
    persons_b: List[Dict[str, Any]],
    max_persons: int,
    match_iou: float = 0.5,
) -> List[Dict[str, Any]]:
    """Greedily match two prediction sets by IoU and average the matches.

    Matched instances are confidence-averaged; unmatched instances from either
    set are kept as-is. The result is truncated to ``max_persons`` by box
    confidence.
    """
    used_b: set = set()
    merged: List[Dict[str, Any]] = []
    for person_a in persons_a:
        best_index, best_iou = -1, match_iou
        for index, person_b in enumerate(persons_b):
            if index in used_b:
                continue
            iou = _bbox_iou(person_a["bbox_xyxy"], person_b["bbox_xyxy"])
            if iou >= best_iou:
                best_index, best_iou = index, iou
        if best_index >= 0:
            used_b.add(best_index)
            merged.append(_average_persons(person_a, persons_b[best_index]))
        else:
            merged.append(person_a)
    for index, person_b in enumerate(persons_b):
        if index not in used_b:
            merged.append(person_b)
    merged.sort(key=lambda person: person["box_conf"], reverse=True)
    return merged[:max_persons]


def _result_to_arrays(result: Any):
    """Pull boxes/keypoints arrays out of an Ultralytics result-like object."""
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None:
        n = 0
        return (
            np.zeros((n, 4)),
            np.zeros((n,)),
            np.zeros((n, 17, 2)),
            np.zeros((n, 17)),
        )
    boxes_xyxy = _to_numpy(boxes.xyxy)
    boxes_conf = _to_numpy(boxes.conf)
    kpts_xy = _to_numpy(keypoints.xy)
    kpts_conf = (
        _to_numpy(keypoints.conf)
        if getattr(keypoints, "conf", None) is not None
        else np.ones(kpts_xy.shape[:2])
    )
    return boxes_xyxy, boxes_conf, kpts_xy, kpts_conf


def _load_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_pseudo_labeling(
    config: PseudoLabelConfig, model: Optional[Any] = None
) -> Dict[str, Any]:
    """Run the teacher over every frame in the manifest and write JSON.

    Args:
        config: Pseudo-labeling configuration.
        model: Optional pre-built model (callable on a list of image paths).
            When None, a YOLO model is loaded from ``config.teacher_weights``.

    Returns:
        The payload dict that was written to ``config.output_path``.
    """
    rows = _load_manifest(config.manifest_path)
    if model is None:
        from ultralytics import YOLO

        model = YOLO(resolve_weights(config.teacher_weights))

    predict_kwargs = dict(
        imgsz=config.imgsz,
        conf=config.conf,
        iou=config.iou,
        device=config.device,
        verbose=False,
    )
    frame_records: List[Dict[str, Any]] = []
    batch = max(1, config.batch_size)
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        paths = [row["frame_path"] for row in chunk]
        results = model(paths, **predict_kwargs)

        flip_results = None
        if config.tta:
            flipped_images = []
            for path in paths:
                image = cv2.imread(path)
                flipped_images.append(cv2.flip(image, 1) if image is not None else None)
            if all(image is not None for image in flipped_images):
                flip_results = model(flipped_images, **predict_kwargs)

        for offset, (row, result) in enumerate(zip(chunk, results)):
            height, width = getattr(result, "orig_shape", (None, None))
            boxes_xyxy, boxes_conf, kpts_xy, kpts_conf = _result_to_arrays(result)
            persons = extract_persons(
                boxes_xyxy, boxes_conf, kpts_xy, kpts_conf, config.max_persons
            )
            if flip_results is not None and width is not None:
                f_boxes, f_conf, f_kxy, f_kconf = _result_to_arrays(
                    flip_results[offset]
                )
                flipped = flip_persons_back(
                    extract_persons(
                        f_boxes, f_conf, f_kxy, f_kconf, config.max_persons
                    ),
                    width,
                )
                persons = merge_persons(
                    persons, flipped, config.max_persons, config.tta_match_iou
                )
            frame_records.append(
                {
                    "frame_path": row["frame_path"],
                    "participant": row["participant"],
                    "angle": row["angle"],
                    "split": row["split"],
                    "source_video": row.get("source_video", ""),
                    "source_timestamp_s": float(row.get("source_timestamp_s", 0.0)),
                    "width": int(width) if width is not None else None,
                    "height": int(height) if height is not None else None,
                    "persons": persons,
                }
            )

    payload = {
        "config": {
            **{
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in asdict(config).items()
            }
        },
        "teacher": config.teacher_weights,
        "frames": frame_records,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def summarize_pseudo_labels(payload: Dict[str, Any]) -> str:
    """Return a short summary: frame and instance counts, empty frames."""
    frames = payload["frames"]
    instances = sum(len(frame["persons"]) for frame in frames)
    empty = sum(1 for frame in frames if not frame["persons"])
    two_person = sum(1 for frame in frames if len(frame["persons"]) >= 2)
    return (
        f"Frames: {len(frames)}; instances: {instances}; "
        f"empty frames: {empty}; frames with >=2 persons: {two_person}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pseudo-label sampled frames with yolo11x-pose."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), type=str)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), type=str)
    parser.add_argument("--teacher-weights", default=DEFAULT_TEACHER, type=str)
    parser.add_argument("--device", default="0", type=str)
    parser.add_argument("--imgsz", default=960, type=int)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--iou", default=0.7, type=float)
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Horizontal-flip test-time augmentation (manual; the model does "
        "not support augment=True for pose).",
    )
    parser.add_argument("--max-persons", default=2, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = PseudoLabelConfig(
        manifest_path=Path(args.manifest),
        output_path=Path(args.output),
        teacher_weights=args.teacher_weights,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        tta=args.tta,
        max_persons=args.max_persons,
        batch_size=args.batch_size,
    )
    payload = run_pseudo_labeling(config)
    print(f"Wrote pseudo-labels: {config.output_path}")
    print(summarize_pseudo_labels(payload))


if __name__ == "__main__":
    main()
