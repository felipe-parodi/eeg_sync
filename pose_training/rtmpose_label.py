"""RTMPose (rtmlib) teacher: a third, architecturally-independent COCO-17 source.

rtmlib's ``Body`` is an RTMDet detector + RTMPose head. Because it shares no
architecture or training pipeline with yolo11x-pose or ViTPose, its errors are
independent, which is exactly what makes a 3-way fused label set stronger than
any single teacher. It also supplies its own boxes (derived from keypoints),
adding genuine box diversity to the fusion.

Output uses the SAME ``pseudo_labels_raw.json`` schema as the other teachers.
See docs/vitpose-teacher_260531.md (fusion plan).
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

from .boxes import clean_detection_boxes

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "pose_training" / "data" / "frame_manifest.csv"
DEFAULT_OUTPUT = REPO_ROOT / "pose_training" / "data" / "pseudo_labels_rtmpose.json"


@dataclass
class RtmposeConfig:
    """Configuration for the RTMPose (rtmlib) labeling stage."""

    manifest_path: Path = DEFAULT_MANIFEST
    output_path: Path = DEFAULT_OUTPUT
    mode: str = "performance"  # rtmlib: performance | balanced | lightweight
    backend: str = "onnxruntime"
    device: str = "cuda"
    conf_floor: float = 0.3
    min_mean_score: float = 0.3
    expansion: float = 1.15
    max_persons: int = 4


def bbox_from_keypoints(
    keypoints_xy: np.ndarray,
    scores: np.ndarray,
    conf_floor: float,
    expansion: float,
) -> List[float]:
    """Derive an xyxy box from the confident keypoints, with padding."""
    keypoints_xy = np.asarray(keypoints_xy, dtype=float).reshape(-1, 2)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    mask = scores >= conf_floor
    pts = keypoints_xy[mask] if mask.sum() >= 2 else keypoints_xy
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    half_w = (x_max - x_min) / 2.0 * expansion
    half_h = (y_max - y_min) / 2.0 * expansion
    return [
        float(cx - half_w),
        float(cy - half_h),
        float(cx + half_w),
        float(cy + half_h),
    ]


def build_person_record(
    keypoints_xy: np.ndarray,
    scores: np.ndarray,
    conf_floor: float,
    expansion: float,
) -> Dict[str, Any]:
    """Build one person record from RTMPose keypoints + scores."""
    keypoints_xy = np.asarray(keypoints_xy, dtype=float).reshape(-1, 2)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    keypoints = [
        [float(x), float(y), float(s)] for (x, y), s in zip(keypoints_xy, scores)
    ]
    return {
        "bbox_xyxy": bbox_from_keypoints(keypoints_xy, scores, conf_floor, expansion),
        "box_conf": float(scores.mean()),
        "keypoints": keypoints,
    }


def top_persons_by_score(scores: np.ndarray, max_persons: int) -> List[int]:
    """Return indices of the top ``max_persons`` detections by mean score."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return []
    means = scores.reshape(scores.shape[0], -1).mean(axis=1)
    return [int(i) for i in np.argsort(means)[::-1][:max_persons]]


def _load_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_rtmpose_labeling(
    config: RtmposeConfig, model: Optional[Any] = None
) -> Dict[str, Any]:
    """Label every manifest frame with rtmlib Body (RTMDet + RTMPose).

    Args:
        config: Stage configuration.
        model: Optional callable returning ``(keypoints, scores)`` for a BGR
            frame. When None, an rtmlib ``Body`` model is constructed.

    Returns:
        The payload dict written to ``config.output_path``.
    """
    rows = _load_manifest(config.manifest_path)
    if model is None:
        # Import torch first so onnxruntime's CUDA EP can use torch's bundled
        # CUDA/cuDNN DLLs (the standalone onnxruntime-gpu has no DLLs of its own
        # in this env). Without this, the CUDA provider silently falls back to
        # CPU.
        import torch  # noqa: F401
        from rtmlib import Body

        model = Body(mode=config.mode, backend=config.backend, device=config.device)

    frame_records: List[Dict[str, Any]] = []
    for row in rows:
        image = cv2.imread(row["frame_path"])
        height, width = image.shape[:2]
        keypoints, scores = model(image)
        keypoints = np.asarray(keypoints, dtype=float)
        scores = np.asarray(scores, dtype=float)
        candidates = [
            build_person_record(
                keypoints[index], scores[index], config.conf_floor, config.expansion
            )
            for index in range(len(scores))
            if scores[index].size and scores[index].mean() >= config.min_mean_score
        ]
        persons: List[Dict[str, Any]] = []
        if candidates:
            kept = clean_detection_boxes(
                np.array([c["bbox_xyxy"] for c in candidates], dtype=float),
                np.array([c["box_conf"] for c in candidates], dtype=float),
            )[: config.max_persons]
            persons = [candidates[i] for i in kept]
        frame_records.append(
            {
                "frame_path": row["frame_path"],
                "participant": row["participant"],
                "angle": row["angle"],
                "split": row["split"],
                "source_video": row.get("source_video", ""),
                "source_timestamp_s": float(row.get("source_timestamp_s", 0.0)),
                "width": int(width),
                "height": int(height),
                "persons": persons,
            }
        )

    payload = {
        "config": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()
        },
        "teacher": f"rtmpose:{config.mode}",
        "frames": frame_records,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def summarize(payload: Dict[str, Any]) -> str:
    frames = payload["frames"]
    instances = sum(len(f["persons"]) for f in frames)
    empty = sum(1 for f in frames if not f["persons"])
    return f"Frames: {len(frames)}; instances: {instances}; empty: {empty}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label frames with rtmlib RTMPose.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), type=str)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), type=str)
    parser.add_argument("--mode", default="performance", type=str)
    parser.add_argument("--backend", default="onnxruntime", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--max-persons", default=4, type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = RtmposeConfig(
        manifest_path=Path(args.manifest),
        output_path=Path(args.output),
        mode=args.mode,
        backend=args.backend,
        device=args.device,
        max_persons=args.max_persons,
    )
    payload = run_rtmpose_labeling(config)
    print(f"Wrote RTMPose labels: {config.output_path}")
    print(summarize(payload))


if __name__ == "__main__":
    main()
