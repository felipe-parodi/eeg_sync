"""Ultralytics 2D pose inference runner with two-person ID stabilization."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .device import resolve_device
from .tracking import (
    AssignedDetection,
    TwoPersonTrackerState,
    assign_two_person_tracks,
)


@dataclass
class UltraRunnerConfig:
    """Configuration for Ultralytics frame inference."""

    model_path: str
    image_folder: str
    output_json: str
    device: str = "auto"
    conf: float = 0.25
    iou: float = 0.7
    max_images: Optional[int] = None
    tracker_backend: str = "internal"
    tracker_name: str = "bytetrack"


def _iter_image_paths(image_folder: Path, max_images: Optional[int]) -> List[Path]:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = sorted(
        [
            path
            for path in image_folder.iterdir()
            if path.is_file() and path.suffix.lower() in image_exts
        ]
    )
    if max_images is not None:
        images = images[:max_images]
    return images


def _resolve_ultralytics_device(device_preference: str) -> str:
    resolved = resolve_device(device_preference)
    return "0" if resolved == "cuda" else "cpu"


def _assign_from_role_indices(
    detections: List[Dict[str, Any]],
    parent_idx: int,
    child_idx: int,
) -> List[AssignedDetection]:
    parent_det = detections[parent_idx]
    child_det = detections[child_idx]
    return [
        AssignedDetection(
            track_id=0,
            track_label="parent",
            bbox=np.asarray(parent_det["bbox"], dtype=float),
            confidence=float(parent_det.get("confidence", 1.0)),
            detection_index=parent_idx,
        ),
        AssignedDetection(
            track_id=1,
            track_label="child",
            bbox=np.asarray(child_det["bbox"], dtype=float),
            confidence=float(child_det.get("confidence", 1.0)),
            detection_index=child_idx,
        ),
    ]


def _resolve_role_indices_from_tracker_ids(
    detections: List[Dict[str, Any]],
    tracker_ids: List[int],
    role_by_tracker_id: Dict[int, int],
) -> Tuple[int, int]:
    """
    Resolve (parent_idx, child_idx) from external tracker IDs.

    We keep a persistent map `tracker_id -> role_id` where role_id:
    - 0 => parent
    - 1 => child
    """
    if len(detections) != 2 or len(tracker_ids) != 2:
        raise ValueError("Expected exactly 2 detections and 2 tracker IDs.")

    known_roles = [role_by_tracker_id.get(track_id) for track_id in tracker_ids]

    if known_roles[0] is not None and known_roles[1] is not None:
        parent_idx = 0 if known_roles[0] == 0 else 1
    elif known_roles[0] is not None:
        parent_idx = 0 if known_roles[0] == 0 else 1
    elif known_roles[1] is not None:
        parent_idx = 1 if known_roles[1] == 0 else 0
    else:
        areas = [
            float((det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1]))
            for det in detections
        ]
        parent_idx = int(np.argmax(areas))

    child_idx = 1 - parent_idx
    role_by_tracker_id[tracker_ids[parent_idx]] = 0
    role_by_tracker_id[tracker_ids[child_idx]] = 1
    return parent_idx, child_idx


def run_ultralytics_pose_on_images(config: UltraRunnerConfig) -> Dict[str, Any]:
    """
    Run Ultralytics pose model on frames and export parent/child-assigned outputs.

    Notes:
    - We export 2D keypoints as pseudo-3D `[x, y, 0, conf]` to keep
      compatibility with the current downstream export contract.
    """
    image_folder = Path(config.image_folder)
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError(
            "Ultralytics is required for this backend. Install with: pip install ultralytics"
        ) from error

    if config.tracker_backend not in {"internal", "roboflow"}:
        raise ValueError(
            f"Invalid tracker_backend '{config.tracker_backend}'. "
            "Expected 'internal' or 'roboflow'."
        )

    ultra_device = _resolve_ultralytics_device(config.device)
    model = YOLO(config.model_path)
    images = _iter_image_paths(image_folder, config.max_images)

    state: Optional[TwoPersonTrackerState] = None
    frame_outputs: List[Dict[str, Any]] = []
    role_by_tracker_id: Dict[int, int] = {}

    rf_tracker = None
    sv = None
    if config.tracker_backend == "roboflow":
        try:
            import supervision as sv  # type: ignore
        except ImportError as error:
            raise ImportError(
                "tracker_backend=roboflow requires `supervision`. "
                "Install with: pip install supervision"
            ) from error

        tracker_name = config.tracker_name.lower()
        if tracker_name == "bytetrack":
            rf_tracker = sv.ByteTrack()
        else:
            raise ValueError(
                f"Unsupported tracker_name '{config.tracker_name}' for roboflow backend. "
                "Currently supported: bytetrack"
            )

    for frame_idx, image_path in enumerate(images):
        result = model(
            str(image_path),
            conf=config.conf,
            iou=config.iou,
            device=ultra_device,
            verbose=False,
        )[0]

        if result.boxes is None or result.keypoints is None:
            continue

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        boxes_conf = result.boxes.conf.cpu().numpy()
        keypoints_xy = result.keypoints.xy.cpu().numpy()
        keypoints_conf = (
            result.keypoints.conf.cpu().numpy()
            if result.keypoints.conf is not None
            else np.ones(keypoints_xy.shape[:2], dtype=float)
        )

        if boxes_xyxy.shape[0] < 2:
            continue

        # Keep top-2 by area.
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (
            boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        )
        top2_indices = np.argsort(areas)[::-1][:2]

        detections = []
        for idx in top2_indices:
            kp = keypoints_xy[idx]
            kp_conf = keypoints_conf[idx]
            pseudo_3d = [
                [float(x), float(y), 0.0, float(conf)] for (x, y), conf in zip(kp, kp_conf)
            ]
            detections.append(
                {
                    "bbox": boxes_xyxy[idx].astype(float),
                    "confidence": float(boxes_conf[idx]),
                    "pred_keypoints_3d": np.asarray(pseudo_3d, dtype=float),
                }
            )

        source_tracker_ids = [None, None]
        if rf_tracker is not None and sv is not None:
            rf_detections = sv.Detections(
                xyxy=np.asarray([det["bbox"] for det in detections], dtype=float),
                confidence=np.asarray(
                    [float(det["confidence"]) for det in detections], dtype=float
                ),
                class_id=np.zeros(len(detections), dtype=int),
            )
            tracked = rf_tracker.update_with_detections(rf_detections)
            tracked_ids = getattr(tracked, "tracker_id", None)

            if tracked_ids is not None and len(tracked_ids) == 2 and all(
                track_id is not None for track_id in tracked_ids
            ):
                source_tracker_ids = [int(track_id) for track_id in tracked_ids]
                parent_idx, child_idx = _resolve_role_indices_from_tracker_ids(
                    detections=detections,
                    tracker_ids=source_tracker_ids,
                    role_by_tracker_id=role_by_tracker_id,
                )
                assigned = _assign_from_role_indices(
                    detections=detections,
                    parent_idx=parent_idx,
                    child_idx=child_idx,
                )
                state = TwoPersonTrackerState(
                    parent_bbox=assigned[0].bbox.copy(),
                    child_bbox=assigned[1].bbox.copy(),
                )
            else:
                assigned, state = assign_two_person_tracks(detections, state=state)
        else:
            assigned, state = assign_two_person_tracks(detections, state=state)

        persons = []
        for assignment in assigned:
            pred = detections[assignment.detection_index]
            person = {
                "track_id": assignment.track_id,
                "track_label": assignment.track_label,
                "bbox_xyxy": assignment.bbox.tolist(),
                "confidence": assignment.confidence,
                "keypoints_3d": pred["pred_keypoints_3d"].tolist(),
            }
            source_tracker_id = source_tracker_ids[assignment.detection_index]
            if source_tracker_id is not None:
                person["source_tracker_id"] = source_tracker_id
            persons.append(person)

        frame_outputs.append(
            {
                "frame_idx": frame_idx,
                "image_name": image_path.name,
                "persons": persons,
            }
        )

    payload = {
        "runner_config": asdict(config),
        "resolved_device": ultra_device,
        "backend": "ultralytics_pose",
        "frames": frame_outputs,
    }
    output_path = Path(config.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Ultralytics 2D pose backend on image frames."
    )
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--image-folder", required=True, type=str)
    parser.add_argument("--output-json", required=True, type=str)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--iou", default=0.7, type=float)
    parser.add_argument("--max-images", default=None, type=int)
    parser.add_argument(
        "--tracker-backend", default="internal", choices=["internal", "roboflow"]
    )
    parser.add_argument("--tracker-name", default="bytetrack", type=str)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = UltraRunnerConfig(
        model_path=args.model_path,
        image_folder=args.image_folder,
        output_json=args.output_json,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_images=args.max_images,
        tracker_backend=args.tracker_backend,
        tracker_name=args.tracker_name,
    )
    run_ultralytics_pose_on_images(cfg)


if __name__ == "__main__":
    main()
