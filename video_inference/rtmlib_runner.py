"""RTMLib pose inference runner with 2D (Body) and 3D (Wholebody3d) support."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .device import resolve_device
from .tracking import (
    TwoPersonTrackerState,
    assign_two_person_tracks,
)

# COCO-WholeBody has 133 keypoints; the first 17 are standard COCO body keypoints.
_COCO17_SLICE = slice(0, 17)


@dataclass
class RtmlibRunnerConfig:
    """Configuration for RTMLib frame inference."""

    image_folder: str
    output_json: str
    device: str = "auto"
    backend: str = "onnxruntime"
    mode: str = "balanced"
    mode_3d: bool = True
    det_frequency: int = 1
    kpt_thr: float = 0.3
    max_persons: int = 2
    max_images: Optional[int] = None
    keep_empty_frames: bool = True
    enforce_exact_person_count: bool = False
    tracker_backend: str = "internal"
    tracker_name: str = "bytetrack"


def _iter_image_paths(image_folder: Path, max_images: Optional[int]) -> List[Path]:
    """Discover and sort frame images by filename."""
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


def _resolve_rtmlib_device(device_preference: str) -> str:
    """Map our device convention to rtmlib/onnxruntime device string."""
    resolved = resolve_device(device_preference)
    # rtmlib accepts "cuda" and "cpu" directly for onnxruntime backend.
    if resolved in ("cuda", "mps"):
        return "cuda"
    return "cpu"


def _bbox_from_keypoints_2d(kpts_2d: np.ndarray, expansion: float = 1.1) -> List[float]:
    """Derive xyxy bounding box from 2D keypoint array.

    Args:
        kpts_2d: Keypoint array of shape (N, 2).
        expansion: Padding factor around the keypoint extent.

    Returns:
        Bounding box as [x1, y1, x2, y2].
    """
    xs = kpts_2d[:, 0]
    ys = kpts_2d[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    half_w = (x_max - x_min) / 2.0 * expansion
    half_h = (y_max - y_min) / 2.0 * expansion
    return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]


def _track_label_for_id(
    track_id: int, max_persons: int, tracker_backend: str
) -> str:
    """Return a human-readable label for a track slot."""
    if max_persons == 2 and tracker_backend == "internal":
        return "parent" if track_id == 0 else "child"
    return f"person_{track_id:02d}"


def _assign_output_track_slot(
    source_track_id: int,
    frame_idx: int,
    max_persons: int,
    source_to_output: Dict[int, int],
    output_to_source: Dict[int, int],
    last_seen_by_output: Dict[int, int],
    active_output_slots: set,
) -> int:
    """Map external tracker IDs into a bounded set of output slots.

    ByteTrack can emit new IDs over time for the same physical person.
    This recycles the oldest inactive slot to keep IDs in [0, max_persons).
    """
    existing_output = source_to_output.get(source_track_id)
    if existing_output is not None:
        last_seen_by_output[existing_output] = frame_idx
        active_output_slots.add(existing_output)
        return existing_output

    for output_slot in range(max_persons):
        if output_slot not in output_to_source:
            source_to_output[source_track_id] = output_slot
            output_to_source[output_slot] = source_track_id
            last_seen_by_output[output_slot] = frame_idx
            active_output_slots.add(output_slot)
            return output_slot

    recyclable_slots = [
        output_slot
        for output_slot in range(max_persons)
        if output_slot not in active_output_slots
    ]
    if not recyclable_slots:
        recyclable_slots = list(range(max_persons))

    recycled_slot = min(
        recyclable_slots,
        key=lambda output_slot: last_seen_by_output.get(output_slot, -1),
    )
    old_source_id = output_to_source.get(recycled_slot)
    if old_source_id is not None:
        source_to_output.pop(old_source_id, None)

    source_to_output[source_track_id] = recycled_slot
    output_to_source[recycled_slot] = source_track_id
    last_seen_by_output[recycled_slot] = frame_idx
    active_output_slots.add(recycled_slot)
    return recycled_slot


def run_rtmlib_pose_on_images(config: RtmlibRunnerConfig) -> Dict[str, Any]:
    """Run rtmlib pose model on extracted frames.

    Supports two solution classes:
    - ``Wholebody3d``: 133 whole-body keypoints with 3D coordinates (RTMW3D).
    - ``Body``: 17 COCO body keypoints with 2D coordinates (RTMPose).

    Both are subsetted/normalised to COCO-17 and output in the same JSON
    payload format expected by the pipeline schema exporter.

    Args:
        config: Runner configuration.

    Returns:
        JSON-serialisable payload with per-frame person detections.
    """
    image_folder = Path(config.image_folder)
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    try:
        if config.mode_3d:
            from rtmlib import Wholebody3d  # type: ignore
        else:
            from rtmlib import Body  # type: ignore
    except ImportError as error:
        raise ImportError(
            "rtmlib is required for this backend. "
            "Install with: pip install rtmlib"
        ) from error

    if config.max_persons < 1:
        raise ValueError("max_persons must be >= 1")

    if config.tracker_backend not in {"internal", "roboflow"}:
        raise ValueError(
            f"Invalid tracker_backend '{config.tracker_backend}'. "
            "Expected 'internal' or 'roboflow'."
        )
    if config.tracker_backend == "internal" and config.max_persons != 2:
        raise ValueError(
            "tracker_backend=internal currently supports max_persons=2 only. "
            "Use tracker_backend=roboflow for 3+ people."
        )

    rtmlib_device = _resolve_rtmlib_device(config.device)

    # Use the solution class directly (not PoseTracker) because
    # PoseTracker has a bug where it returns only 2 values for
    # RTMPose3d in certain tracking code paths.  We use our own
    # TwoPersonTrackerState for ID assignment anyway.
    if config.mode_3d:
        model = Wholebody3d(
            mode=config.mode,
            backend=config.backend,
            device=rtmlib_device,
        )
    else:
        model = Body(
            mode=config.mode,
            backend=config.backend,
            device=rtmlib_device,
        )

    # Set up tracker.
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
            rf_tracker = sv.ByteTrack(track_buffer=60)
        else:
            raise ValueError(
                f"Unsupported tracker_name '{config.tracker_name}' for "
                f"roboflow backend. Currently supported: bytetrack"
            )

    images = _iter_image_paths(image_folder, config.max_images)
    two_person_state: Optional[TwoPersonTrackerState] = None
    output_id_by_source_tracker_id: Dict[int, int] = {}
    source_tracker_id_by_output_id: Dict[int, int] = {}
    last_seen_frame_by_output_id: Dict[int, int] = {}
    frame_outputs: List[Dict[str, Any]] = []

    for frame_idx, image_path in enumerate(images):
        import cv2  # lazy import — only needed at inference time

        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"WARNING: Could not read {image_path.name}, skipping.")
            continue

        frame_output: Dict[str, Any] = {
            "frame_idx": frame_idx,
            "image_name": image_path.name,
            "persons": [],
        }

        if config.mode_3d:
            keypoints_3d, scores, _kpts_simcc, keypoints_2d = model(frame)
        else:
            keypoints_2d_raw, scores = model(frame)
            keypoints_2d = keypoints_2d_raw
            keypoints_3d = None

        # Handle empty detections.
        if not isinstance(scores, np.ndarray) or scores.size == 0:
            if config.keep_empty_frames and not config.enforce_exact_person_count:
                frame_outputs.append(frame_output)
            continue

        # Ensure arrays are at least 2-D: (N_persons, N_kpts, ...).
        if scores.ndim == 1:
            scores = scores[np.newaxis, :]
        if keypoints_2d.ndim == 2:
            keypoints_2d = keypoints_2d[np.newaxis, :]
        if keypoints_3d is not None and keypoints_3d.ndim == 2:
            keypoints_3d = keypoints_3d[np.newaxis, :]

        n_persons = scores.shape[0]

        # Build raw detections list (before tracking assignment).
        detections: List[Dict[str, Any]] = []
        for p_idx in range(min(n_persons, config.max_persons)):
            kpts_2d_person = keypoints_2d[p_idx]  # (N_kpts, 2)
            scores_person = scores[p_idx]  # (N_kpts,)

            # Subset to COCO-17 if wholebody (133 keypoints).
            kpts_2d_coco17 = kpts_2d_person[_COCO17_SLICE]
            scores_coco17 = scores_person[_COCO17_SLICE]

            bbox = _bbox_from_keypoints_2d(kpts_2d_coco17)
            mean_conf = float(np.mean(scores_coco17[scores_coco17 > config.kpt_thr]))
            if np.isnan(mean_conf):
                mean_conf = float(np.mean(scores_coco17))

            if config.mode_3d and keypoints_3d is not None:
                kpts_3d_person = keypoints_3d[p_idx][_COCO17_SLICE]  # (17, 3)
                # Use 2D pixel coords for x,y and 3D model output for z depth.
                # kpts_3d_person x,y are in model-normalised space (not pixels),
                # so we must use kpts_2d_coco17 (pixel space) for visualisation
                # and downstream pipeline compatibility.
                kp_list = [
                    [float(kpts_2d_coco17[k, 0]),
                     float(kpts_2d_coco17[k, 1]),
                     float(kpts_3d_person[k, 2]),
                     float(scores_coco17[k])]
                    for k in range(kpts_3d_person.shape[0])
                ]
            else:
                kp_list = [
                    [float(kpts_2d_coco17[k, 0]),
                     float(kpts_2d_coco17[k, 1]),
                     0.0,
                     float(scores_coco17[k])]
                    for k in range(kpts_2d_coco17.shape[0])
                ]

            detections.append({
                "bbox": bbox,
                "confidence": mean_conf,
                "pred_keypoints_3d": kp_list,
            })

        if not detections:
            if config.keep_empty_frames and not config.enforce_exact_person_count:
                frame_outputs.append(frame_output)
            continue

        # Assign track IDs via ByteTrack or internal 2-person tracker.
        persons: List[Dict[str, Any]] = []
        if rf_tracker is not None and sv is not None:
            rf_detections = sv.Detections(
                xyxy=np.asarray([d["bbox"] for d in detections], dtype=float),
                confidence=np.asarray(
                    [float(d["confidence"]) for d in detections], dtype=float
                ),
                class_id=np.zeros(len(detections), dtype=int),
            )
            tracked = rf_tracker.update_with_detections(rf_detections)
            tracked_ids = getattr(tracked, "tracker_id", None)
            if tracked_ids is not None:
                tracked_ids_list = np.asarray(tracked_ids).tolist()
                active_output_slots: set = set()
                for det_idx, source_track_id in enumerate(tracked_ids_list):
                    if det_idx >= len(detections):
                        break
                    if source_track_id is None:
                        continue
                    source_track_id = int(source_track_id)
                    output_track_id = _assign_output_track_slot(
                        source_track_id=source_track_id,
                        frame_idx=frame_idx,
                        max_persons=config.max_persons,
                        source_to_output=output_id_by_source_tracker_id,
                        output_to_source=source_tracker_id_by_output_id,
                        last_seen_by_output=last_seen_frame_by_output_id,
                        active_output_slots=active_output_slots,
                    )
                    det = detections[det_idx]
                    persons.append({
                        "track_id": output_track_id,
                        "track_label": _track_label_for_id(
                            output_track_id,
                            config.max_persons,
                            config.tracker_backend,
                        ),
                        "bbox_xyxy": [float(v) for v in det["bbox"]],
                        "confidence": det["confidence"],
                        "keypoints_3d": det["pred_keypoints_3d"],
                        "source_tracker_id": source_track_id,
                    })
        elif len(detections) >= 2 and config.max_persons == 2:
            top2 = sorted(
                detections,
                key=lambda det: float(
                    (det["bbox"][2] - det["bbox"][0])
                    * (det["bbox"][3] - det["bbox"][1])
                ),
                reverse=True,
            )[:2]
            assigned, two_person_state = assign_two_person_tracks(
                top2, state=two_person_state
            )
            for assignment in assigned:
                det = top2[assignment.detection_index]
                persons.append({
                    "track_id": assignment.track_id,
                    "track_label": assignment.track_label,
                    "bbox_xyxy": [float(v) for v in assignment.bbox.tolist()],
                    "confidence": assignment.confidence,
                    "keypoints_3d": det["pred_keypoints_3d"],
                })
        else:
            for p_idx, det in enumerate(detections):
                label = _track_label_for_id(
                    p_idx, config.max_persons, config.tracker_backend
                )
                persons.append({
                    "track_id": p_idx,
                    "track_label": label,
                    "bbox_xyxy": [float(v) for v in det["bbox"]],
                    "confidence": det["confidence"],
                    "keypoints_3d": det["pred_keypoints_3d"],
                })

        persons = sorted(persons, key=lambda p: int(p["track_id"]))
        frame_output["persons"] = persons
        frame_output["num_persons_detected"] = len(persons)

        if config.enforce_exact_person_count and len(persons) != config.max_persons:
            if config.keep_empty_frames:
                frame_output["persons"] = []
                frame_output["num_persons_detected"] = 0
                frame_outputs.append(frame_output)
            continue

        if persons or config.keep_empty_frames:
            frame_outputs.append(frame_output)

    payload = {
        "runner_config": asdict(config),
        "resolved_device": rtmlib_device,
        "backend": "rtmlib_wholebody3d" if config.mode_3d else "rtmlib_body",
        "frames": frame_outputs,
    }
    output_path = Path(config.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for standalone rtmlib runner."""
    parser = argparse.ArgumentParser(
        description="Run rtmlib pose backend on image frames."
    )
    parser.add_argument("--image-folder", required=True, type=str)
    parser.add_argument("--output-json", required=True, type=str)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument(
        "--backend", default="onnxruntime",
        choices=["onnxruntime", "opencv", "openvino"],
    )
    parser.add_argument(
        "--mode", default="balanced",
        choices=["performance", "balanced", "lightweight"],
    )
    parser.add_argument(
        "--mode-3d", action=argparse.BooleanOptionalAction, default=True,
        help="Use Wholebody3d (3D) if set, Body (2D) otherwise.",
    )
    parser.add_argument("--det-frequency", default=1, type=int)
    parser.add_argument("--kpt-thr", default=0.3, type=float)
    parser.add_argument("--max-persons", default=2, type=int)
    parser.add_argument("--max-images", default=None, type=int)
    parser.add_argument(
        "--keep-empty-frames",
        action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--enforce-exact-person-count",
        action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument(
        "--tracker-backend",
        default="internal",
        choices=["internal", "roboflow"],
    )
    parser.add_argument("--tracker-name", default="bytetrack", type=str)
    return parser


def main() -> None:
    """Entry point for standalone rtmlib runner."""
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = RtmlibRunnerConfig(
        image_folder=args.image_folder,
        output_json=args.output_json,
        device=args.device,
        backend=args.backend,
        mode=args.mode,
        mode_3d=args.mode_3d,
        det_frequency=args.det_frequency,
        kpt_thr=args.kpt_thr,
        max_persons=args.max_persons,
        max_images=args.max_images,
        keep_empty_frames=args.keep_empty_frames,
        enforce_exact_person_count=args.enforce_exact_person_count,
        tracker_backend=args.tracker_backend,
        tracker_name=args.tracker_name,
    )
    run_rtmlib_pose_on_images(cfg)


if __name__ == "__main__":
    main()
