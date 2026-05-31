"""Top-down ViTPose teacher: stronger COCO-17 pseudo-labels.

A person detector (YOLO) supplies boxes; ViTPose (HuggingFace ``transformers``)
predicts keypoints on each cropped, normalized person box. Top-down inference is
markedly more accurate than bottom-up yolo11x-pose for small, occluded, and
overhead people, so the resulting labels can let the distilled student exceed
the current portal model on those cases.

Output is written in the SAME ``pseudo_labels_raw.json`` schema as
``pseudo_label.py``, so ``dataset.py`` / ``train.py`` / ``evaluate.py`` consume
it unchanged. See docs/vitpose-teacher_260531.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .boxes import clean_detection_boxes

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "pose_training" / "data" / "frame_manifest.csv"
DEFAULT_OUTPUT = REPO_ROOT / "pose_training" / "data" / "pseudo_labels_vitpose.json"
DEFAULT_DETECTOR = "yolo11x-pose.pt"
DEFAULT_VITPOSE = "usyd-community/vitpose-base-simple"


@dataclass
class VitPoseConfig:
    """Configuration for the ViTPose top-down labeling stage."""

    manifest_path: Path = DEFAULT_MANIFEST
    output_path: Path = DEFAULT_OUTPUT
    detector_weights: str = DEFAULT_DETECTOR
    vitpose_model: str = DEFAULT_VITPOSE
    det_device: str = "1"
    torch_device: str = "cuda:1"
    det_conf: float = 0.3
    det_imgsz: int = 960
    max_persons: int = 6
    person_class_id: int = 0
    batch_size: int = 16


def boxes_xyxy_to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    """Convert xyxy boxes to COCO xywh (the format ViTPose's processor wants)."""
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    xywh = boxes.copy()
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    return xywh


def top_person_boxes(
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    class_ids: np.ndarray,
    person_class_id: int,
    max_persons: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep the top ``max_persons`` person boxes by confidence.

    Returns:
        (boxes_xyxy, confs) for the kept persons, highest confidence first.
    """
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    confs = np.asarray(confs, dtype=float).reshape(-1)
    class_ids = np.asarray(class_ids).reshape(-1)
    person = np.where(class_ids == person_class_id)[0]
    if person.size == 0:
        return np.zeros((0, 4)), np.zeros((0,))
    person_boxes = boxes_xyxy[person]
    person_confs = confs[person]
    # Drop overlapping/contained duplicate boxes before posing, so the same
    # person is not handed to ViTPose several times (the NMS-violation bug).
    kept = clean_detection_boxes(person_boxes, person_confs)[:max_persons]
    return person_boxes[kept], person_confs[kept]


def build_person_record(
    bbox_xyxy: np.ndarray,
    box_conf: float,
    keypoints_xy: np.ndarray,
    keypoint_scores: np.ndarray,
) -> Dict[str, Any]:
    """Assemble one person record in the shared pseudo-label schema."""
    keypoints_xy = np.asarray(keypoints_xy, dtype=float).reshape(-1, 2)
    keypoint_scores = np.asarray(keypoint_scores, dtype=float).reshape(-1)
    keypoints = [
        [float(x), float(y), float(score)]
        for (x, y), score in zip(keypoints_xy, keypoint_scores)
    ]
    return {
        "bbox_xyxy": [float(v) for v in np.asarray(bbox_xyxy, dtype=float).reshape(-1)],
        "box_conf": float(box_conf),
        "keypoints": keypoints,
    }


def _load_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def run_vitpose_labeling(
    config: VitPoseConfig,
    detector: Optional[Any] = None,
    processor: Optional[Any] = None,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """Label every manifest frame with detector boxes + ViTPose keypoints.

    Args:
        config: Stage configuration.
        detector: Optional YOLO model (callable on image paths). Loaded from
            ``config.detector_weights`` when None.
        processor: Optional ViTPose image processor. Loaded when None.
        model: Optional ViTPose model. Loaded when None.

    Returns:
        The payload dict written to ``config.output_path``.
    """
    import inspect

    import torch

    rows = _load_manifest(config.manifest_path)
    if detector is None:
        from ultralytics import YOLO

        detector = YOLO(config.detector_weights)
    if processor is None or model is None:
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        processor = processor or AutoProcessor.from_pretrained(config.vitpose_model)
        model = (
            model
            or VitPoseForPoseEstimation.from_pretrained(config.vitpose_model)
            .to(config.torch_device)
            .eval()
        )

    # ViTPose-plus checkpoints are mixture-of-experts and need a dataset index
    # to select the keypoint format; 0 selects COCO-17. The "simple" checkpoints
    # have no such argument.
    needs_dataset_index = "dataset_index" in inspect.signature(model.forward).parameters

    frame_records: List[Dict[str, Any]] = []
    batch = max(1, config.batch_size)
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        paths = [row["frame_path"] for row in chunk]
        det_results = detector(
            paths,
            imgsz=config.det_imgsz,
            conf=config.det_conf,
            device=config.det_device,
            classes=[config.person_class_id],
            verbose=False,
        )
        for row, det in zip(chunk, det_results):
            image_bgr = cv2.imread(row["frame_path"])
            height, width = image_bgr.shape[:2]
            record: Dict[str, Any] = {
                "frame_path": row["frame_path"],
                "participant": row["participant"],
                "angle": row["angle"],
                "split": row["split"],
                "source_video": row.get("source_video", ""),
                "source_timestamp_s": float(row.get("source_timestamp_s", 0.0)),
                "width": int(width),
                "height": int(height),
                "persons": [],
            }
            if det.boxes is None or len(det.boxes) == 0:
                frame_records.append(record)
                continue
            class_ids = (
                _to_numpy(det.boxes.cls).astype(int)
                if getattr(det.boxes, "cls", None) is not None
                else np.zeros(len(det.boxes), dtype=int)
            )
            boxes_xyxy, confs = top_person_boxes(
                _to_numpy(det.boxes.xyxy),
                _to_numpy(det.boxes.conf),
                class_ids,
                config.person_class_id,
                config.max_persons,
            )
            if boxes_xyxy.shape[0] == 0:
                frame_records.append(record)
                continue

            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            boxes_xywh = boxes_xyxy_to_xywh(boxes_xyxy)
            inputs = processor(
                rgb, boxes=[boxes_xywh.tolist()], return_tensors="pt"
            ).to(config.torch_device)
            forward_kwargs: Dict[str, Any] = {}
            if needs_dataset_index:
                forward_kwargs["dataset_index"] = torch.zeros(
                    boxes_xywh.shape[0], dtype=torch.int64, device=config.torch_device
                )
            with torch.no_grad():
                outputs = model(**inputs, **forward_kwargs)
            pose = processor.post_process_pose_estimation(
                outputs, boxes=[boxes_xywh.tolist()]
            )[0]
            persons = []
            for index, person in enumerate(pose):
                if index >= boxes_xyxy.shape[0]:
                    break
                persons.append(
                    build_person_record(
                        boxes_xyxy[index],
                        float(confs[index]),
                        _to_numpy(person["keypoints"]),
                        _to_numpy(person["scores"]),
                    )
                )
            record["persons"] = persons
            frame_records.append(record)

    payload = {
        "config": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()
        },
        "teacher": f"vitpose:{config.vitpose_model}",
        "frames": frame_records,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def summarize(payload: Dict[str, Any]) -> str:
    """Short summary: frame/instance counts and empty frames."""
    frames = payload["frames"]
    instances = sum(len(f["persons"]) for f in frames)
    empty = sum(1 for f in frames if not f["persons"])
    return f"Frames: {len(frames)}; instances: {instances}; empty: {empty}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label frames with detector boxes + ViTPose keypoints."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), type=str)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), type=str)
    parser.add_argument("--detector-weights", default=DEFAULT_DETECTOR, type=str)
    parser.add_argument("--vitpose-model", default=DEFAULT_VITPOSE, type=str)
    parser.add_argument("--det-device", default="1", type=str)
    parser.add_argument("--torch-device", default="cuda:1", type=str)
    parser.add_argument("--det-conf", default=0.3, type=float)
    parser.add_argument("--max-persons", default=6, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = VitPoseConfig(
        manifest_path=Path(args.manifest),
        output_path=Path(args.output),
        detector_weights=args.detector_weights,
        vitpose_model=args.vitpose_model,
        det_device=args.det_device,
        torch_device=args.torch_device,
        det_conf=args.det_conf,
        max_persons=args.max_persons,
        batch_size=args.batch_size,
    )
    payload = run_vitpose_labeling(config)
    print(f"Wrote ViTPose labels: {config.output_path}")
    print(summarize(payload))


if __name__ == "__main__":
    main()
