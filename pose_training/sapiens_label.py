"""Sapiens (Meta) top-down teacher: a 4th, COCO-bias-breaking COCO-17 source.

Sapiens is a ViT pretrained on 300M human images and is state-of-the-art on
hard/occluded/unusual poses -- exactly the cases the three COCO-trained teachers
(yolo11x-pose, ViTPose, RTMPose) share a blind spot on. We run the goliath
torchscript checkpoint top-down on detector boxes; its 308 keypoints follow the
COCO-WholeBody order, so the first 17 ARE COCO-17 (verified visually).

Pipeline per person box: affine-crop to the model's 768x1024 input, normalize
(ImageNet), run -> (308, 256, 192) heatmaps, take the first 17, argmax-decode to
the crop, then inverse-affine back to image coordinates. Output uses the shared
``pseudo_labels_raw.json`` schema.
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

from video_inference.weights import resolve_weights

from .boxes import clean_detection_boxes

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "pose_training" / "data" / "frame_manifest.csv"
DEFAULT_OUTPUT = REPO_ROOT / "pose_training" / "data" / "pseudo_labels_sapiens.json"
DEFAULT_DETECTOR = "yolo11x-pose.pt"
SAPIENS_REPO = "facebook/sapiens-pose-0.6b-torchscript"
MODEL_W, MODEL_H = 768, 1024
HEAT_W, HEAT_H = 192, 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class SapiensConfig:
    """Configuration for the Sapiens top-down labeling stage."""

    manifest_path: Path = DEFAULT_MANIFEST
    output_path: Path = DEFAULT_OUTPUT
    detector_weights: str = DEFAULT_DETECTOR
    sapiens_repo: str = SAPIENS_REPO
    det_device: str = "1"
    torch_device: str = "cuda:1"
    det_conf: float = 0.3
    det_imgsz: int = 960
    max_persons: int = 4
    person_class_id: int = 0
    pad: float = 1.25


def aspect_affine(bbox: List[float], pad: float) -> np.ndarray:
    """Return the 2x3 affine mapping image -> model input for a person box.

    The box is padded and forced to the model's 3:4 (W:H) aspect ratio so the
    crop is not distorted, then mapped onto the MODEL_W x MODEL_H canvas.
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * pad, (y2 - y1) * pad
    aspect = MODEL_W / MODEL_H
    if w / max(h, 1e-6) > aspect:
        h = w / aspect
    else:
        w = h * aspect
    src = np.array(
        [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx - w / 2, cy + h / 2]],
        dtype=np.float32,
    )
    dst = np.array([[0, 0], [MODEL_W, 0], [0, MODEL_H]], dtype=np.float32)
    return cv2.getAffineTransform(src, dst)


def preprocess_crop(image_bgr: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Warp+normalize one person crop into a (3, H, W) float array."""
    crop = cv2.warpAffine(image_bgr, affine, (MODEL_W, MODEL_H), flags=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return rgb.transpose(2, 0, 1)


def decode_keypoints(heatmaps: np.ndarray, affine: np.ndarray) -> List[List[float]]:
    """Decode the first 17 heatmaps to COCO-17 image-space keypoints.

    Args:
        heatmaps: (>=17, HEAT_H, HEAT_W) array.
        affine: The image->input affine used for this crop.

    Returns:
        17 x [x, y, score] in original image coordinates.
    """
    inv = cv2.invertAffineTransform(affine)
    keypoints: List[List[float]] = []
    for k in range(17):
        heat = heatmaps[k]
        idx = int(np.argmax(heat))
        hy, hx = divmod(idx, heat.shape[1])
        score = float(heat[hy, hx])
        # heatmap cell center -> model input pixels
        x_in = (hx + 0.5) * (MODEL_W / HEAT_W)
        y_in = (hy + 0.5) * (MODEL_H / HEAT_H)
        img_x = inv[0, 0] * x_in + inv[0, 1] * y_in + inv[0, 2]
        img_y = inv[1, 0] * x_in + inv[1, 1] * y_in + inv[1, 2]
        keypoints.append([float(img_x), float(img_y), score])
    return keypoints


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def _load_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _person_boxes(det: Any, config: SapiensConfig) -> Tuple[np.ndarray, np.ndarray]:
    if det.boxes is None or len(det.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,))
    class_ids = (
        _to_numpy(det.boxes.cls).astype(int)
        if getattr(det.boxes, "cls", None) is not None
        else np.zeros(len(det.boxes), dtype=int)
    )
    boxes = _to_numpy(det.boxes.xyxy)
    confs = _to_numpy(det.boxes.conf)
    person = np.where(class_ids == config.person_class_id)[0]
    if person.size == 0:
        return np.zeros((0, 4)), np.zeros((0,))
    pboxes, pconfs = boxes[person], confs[person]
    kept = clean_detection_boxes(pboxes, pconfs)[: config.max_persons]
    return pboxes[kept], pconfs[kept]


def run_sapiens_labeling(
    config: SapiensConfig,
    detector: Optional[Any] = None,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """Label every manifest frame with detector boxes + Sapiens keypoints."""
    import torch

    rows = _load_manifest(config.manifest_path)
    if detector is None:
        from ultralytics import YOLO

        detector = YOLO(resolve_weights(config.detector_weights))
    if model is None:
        from huggingface_hub import hf_hub_download

        files = _repo_pt2(config.sapiens_repo)
        path = hf_hub_download(config.sapiens_repo, files)
        model = torch.jit.load(path).to(config.torch_device).eval()

    frame_records: List[Dict[str, Any]] = []
    for row in rows:
        image = cv2.imread(row["frame_path"])
        height, width = image.shape[:2]
        det = detector(
            row["frame_path"],
            imgsz=config.det_imgsz,
            conf=config.det_conf,
            device=config.det_device,
            classes=[config.person_class_id],
            verbose=False,
        )[0]
        boxes, confs = _person_boxes(det, config)
        persons: List[Dict[str, Any]] = []
        if boxes.shape[0] > 0:
            affines = [aspect_affine(b.tolist(), config.pad) for b in boxes]
            batch = np.stack([preprocess_crop(image, a) for a in affines])
            with torch.no_grad():
                heatmaps = model(torch.from_numpy(batch).to(config.torch_device))
            heatmaps = _to_numpy(heatmaps)
            for i in range(boxes.shape[0]):
                persons.append(
                    {
                        "bbox_xyxy": [float(v) for v in boxes[i]],
                        "box_conf": float(confs[i]),
                        "keypoints": decode_keypoints(heatmaps[i], affines[i]),
                    }
                )
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
        "teacher": f"sapiens:{config.sapiens_repo}",
        "frames": frame_records,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _repo_pt2(repo: str) -> str:
    from huggingface_hub import list_repo_files

    return next(f for f in list_repo_files(repo) if f.endswith(".pt2"))


def summarize(payload: Dict[str, Any]) -> str:
    frames = payload["frames"]
    instances = sum(len(f["persons"]) for f in frames)
    empty = sum(1 for f in frames if not f["persons"])
    return f"Frames: {len(frames)}; instances: {instances}; empty: {empty}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label frames with Sapiens pose.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), type=str)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), type=str)
    parser.add_argument("--sapiens-repo", default=SAPIENS_REPO, type=str)
    parser.add_argument("--det-device", default="1", type=str)
    parser.add_argument("--torch-device", default="cuda:1", type=str)
    parser.add_argument("--max-persons", default=4, type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = SapiensConfig(
        manifest_path=Path(args.manifest),
        output_path=Path(args.output),
        sapiens_repo=args.sapiens_repo,
        det_device=args.det_device,
        torch_device=args.torch_device,
        max_persons=args.max_persons,
    )
    payload = run_sapiens_labeling(config)
    print(f"Wrote Sapiens labels: {config.output_path}")
    print(summarize(payload))


if __name__ == "__main__":
    main()
