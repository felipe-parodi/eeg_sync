"""Runtime integration wrapper for SAM-3D-Body submodule."""

from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .device import resolve_device, resolve_inference_mode
from .tracking import TwoPersonTrackerState, assign_two_person_tracks


SUBMODULE_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "sam-3d-body"


@dataclass
class RunnerConfig:
    """Configuration for a SAM-3D frame inference run."""

    checkpoint_path: str
    mhr_path: str
    image_folder: str
    output_json: str
    device: str = "auto"
    inference_mode: str = "auto"
    detector_name: str = "sam3"
    bbox_thresh: float = 0.5
    max_images: Optional[int] = None
    use_mask: bool = False


def _ensure_submodule_importable() -> None:
    if not SUBMODULE_ROOT.exists():
        raise FileNotFoundError(
            f"SAM-3D submodule not found at {SUBMODULE_ROOT}. "
            "Initialize submodules before running inference."
        )
    submodule_str = str(SUBMODULE_ROOT)
    if submodule_str not in sys.path:
        sys.path.insert(0, submodule_str)


def _patch_cpu_device_paths(model: Any) -> None:
    """
    Patch CUDA-hardcoded method paths for CPU compatibility.

    Upstream currently hardcodes `.cuda()` in get_ray_condition.
    """

    def _get_ray_condition_cpu_safe(self, batch):  # noqa: ANN001
        import torch

        b, n, _, h, w = batch["img"].shape
        device = batch["img"].device
        meshgrid_xy = (
            torch.stack(
                torch.meshgrid(torch.arange(h), torch.arange(w), indexing="xy"), dim=2
            )[None, None, :, :, :]
            .repeat(b, n, 1, 1, 1)
            .to(device)
        )
        meshgrid_xy = (
            meshgrid_xy / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
        )
        meshgrid_xy = (
            meshgrid_xy
            - batch["affine_trans"][:, :, None, None, [0, 1], [2, 2]]
            / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
        )
        meshgrid_xy = (
            meshgrid_xy - batch["cam_int"][:, None, None, None, [0, 1], [2, 2]]
        )
        meshgrid_xy = (
            meshgrid_xy / batch["cam_int"][:, None, None, None, [0, 1], [0, 1]]
        )
        return meshgrid_xy.permute(0, 1, 4, 2, 3).to(batch["img"].dtype)

    model.get_ray_condition = types.MethodType(_get_ray_condition_cpu_safe, model)


def _build_estimator(config: RunnerConfig) -> tuple[Any, str, str]:
    _ensure_submodule_importable()

    import torch
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from tools.build_detector import HumanDetector

    device = resolve_device(config.device)
    inference_mode = resolve_inference_mode(device, config.inference_mode)

    torch_device = torch.device(device)
    model, model_cfg = load_sam_3d_body(
        config.checkpoint_path, device=torch_device, mhr_path=config.mhr_path
    )
    if device == "cpu":
        _patch_cpu_device_paths(model)

    detector = HumanDetector(name=config.detector_name, device=torch_device)

    # Avoid segmentor/FOV by default for stability and speed.
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=detector,
        human_segmentor=None,
        fov_estimator=None,
    )
    return estimator, device, inference_mode


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


def run_sam3d_on_images(config: RunnerConfig) -> Dict[str, Any]:
    """
    Run SAM-3D-Body on an image folder and export per-frame person outputs.

    This wrapper is intentionally minimal and designed for integration testing.
    """
    image_folder = Path(config.image_folder)
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    estimator, resolved_device, resolved_mode = _build_estimator(config)
    images = _iter_image_paths(image_folder, config.max_images)

    state: Optional[TwoPersonTrackerState] = None
    frame_outputs: List[Dict[str, Any]] = []

    for frame_idx, image_path in enumerate(images):
        predictions = estimator.process_one_image(
            str(image_path),
            bbox_thr=config.bbox_thresh,
            use_mask=config.use_mask and resolved_device == "cuda",
            inference_type=resolved_mode,
        )

        if len(predictions) < 2:
            # Skip under-detected frames for now.
            continue

        # Keep top-2 detections by bbox area.
        detections = sorted(
            predictions,
            key=lambda det: float(
                max(0.0, det["bbox"][2] - det["bbox"][0])
                * max(0.0, det["bbox"][3] - det["bbox"][1])
            ),
            reverse=True,
        )[:2]
        det_rows = [
            {
                "bbox": np.asarray(det["bbox"], dtype=float),
                "confidence": float(det.get("mask_score", 1.0)),
                "pred_keypoints_3d": np.asarray(det["pred_keypoints_3d"], dtype=float),
            }
            for det in detections
        ]
        assigned, state = assign_two_person_tracks(det_rows, state=state)

        persons = []
        for assignment in assigned:
            pred = det_rows[assignment.detection_index]
            persons.append(
                {
                    "track_id": assignment.track_id,
                    "track_label": assignment.track_label,
                    "bbox_xyxy": assignment.bbox.tolist(),
                    "confidence": assignment.confidence,
                    "keypoints_3d": pred["pred_keypoints_3d"].tolist(),
                }
            )

        frame_outputs.append(
            {
                "frame_idx": frame_idx,
                "image_name": image_path.name,
                "persons": persons,
            }
        )

    payload = {
        "runner_config": asdict(config),
        "resolved_device": resolved_device,
        "resolved_inference_mode": resolved_mode,
        "frames": frame_outputs,
    }
    output_path = Path(config.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SAM-3D-Body on image frames (GPU+CPU compatible wrapper)."
    )
    parser.add_argument("--checkpoint-path", required=True, type=str)
    parser.add_argument("--mhr-path", required=True, type=str)
    parser.add_argument("--image-folder", required=True, type=str)
    parser.add_argument("--output-json", required=True, type=str)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--inference-mode", default="auto", choices=["auto", "full", "body"]
    )
    parser.add_argument("--detector-name", default="sam3", type=str)
    parser.add_argument("--bbox-thresh", default=0.5, type=float)
    parser.add_argument("--max-images", default=None, type=int)
    parser.add_argument("--use-mask", action="store_true", default=False)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = RunnerConfig(
        checkpoint_path=args.checkpoint_path,
        mhr_path=args.mhr_path,
        image_folder=args.image_folder,
        output_json=args.output_json,
        device=args.device,
        inference_mode=args.inference_mode,
        detector_name=args.detector_name,
        bbox_thresh=args.bbox_thresh,
        max_images=args.max_images,
        use_mask=args.use_mask,
    )
    run_sam3d_on_images(cfg)


if __name__ == "__main__":
    main()
