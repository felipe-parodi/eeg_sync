"""Refine parent/child keypoints with a top-down pose model (ViTPose).

The deployed pose model (a fine-tuned YOLO) finds people and keypoints in one
pass. Once identities are fixed, a heavier top-down model can re-estimate each
person's keypoints from their box. On past portal sessions ViTPose-plus
(Large or Huge) cut frame-to-frame keypoint jitter by ~24-38% versus the YOLO
keypoints (same frames, same people).

Top-down models have one failure mode that matters here: given the box of a
parent bending over a child, they can put the skeleton on the child. So the
refined skeleton is used only when it describes the same person as the YOLO
skeleton (keypoint similarity >= ``min_similarity``), and per keypoint only when
the refiner is confident. Keypoint confidences stay the YOLO values, because the
downstream interpolation and smoothing thresholds are calibrated to them.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from video_inference.offline_assoc import keypoint_similarity

# Large matches Huge on keypoint jitter (0.763 vs 0.766 of the YOLO jitter on
# 4 past cameras) at ~60% of the time; Base gives up much of the gain (0.905).
DEFAULT_REFINER = "usyd-community/vitpose-plus-large"

# [(frame_idx, (n, 4) xyxy boxes), ...] -> per frame, one (17, 3) array
# (x, y, score) per box. Called with a batch of frames at a time.
Refiner = Callable[[list[tuple[int, np.ndarray]]], list[list[np.ndarray]]]


def fuse_person_keypoints(
    student: np.ndarray,
    refined: np.ndarray,
    box_height: float,
    min_similarity: float = 0.5,
    min_refined_score: float = 0.3,
) -> tuple[np.ndarray, bool]:
    """Combine one person's YOLO and top-down keypoints.

    Args:
        student: (17, 3) YOLO keypoints (x, y, conf).
        refined: (17, 3) top-down keypoints (x, y, score).
        box_height: Person box height in pixels (keypoint similarity scale).
        min_similarity: Below this the refiner is assumed to have locked onto
            someone else, and the YOLO keypoints are kept unchanged.
        min_refined_score: Refined positions are used only for keypoints the
            refiner scores at least this high.

    Returns:
        ``(keypoints, used_refined)``; confidences are always the YOLO ones.
    """
    sim = keypoint_similarity(student, refined, scale=0.7 * box_height)
    if np.isnan(sim) or sim < min_similarity:
        return student.copy(), False
    fused = student.copy()
    take = refined[:, 2] >= min_refined_score
    fused[take, :2] = refined[take, :2]
    return fused, True


def refine_pose_table(
    tracks: pd.DataFrame,
    pose: pd.DataFrame,
    refiner: Refiner,
    batch_frames: int = 16,
) -> tuple[pd.DataFrame, dict]:
    """Refine every person-frame in a ``pose_3d`` table.

    Args:
        tracks: ``tracks_2d`` table (one row per person per frame).
        pose: Matching ``pose_3d`` table (17 rows per person per frame).
        refiner: Callable returning top-down keypoints for a batch of frames.
        batch_frames: Frames per refiner call.

    Returns:
        ``(refined_pose, summary)``. The refined table has the same columns
        and rows as ``pose``.
    """
    pose = pose.copy()
    kp_index = {f"kp_{k:03d}": k for k in range(17)}
    pose["_k"] = pose["keypoint_name"].map(kp_index)
    pose = pose.sort_values(["frame_idx", "track_id", "_k"]).reset_index(drop=True)
    xy = pose[["x_m", "y_m", "keypoint_confidence"]].to_numpy(dtype=float)
    starts = {
        (int(f), int(t)): i
        for i, (f, t, k) in enumerate(pose[["frame_idx", "track_id", "_k"]].to_numpy())
        if k == 0
    }
    refined_count = person_count = 0
    per_frame = [
        (
            int(frame_idx),
            frame_tracks["track_id"].to_numpy(int),
            frame_tracks[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float),
        )
        for frame_idx, frame_tracks in tracks.groupby("frame_idx", sort=True)
    ]
    for b in range(0, len(per_frame), max(1, batch_frames)):
        batch = per_frame[b : b + batch_frames]
        results = refiner([(f, boxes) for f, _, boxes in batch])
        for (frame_idx, track_ids, boxes), refs in zip(batch, results):
            for track_id, box, ref in zip(track_ids, boxes, refs):
                start = starts.get((frame_idx, int(track_id)))
                if start is None:
                    continue
                person_count += 1
                fused, used = fuse_person_keypoints(
                    xy[start : start + 17], ref, box_height=box[3] - box[1]
                )
                if used:
                    xy[start : start + 17] = fused
                    refined_count += 1
    pose[["x_m", "y_m", "keypoint_confidence"]] = xy
    pose = pose.drop(columns="_k")
    summary = {
        "persons": person_count,
        "persons_refined": refined_count,
        "share_refined": refined_count / person_count if person_count else 0.0,
    }
    return pose, summary


# ViTPose input and heatmap geometry (COCO top-down): crops are 192x256 and
# heatmaps 48x64. The math below mirrors transformers' VitPoseImageProcessor
# (box_to_center_and_scale, get_warp_matrix, DARK decoding, transform_preds),
# but crops with OpenCV and decodes on the GPU: the reference implementation
# warps each crop with scipy and decodes heatmaps on the CPU, which took ~2/3
# of the refinement time.
_IN_W, _IN_H = 192, 256
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _center_scale(boxes_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Box -> (center, scale/200) with the 192:256 aspect and 1.25 padding."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    center = np.stack([boxes[:, 0] + w * 0.5, boxes[:, 1] + h * 0.5], axis=1)
    aspect = _IN_W / _IN_H
    h = np.where(w > aspect * h, w / aspect, h)
    w = np.where(w < aspect * h, h * aspect, w)
    scale = np.stack([w, h], axis=1) / 200.0 * 1.25
    return center.astype(np.float32), scale.astype(np.float32)


def crop_people(image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    """Affine-crop each person box to a 192x256 RGB patch (uint8, NHWC)."""
    import cv2

    centers, scales = _center_scale(boxes_xyxy)
    crops = np.zeros((len(centers), _IN_H, _IN_W, 3), dtype=np.uint8)
    out_w, out_h = _IN_W - 1.0, _IN_H - 1.0
    for i, (center, scale) in enumerate(zip(centers, scales)):
        roi_w, roi_h = scale * 200.0
        sx, sy = out_w / roi_w, out_h / roi_h
        matrix = np.array(
            [
                [sx, 0.0, sx * (-center[0] + 0.5 * roi_w)],
                [0.0, sy, sy * (-center[1] + 0.5 * roi_h)],
            ],
            dtype=np.float32,
        )
        crops[i] = cv2.warpAffine(
            image_rgb,
            matrix,
            (_IN_W, _IN_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return crops


def normalize_crops(crops_nhwc: Any, device: str, dtype: Any) -> Any:
    """uint8 NHWC crops -> ImageNet-normalized NCHW tensor on ``device``."""
    import torch

    x = (
        crops_nhwc.to(device=device, non_blocking=True).permute(0, 3, 1, 2).float()
        / 255.0
    )
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    return ((x - mean) / std).to(dtype)


def _gaussian_blur_like_scipy(
    heatmaps: Any, sigma: float = 0.8, radius: int = 5
) -> Any:
    """Separable Gaussian blur matching scipy.ndimage.gaussian_filter (mode='reflect')."""
    import torch
    from torch.nn import functional

    offsets = torch.arange(
        -radius, radius + 1, device=heatmaps.device, dtype=heatmaps.dtype
    )
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    n, k, h, w = heatmaps.shape
    x = heatmaps.reshape(n * k, 1, h, w)
    # scipy 'reflect' mirrors including the edge sample (d c b a | a b c d).
    x = torch.cat([x[..., :radius].flip(-1), x, x[..., -radius:].flip(-1)], dim=-1)
    x = functional.conv2d(x, kernel.view(1, 1, 1, -1))
    x = torch.cat(
        [x[..., :radius, :].flip(-2), x, x[..., -radius:, :].flip(-2)], dim=-2
    )
    x = functional.conv2d(x, kernel.view(1, 1, -1, 1))
    return x.reshape(n, k, h, w)


def decode_heatmaps(heatmaps: Any, boxes_xyxy: np.ndarray) -> np.ndarray:
    """Heatmaps (N, 17, 64, 48) -> image keypoints (N, 17, 3) as (x, y, score).

    Argmax plus DARK sub-pixel refinement (log-heatmap Taylor expansion), then
    the unbiased mapping back to image coordinates, all on the heatmaps' device.
    """
    import torch
    from torch.nn import functional

    heat = heatmaps.float()
    n, k, h, w = heat.shape
    flat = heat.reshape(n, k, -1)
    scores, idx = flat.max(dim=-1)
    x = (idx % w).float()
    y = torch.div(idx, w, rounding_mode="floor").float()
    log_heat = torch.log(_gaussian_blur_like_scipy(heat).clamp(0.001, 50.0))
    padded = functional.pad(
        log_heat.reshape(n * k, 1, h, w), (1, 1, 1, 1), mode="replicate"
    )
    padded = padded.reshape(n, k, h + 2, w + 2)
    px = x.long() + 1
    py = y.long() + 1

    def at(dx: int, dy: int) -> Any:
        flat_idx = (py + dy) * (w + 2) + (px + dx)
        return padded.reshape(n, k, -1).gather(-1, flat_idx.unsqueeze(-1)).squeeze(-1)

    c = at(0, 0)
    gx = 0.5 * (at(1, 0) - at(-1, 0))
    gy = 0.5 * (at(0, 1) - at(0, -1))
    hxx = at(1, 0) - 2 * c + at(-1, 0)
    hyy = at(0, 1) - 2 * c + at(0, -1)
    hxy = 0.5 * (
        at(1, 1) - at(1, 0) - at(0, 1) + 2 * c - at(-1, 0) - at(0, -1) + at(-1, -1)
    )
    eps = float(np.finfo(np.float32).eps)
    hxx_e, hyy_e = hxx + eps, hyy + eps
    det = hxx_e * hyy_e - hxy * hxy
    inv_xx, inv_yy, inv_xy = hyy_e / det, hxx_e / det, -hxy / det
    x = x - (inv_xx * gx + inv_xy * gy)
    y = y - (inv_xy * gx + inv_yy * gy)

    centers, scales = _center_scale(boxes_xyxy)
    centers_t = torch.from_numpy(centers).to(heat.device)
    roi = torch.from_numpy(scales * 200.0).to(heat.device)
    img_x = x * (roi[:, :1] / (w - 1.0)) + centers_t[:, :1] - roi[:, :1] * 0.5
    img_y = y * (roi[:, 1:] / (h - 1.0)) + centers_t[:, 1:] - roi[:, 1:] * 0.5
    return torch.stack([img_x, img_y, scores], dim=-1).cpu().numpy()


def make_vitpose_refiner(
    frames_dir: Path,
    model_id: str = DEFAULT_REFINER,
    device: str = "cuda:0",
) -> Refiner:
    """Build a refiner that runs ViTPose on a camera's extracted frames.

    Args:
        frames_dir: Directory with ``frame_index.csv`` and frame images.
        model_id: Hugging Face ViTPose checkpoint.
        device: Torch device.

    Returns:
        A :data:`Refiner` callable.
    """
    import inspect

    import cv2
    import torch
    from transformers import VitPoseForPoseEstimation

    index = pd.read_csv(frames_dir / "frame_index.csv")
    names = dict(zip(index["frame_idx"].astype(int), index["image_name"]))
    on_gpu = not str(device).startswith("cpu")
    model = VitPoseForPoseEstimation.from_pretrained(model_id).to(device).eval()
    dtype = torch.float16 if on_gpu else torch.float32
    model = model.to(dtype)
    needs_dataset_index = "dataset_index" in inspect.signature(model.forward).parameters

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=8)

    def _read(frame_idx: int) -> np.ndarray:
        return cv2.cvtColor(
            cv2.imread(str(frames_dir / names[frame_idx])), cv2.COLOR_BGR2RGB
        )

    def refine(batch: list[tuple[int, np.ndarray]]) -> list[list[np.ndarray]]:
        batch = [(f, np.asarray(b, dtype=float).reshape(-1, 4)) for f, b in batch]
        results: list[list[np.ndarray]] = [[] for _ in batch]
        keep = [i for i, (_, b) in enumerate(batch) if len(b)]
        if not keep:
            return results
        images = list(pool.map(_read, [batch[i][0] for i in keep]))
        crops = np.concatenate(
            [crop_people(img, batch[i][1]) for img, i in zip(images, keep)]
        )
        boxes = np.concatenate([batch[i][1] for i in keep])
        pixels = normalize_crops(torch.from_numpy(crops), device, dtype)
        kwargs = {}
        if needs_dataset_index:  # ViTPose-plus: index 0 selects COCO-17
            kwargs["dataset_index"] = torch.zeros(
                len(boxes), dtype=torch.int64, device=device
            )
        with torch.no_grad():
            heatmaps = model(pixel_values=pixels, **kwargs).heatmaps
        keypoints = decode_heatmaps(heatmaps, boxes)
        start = 0
        for i in keep:
            n = len(batch[i][1])
            results[i] = list(keypoints[start : start + n])
            start += n
        return results

    return refine


def main(argv: Sequence[str] | None = None) -> None:
    """Refine a camera's parent/child keypoints with ViTPose."""
    parser = argparse.ArgumentParser(
        description="Re-estimate parent and child keypoints with a stronger pose model."
    )
    parser.add_argument("--camera-dir", type=Path, required=True)
    parser.add_argument("--tracks-input", default="tracks_2d_filtered.csv")
    parser.add_argument("--pose-input", default="pose_3d_filtered.csv")
    parser.add_argument("--pose-output", default="pose_3d_refined.csv")
    parser.add_argument("--summary-output", default="keypoint_refinement_summary.json")
    parser.add_argument("--model", default=DEFAULT_REFINER)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    camera_dir = args.camera_dir
    tracks = pd.read_csv(camera_dir / args.tracks_input)
    pose = pd.read_csv(camera_dir / args.pose_input)
    from video_inference.device import resolve_device

    device = args.device
    if not device.startswith("cuda:"):
        device = resolve_device(device)  # "auto" -> cuda / mps / cpu
    if device == "cuda":
        device = "cuda:0"
    refiner = make_vitpose_refiner(
        camera_dir / "frames", model_id=args.model, device=device
    )
    refined, summary = refine_pose_table(tracks, pose, refiner)
    refined.to_csv(camera_dir / args.pose_output, index=False)
    summary["model"] = args.model
    (camera_dir / args.summary_output).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"[refine] {camera_dir.name}: refined keypoints for {summary['share_refined']:.0%} "
        f"of {summary['persons']} person-frames."
    )


if __name__ == "__main__":
    main()
