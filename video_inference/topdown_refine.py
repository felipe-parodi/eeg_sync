"""Refine parent/child keypoints with a top-down pose model (ViTPose).

The deployed pose model (a fine-tuned YOLO) finds people and keypoints in one
pass. Once identities are fixed, a heavier top-down model can re-estimate each
person's keypoints from their box. On past portal sessions ViTPose-plus-Huge
cut frame-to-frame keypoint jitter by a median of ~38% versus the YOLO
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

import numpy as np
import pandas as pd

from video_inference.offline_assoc import keypoint_similarity

DEFAULT_REFINER = "usyd-community/vitpose-plus-huge"

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
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    index = pd.read_csv(frames_dir / "frame_index.csv")
    names = dict(zip(index["frame_idx"].astype(int), index["image_name"]))
    processor = AutoProcessor.from_pretrained(model_id)
    on_gpu = not str(device).startswith("cpu")
    model = VitPoseForPoseEstimation.from_pretrained(model_id).to(device).eval()
    if on_gpu:
        model = model.half()
    needs_dataset_index = "dataset_index" in inspect.signature(model.forward).parameters

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=8)

    def _read(frame_idx: int) -> np.ndarray:
        return cv2.cvtColor(
            cv2.imread(str(frames_dir / names[frame_idx])), cv2.COLOR_BGR2RGB
        )

    def refine(batch: list[tuple[int, np.ndarray]]) -> list[list[np.ndarray]]:
        batch = [(f, np.asarray(b, dtype=float).reshape(-1, 4)) for f, b in batch]
        keep = [i for i, (_, b) in enumerate(batch) if len(b)]
        results: list[list[np.ndarray]] = [[] for _ in batch]
        if not keep:
            return results
        images = list(pool.map(_read, [batch[i][0] for i in keep]))
        boxes_xywh = []
        for i in keep:
            xywh = batch[i][1].copy()
            xywh[:, 2:] -= xywh[:, :2]
            boxes_xywh.append(xywh.tolist())
        inputs = processor(images, boxes=boxes_xywh, return_tensors="pt").to(device)
        if on_gpu:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        kwargs = {}
        if needs_dataset_index:  # ViTPose-plus: index 0 selects COCO-17
            kwargs["dataset_index"] = torch.zeros(
                inputs["pixel_values"].shape[0], dtype=torch.int64, device=device
            )
        with torch.no_grad():
            outputs = model(**inputs, **kwargs)
        outputs.heatmaps = outputs.heatmaps.float()
        people = processor.post_process_pose_estimation(outputs, boxes=boxes_xywh)
        for i, frame_people in zip(keep, people):
            results[i] = [
                np.c_[p["keypoints"].cpu().numpy(), p["scores"].cpu().numpy()]
                for p in frame_people
            ]
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
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    camera_dir = args.camera_dir
    tracks = pd.read_csv(camera_dir / args.tracks_input)
    pose = pd.read_csv(camera_dir / args.pose_input)
    from video_inference.device import resolve_device

    device = resolve_device(args.device)  # "auto" -> cuda / mps / cpu
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
