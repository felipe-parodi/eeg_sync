"""Run Gazelle gaze estimation on video frames with precomputed head bboxes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from gaze_analysis.config import (
    SessionConfig,
    add_time_range_args,
    filter_by_time_range,
    load_session_config,
    resolve_time_range,
)
from gaze_analysis.gaze_schema import GAZE_HEATMAP_COLUMNS
from gaze_analysis.head_bbox import extract_head_bboxes


@dataclass
class GazelleConfig:
    """Configuration for Gazelle inference on one camera."""

    camera_dir: str
    session_config_path: str
    camera_id: str = "camera_a"
    device: str = "auto"
    model_name: str = "gazelle_dinov2_vitl14_inout"
    batch_size: int = 16
    max_frames: int = 0
    start_s: float | None = None
    end_s: float | None = None
    time_block: str | None = None
    pose_input: str = "pose_3d.csv"
    tracks_input: str = "tracks_2d.csv"
    output_csv: str = "gaze_heatmap.csv"
    output_npz: str = "gaze_heatmaps.npz"


def _load_gazelle_model(
    model_name: str, device: str
) -> Tuple[Any, Any]:
    """Load a Gazelle model via torch.hub.

    Args:
        model_name: One of the gazelle model names from hubconf.py.
        device: Torch device string ("cpu", "cuda", "mps").

    Returns:
        (model, transform) tuple.
    """
    import torch

    model, transform = torch.hub.load(
        "fkryan/gazelle",
        model_name,
        pretrained=True,
        source="github",
    )
    model = model.to(device)
    model.eval()
    return model, transform


def _resolve_device(preference: str) -> str:
    """Resolve device preference to actual device string."""
    from video_inference.device import resolve_device

    return resolve_device(preference)


def _load_frame_image(frame_path: Path, transform: Any) -> Any:
    """Load a single frame and apply the Gazelle transform.

    Args:
        frame_path: Path to the JPEG frame.
        transform: Torchvision transform from Gazelle.

    Returns:
        Transformed image tensor.
    """
    from PIL import Image

    img = Image.open(frame_path).convert("RGB")
    return transform(img)


def _find_frame_paths(camera_dir: Path) -> Dict[int, Path]:
    """Map frame_idx to frame file paths.

    Args:
        camera_dir: Camera output directory containing frames/ subdirectory.

    Returns:
        Dict mapping frame_idx (int) to Path.
    """
    frames_dir = camera_dir / "frames"
    if not frames_dir.exists():
        return {}

    frame_paths: Dict[int, Path] = {}
    for p in sorted(frames_dir.glob("frame_*.jpg")):
        # Filename format: frame_000001.jpg
        try:
            idx = int(p.stem.split("_")[1])
            frame_paths[idx] = p
        except (IndexError, ValueError):
            continue
    return frame_paths


def run_gazelle_inference(config: GazelleConfig) -> Dict[str, Any]:
    """Run Gazelle gaze estimation on all frames for one camera.

    Loads pose_3d and tracks_2d CSVs, extracts head bboxes, runs Gazelle
    model inference in batches, and writes gaze_heatmap.csv + gaze_heatmaps.npz.

    Args:
        config: Gazelle inference configuration.

    Returns:
        Summary dict with row counts and settings.
    """
    import torch

    camera_dir = Path(config.camera_dir)
    session_config = load_session_config(config.session_config_path)
    mapping = session_config.get_camera_mapping(config.camera_id)

    # Load existing pipeline outputs
    pose_df = pd.read_csv(camera_dir / config.pose_input)
    tracks_df = pd.read_csv(camera_dir / config.tracks_input)

    # Extract head bounding boxes
    head_bboxes_df = extract_head_bboxes(
        pose_df,
        tracks_df,
        image_width=session_config.image_width,
        image_height=session_config.image_height,
    )

    # Find available frames on disk
    frame_paths = _find_frame_paths(camera_dir)
    available_frames = set(frame_paths.keys())
    head_bboxes_df = head_bboxes_df[
        head_bboxes_df["frame_idx"].isin(available_frames)
    ].copy()

    # Apply time-range filter
    t_start, t_end = resolve_time_range(
        session_config, config.start_s, config.end_s, config.time_block
    )
    head_bboxes_df = filter_by_time_range(head_bboxes_df, t_start, t_end)

    if config.max_frames > 0:
        unique_frames = sorted(head_bboxes_df["frame_idx"].unique())[:config.max_frames]
        head_bboxes_df = head_bboxes_df[
            head_bboxes_df["frame_idx"].isin(unique_frames)
        ].copy()

    # Resolve device and load model
    device = _resolve_device(config.device)
    print(f"Loading Gazelle model '{config.model_name}' on {device}...")
    model, transform = _load_gazelle_model(config.model_name, device)

    # Group head bboxes by frame for batched processing
    frame_groups = head_bboxes_df.groupby("frame_idx", sort=True)
    frame_indices = sorted(frame_groups.groups.keys())

    result_rows: List[dict] = []
    all_heatmaps: List[np.ndarray] = []
    heatmap_keys: List[str] = []

    batch_size = max(1, config.batch_size)
    total_frames = len(frame_indices)
    print(f"Running Gazelle on {total_frames} frames (batch_size={batch_size})...")

    for batch_start in range(0, total_frames, batch_size):
        batch_frame_idxs = frame_indices[batch_start : batch_start + batch_size]

        # Build batch tensors
        batch_images = []
        batch_bboxes: List[List[Tuple[float, float, float, float]]] = []
        batch_meta: List[List[dict]] = []

        for fidx in batch_frame_idxs:
            frame_path = frame_paths[fidx]
            img_tensor = _load_frame_image(frame_path, transform)
            batch_images.append(img_tensor)

            frame_bboxes = frame_groups.get_group(fidx)
            bboxes_for_frame: List[Tuple[float, float, float, float]] = []
            meta_for_frame: List[dict] = []
            for _, row in frame_bboxes.iterrows():
                bboxes_for_frame.append(
                    (row["head_x1"], row["head_y1"], row["head_x2"], row["head_y2"])
                )
                meta_for_frame.append(
                    {
                        "frame_idx": int(row["frame_idx"]),
                        "timestamp_s": float(row["timestamp_s"]),
                        "track_id": int(row["track_id"]),
                        "head_source": row["head_source"],
                    }
                )
            batch_bboxes.append(bboxes_for_frame)
            batch_meta.append(meta_for_frame)

        images_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():
            output = model({"images": images_tensor, "bboxes": batch_bboxes})

        heatmaps = output["heatmap"]  # list of [N_people, 64, 64] tensors
        inout_scores = output.get("inout")  # list of [N_people] tensors or None

        for img_idx, meta_list in enumerate(batch_meta):
            hm = heatmaps[img_idx].cpu().numpy()  # [N_people, 64, 64]
            inout = (
                inout_scores[img_idx].cpu().numpy()
                if inout_scores is not None
                else np.zeros(len(meta_list))
            )

            for person_idx, meta in enumerate(meta_list):
                heatmap_2d = hm[person_idx]  # [64, 64]
                peak_idx = np.unravel_index(np.argmax(heatmap_2d), heatmap_2d.shape)
                peak_y, peak_x = peak_idx
                peak_value = float(heatmap_2d[peak_y, peak_x])

                # Normalize peak to [0, 1]
                gaze_peak_x = float(peak_x) / 63.0
                gaze_peak_y = float(peak_y) / 63.0

                result_rows.append(
                    {
                        "frame_idx": meta["frame_idx"],
                        "timestamp_s": meta["timestamp_s"],
                        "track_id": meta["track_id"],
                        "gaze_peak_x": gaze_peak_x,
                        "gaze_peak_y": gaze_peak_y,
                        "gaze_peak_value": peak_value,
                        "inout_score": float(inout[person_idx]),
                        "head_source": meta["head_source"],
                    }
                )

                # Store raw heatmap for synchrony analysis
                key = f"f{meta['frame_idx']:06d}_t{meta['track_id']}"
                all_heatmaps.append(heatmap_2d)
                heatmap_keys.append(key)

        if (batch_start // batch_size + 1) % 10 == 0 or batch_start + batch_size >= total_frames:
            processed = min(batch_start + batch_size, total_frames)
            print(f"  Processed {processed}/{total_frames} frames")

    # Write outputs
    gaze_df = pd.DataFrame(result_rows, columns=GAZE_HEATMAP_COLUMNS)
    gaze_csv_path = camera_dir / config.output_csv
    gaze_df.to_csv(gaze_csv_path, index=False)

    # Save raw heatmaps as compressed numpy archive
    if all_heatmaps:
        heatmap_array = np.stack(all_heatmaps)  # [N, 64, 64]
        npz_path = camera_dir / config.output_npz
        np.savez_compressed(
            npz_path,
            heatmaps=heatmap_array,
            keys=np.array(heatmap_keys),
        )
        npz_size_mb = npz_path.stat().st_size / (1024 * 1024)
    else:
        npz_size_mb = 0.0

    summary = {
        "model_name": config.model_name,
        "device": device,
        "batch_size": config.batch_size,
        "total_frames": total_frames,
        "total_gaze_rows": len(gaze_df),
        "heatmap_npz_size_mb": round(npz_size_mb, 1),
        "output_csv": str(gaze_csv_path),
    }
    summary_path = camera_dir / "gaze_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(gaze_df)} gaze rows to {gaze_csv_path}")
    print(f"Wrote heatmaps ({npz_size_mb:.1f} MB) to {camera_dir / config.output_npz}")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for gaze-infer."""
    parser = argparse.ArgumentParser(
        prog="gaze-infer",
        description="Run Gazelle gaze estimation on video frames.",
    )
    parser.add_argument("--camera-dir", required=True, type=str)
    parser.add_argument("--session-config", required=True, type=str)
    parser.add_argument("--camera-id", default="camera_a", type=str)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument(
        "--model-name",
        default="gazelle_dinov2_vitl14_inout",
        type=str,
        choices=[
            "gazelle_dinov2_vitb14",
            "gazelle_dinov2_vitl14",
            "gazelle_dinov2_vitb14_inout",
            "gazelle_dinov2_vitl14_inout",
        ],
    )
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument(
        "--max-frames",
        default=0,
        type=int,
        help="Limit to first N frames (0 = all frames).",
    )
    parser.add_argument("--pose-input", default="pose_3d.csv", type=str)
    parser.add_argument("--tracks-input", default="tracks_2d.csv", type=str)
    parser.add_argument("--output-csv", default="gaze_heatmap.csv", type=str)
    parser.add_argument("--output-npz", default="gaze_heatmaps.npz", type=str)
    add_time_range_args(parser)
    return parser


def main() -> None:
    """CLI entry point for gaze-infer."""
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = GazelleConfig(
        camera_dir=args.camera_dir,
        session_config_path=args.session_config,
        camera_id=args.camera_id,
        device=args.device,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        start_s=args.start_s,
        end_s=args.end_s,
        time_block=args.time_block,
        pose_input=args.pose_input,
        tracks_input=args.tracks_input,
        output_csv=args.output_csv,
        output_npz=args.output_npz,
    )
    run_gazelle_inference(cfg)


if __name__ == "__main__":
    main()
