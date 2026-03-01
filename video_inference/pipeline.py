"""End-to-end video inference pipeline orchestration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .compress import CompressionResult, compress_video
from .frames import FrameExtractionResult, extract_frames_ffmpeg
from .sam3d_runner import RunnerConfig, run_sam3d_on_images
from .schema import ValidationResult, validate_session_output


@dataclass
class CameraPipelineSummary:
    """Per-camera pipeline summary."""

    camera_id: str
    source_video: str
    compressed_video: str
    frames_dir: str
    raw_output_json: str
    schema_valid: bool
    frame_count: int
    errors: List[str]


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""

    video_a: str
    video_b: Optional[str]
    output_dir: str
    checkpoint_path: str
    mhr_path: str
    session_id: Optional[str] = None
    device: str = "auto"
    inference_mode: str = "auto"
    detector_name: str = "sam3"
    bbox_thresh: float = 0.5
    max_images: Optional[int] = None
    use_mask: bool = False
    frame_rate: float = 15.0
    max_width: int = 1280
    crf: int = 23
    preset: str = "medium"
    ffmpeg_bin: str = "ffmpeg"
    reuse_existing: bool = False
    dry_run: bool = False


def _default_session_id() -> str:
    return datetime.now().strftime("session_%Y%m%d_%H%M%S")


def _camera_inputs(config: PipelineConfig) -> List[tuple[str, str]]:
    camera_list = [("camera_a", config.video_a)]
    if config.video_b:
        camera_list.append(("camera_b", config.video_b))
    return camera_list


def _payload_to_tracks_and_pose(
    payload: Dict[str, Any],
    frame_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    track_rows: List[Dict[str, Any]] = []
    pose_rows: List[Dict[str, Any]] = []

    for frame in payload.get("frames", []):
        frame_idx = int(frame["frame_idx"])
        timestamp_s = float(frame_idx / frame_rate)
        persons = frame.get("persons", [])

        for person in persons:
            bbox = person["bbox_xyxy"]
            track_id = int(person["track_id"])
            track_label = str(person["track_label"])
            confidence = float(person.get("confidence", 1.0))

            track_rows.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_s": timestamp_s,
                    "track_id": track_id,
                    "track_label": track_label,
                    "bbox_x1": float(bbox[0]),
                    "bbox_y1": float(bbox[1]),
                    "bbox_x2": float(bbox[2]),
                    "bbox_y2": float(bbox[3]),
                    "track_confidence": confidence,
                }
            )

            keypoints = person.get("keypoints_3d", [])
            for kp_idx, keypoint in enumerate(keypoints):
                if len(keypoint) < 3:
                    continue
                keypoint_confidence = (
                    float(keypoint[3]) if len(keypoint) >= 4 else confidence
                )
                pose_rows.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_s": timestamp_s,
                        "track_id": track_id,
                        "track_label": track_label,
                        "keypoint_name": f"kp_{kp_idx:03d}",
                        "x_m": float(keypoint[0]),
                        "y_m": float(keypoint[1]),
                        "z_m": float(keypoint[2]),
                        "keypoint_confidence": keypoint_confidence,
                    }
                )

    tracks_df = pd.DataFrame(track_rows)
    pose_df = pd.DataFrame(pose_rows)
    return tracks_df, pose_df


def _write_schema_outputs(
    payload: Dict[str, Any],
    camera_output_dir: Path,
    camera_id: str,
    source_video: str,
    frame_rate: float,
) -> ValidationResult:
    tracks_df, pose_df = _payload_to_tracks_and_pose(payload, frame_rate=frame_rate)

    tracks_path = camera_output_dir / "tracks_2d.csv"
    pose_path = camera_output_dir / "pose_3d.csv"
    manifest_path = camera_output_dir / "manifest.json"

    tracks_df.to_csv(tracks_path, index=False)
    pose_df.to_csv(pose_path, index=False)

    manifest = {
        "schema_version": "0.1.0",
        "session_id": camera_output_dir.name,
        "source_videos": [
            {
                "camera_id": camera_id,
                "relative_path": source_video,
                "fps": frame_rate,
            }
        ],
        "assumptions": {
            "max_persons": 2,
            "id_policy": "parent_larger_child_smaller_with_temporal_consistency",
        },
        "outputs": {
            "tracks_2d": tracks_path.name,
            "pose_3d": pose_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return validate_session_output(camera_output_dir)


def run_camera_pipeline(
    camera_id: str,
    source_video: str,
    session_dir: Path,
    config: PipelineConfig,
    compress_fn: Callable[..., CompressionResult] = compress_video,
    extract_fn: Callable[..., FrameExtractionResult] = extract_frames_ffmpeg,
    infer_fn: Callable[[RunnerConfig], Dict[str, Any]] = run_sam3d_on_images,
) -> CameraPipelineSummary:
    camera_dir = session_dir / camera_id
    intermediate_dir = camera_dir / "intermediate"
    frames_dir = camera_dir / "frames"
    compressed_video_path = intermediate_dir / "compressed.mp4"
    raw_output_json = intermediate_dir / "sam3d_raw.json"

    camera_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    if config.reuse_existing and compressed_video_path.exists():
        compression_result = CompressionResult(
            input_path=Path(source_video),
            output_path=compressed_video_path,
            command=[],
            executed=False,
        )
    else:
        compression_result = compress_fn(
            input_path=source_video,
            output_path=compressed_video_path,
            target_fps=config.frame_rate,
            max_width=config.max_width,
            crf=config.crf,
            preset=config.preset,
            ffmpeg_bin=config.ffmpeg_bin,
            overwrite=not config.reuse_existing,
            dry_run=config.dry_run,
        )

    if config.reuse_existing and (frames_dir / "frame_index.csv").exists():
        existing_frames = sorted(frames_dir.glob("frame_*.jpg"))
        extraction_result = FrameExtractionResult(
            video_path=compression_result.output_path,
            frames_dir=frames_dir,
            frame_rate=config.frame_rate,
            frame_paths=existing_frames,
            command=[],
            executed=False,
        )
    else:
        extraction_result = extract_fn(
            video_path=compression_result.output_path,
            frames_dir=frames_dir,
            frame_rate=config.frame_rate,
            ffmpeg_bin=config.ffmpeg_bin,
            overwrite=not config.reuse_existing,
            dry_run=config.dry_run,
        )

    runner_config = RunnerConfig(
        checkpoint_path=config.checkpoint_path,
        mhr_path=config.mhr_path,
        image_folder=str(extraction_result.frames_dir),
        output_json=str(raw_output_json),
        device=config.device,
        inference_mode=config.inference_mode,
        detector_name=config.detector_name,
        bbox_thresh=config.bbox_thresh,
        max_images=config.max_images,
        use_mask=config.use_mask,
    )

    if config.reuse_existing and raw_output_json.exists():
        payload = json.loads(raw_output_json.read_text(encoding="utf-8"))
    else:
        payload = infer_fn(runner_config)

    validation = _write_schema_outputs(
        payload=payload,
        camera_output_dir=camera_dir,
        camera_id=camera_id,
        source_video=source_video,
        frame_rate=extraction_result.frame_rate,
    )

    return CameraPipelineSummary(
        camera_id=camera_id,
        source_video=source_video,
        compressed_video=str(compression_result.output_path),
        frames_dir=str(extraction_result.frames_dir),
        raw_output_json=str(raw_output_json),
        schema_valid=validation.is_valid,
        frame_count=len(payload.get("frames", [])),
        errors=validation.errors,
    )


def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    session_id = config.session_id or _default_session_id()
    session_dir = Path(config.output_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    camera_summaries = []
    for camera_id, video_path in _camera_inputs(config):
        summary = run_camera_pipeline(
            camera_id=camera_id,
            source_video=video_path,
            session_dir=session_dir,
            config=config,
        )
        camera_summaries.append(summary)

    overall_valid = all(summary.schema_valid for summary in camera_summaries)
    summary_payload = {
        "session_id": session_id,
        "output_dir": str(session_dir),
        "overall_valid": overall_valid,
        "config": asdict(config),
        "cameras": [asdict(summary) for summary in camera_summaries],
    }
    (session_dir / "session_summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    return summary_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-infer",
        description="Video inference pipeline (compress -> frames -> sam3d -> export).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run full video inference pipeline.")
    run_parser.add_argument("--video-a", required=True, type=str)
    run_parser.add_argument("--video-b", default=None, type=str)
    run_parser.add_argument("--output-dir", default="video_inference/output", type=str)
    run_parser.add_argument("--session-id", default=None, type=str)
    run_parser.add_argument("--checkpoint-path", required=True, type=str)
    run_parser.add_argument("--mhr-path", required=True, type=str)
    run_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    run_parser.add_argument(
        "--inference-mode",
        default="auto",
        choices=["auto", "full", "body"],
    )
    run_parser.add_argument("--detector-name", default="sam3", type=str)
    run_parser.add_argument("--bbox-thresh", default=0.5, type=float)
    run_parser.add_argument("--max-images", default=None, type=int)
    run_parser.add_argument("--use-mask", action="store_true", default=False)
    run_parser.add_argument("--frame-rate", default=15.0, type=float)
    run_parser.add_argument("--max-width", default=1280, type=int)
    run_parser.add_argument("--crf", default=23, type=int)
    run_parser.add_argument("--preset", default="medium", type=str)
    run_parser.add_argument("--ffmpeg-bin", default="ffmpeg", type=str)
    run_parser.add_argument("--reuse-existing", action="store_true", default=False)
    run_parser.add_argument("--dry-run", action="store_true", default=False)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "run":
        cfg = PipelineConfig(
            video_a=args.video_a,
            video_b=args.video_b,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint_path,
            mhr_path=args.mhr_path,
            session_id=args.session_id,
            device=args.device,
            inference_mode=args.inference_mode,
            detector_name=args.detector_name,
            bbox_thresh=args.bbox_thresh,
            max_images=args.max_images,
            use_mask=args.use_mask,
            frame_rate=args.frame_rate,
            max_width=args.max_width,
            crf=args.crf,
            preset=args.preset,
            ffmpeg_bin=args.ffmpeg_bin,
            reuse_existing=args.reuse_existing,
            dry_run=args.dry_run,
        )
        run_pipeline(cfg)
