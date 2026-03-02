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
from .ultralytics_runner import UltraRunnerConfig, run_ultralytics_pose_on_images


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
    checkpoint_path: str = ""
    mhr_path: str = ""
    session_id: Optional[str] = None
    device: str = "auto"
    inference_backend: str = "sam3d"
    inference_mode: str = "auto"
    detector_name: str = "sam3"
    ultralytics_model_path: str = "yolo11m-pose.pt"
    tracker_backend: str = "internal"
    tracker_name: str = "bytetrack"
    max_persons: int = 2
    enforce_exact_person_count: bool = False
    keep_empty_frames: bool = True
    person_class_id: int = 0
    bbox_thresh: float = 0.3
    imgsz: int = 640
    nms_iou_thresh: float = 0.45
    batch_size: int = 8
    max_images: Optional[int] = None
    use_mask: bool = False
    frame_rate: float = 5.0
    max_width: int = 1280
    crf: int = 23
    preset: str = "medium"
    ffmpeg_bin: str = "ffmpeg"
    reuse_existing: bool = False
    skip_compress: bool = False
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
    max_persons: int,
    enforce_exact_person_count: bool,
    id_policy: str,
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
            "max_persons": int(max_persons),
            "enforce_exact_person_count": bool(enforce_exact_person_count),
            "id_policy": id_policy,
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
    infer_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
) -> CameraPipelineSummary:
    camera_dir = session_dir / camera_id
    intermediate_dir = camera_dir / "intermediate"
    frames_dir = camera_dir / "frames"
    compressed_video_path = intermediate_dir / "compressed.mp4"
    raw_output_json = intermediate_dir / "inference_raw.json"

    camera_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    source_video_path = Path(source_video)
    if not source_video_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_video_path}")

    # Warn if the video is suspiciously large (likely uncompressed).
    _MAX_RECOMMENDED_MB = 50
    video_size_mb = source_video_path.stat().st_size / (1024 * 1024)
    if video_size_mb > _MAX_RECOMMENDED_MB and config.skip_compress:
        print(
            f"WARNING: {source_video_path.name} is {video_size_mb:.0f} MB "
            f"(recommended < {_MAX_RECOMMENDED_MB} MB). "
            "Running inference on large/uncompressed video will be very slow. "
            "Consider compressing first with video-compress-rapid."
        )

    if config.skip_compress:
        compression_result = CompressionResult(
            input_path=source_video_path,
            output_path=source_video_path,
            command=[],
            executed=False,
        )
    elif config.reuse_existing and compressed_video_path.exists():
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

    if config.inference_backend == "sam3d":
        if config.max_persons != 2:
            raise ValueError("sam3d backend currently supports max_persons=2 only")
        if config.tracker_backend != "internal":
            raise ValueError("sam3d backend currently supports tracker_backend=internal only")

        runner_config: Any = RunnerConfig(
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
        selected_infer_fn = infer_fn or run_sam3d_on_images
        id_policy = "parent_larger_child_smaller_with_temporal_consistency"
    elif config.inference_backend == "ultralytics":
        runner_config = UltraRunnerConfig(
            model_path=config.ultralytics_model_path,
            image_folder=str(extraction_result.frames_dir),
            output_json=str(raw_output_json),
            device=config.device,
            conf=config.bbox_thresh,
            imgsz=config.imgsz,
            nms_iou_thresh=config.nms_iou_thresh,
            batch_size=config.batch_size,
            max_images=config.max_images,
            tracker_backend=config.tracker_backend,
            tracker_name=config.tracker_name,
            max_persons=config.max_persons,
            enforce_exact_person_count=config.enforce_exact_person_count,
            keep_empty_frames=config.keep_empty_frames,
            person_class_id=config.person_class_id,
        )
        selected_infer_fn = infer_fn or run_ultralytics_pose_on_images
        if config.max_persons == 2 and config.tracker_backend == "internal":
            id_policy = "parent_larger_child_smaller_with_temporal_consistency"
        else:
            id_policy = f"{config.tracker_backend}_stable_ids_top_{config.max_persons}"
    else:
        raise ValueError(
            f"Unsupported inference_backend '{config.inference_backend}'. "
            "Expected 'sam3d' or 'ultralytics'."
        )

    if config.reuse_existing and raw_output_json.exists():
        payload = json.loads(raw_output_json.read_text(encoding="utf-8"))
    else:
        payload = selected_infer_fn(runner_config)

    validation = _write_schema_outputs(
        payload=payload,
        camera_output_dir=camera_dir,
        camera_id=camera_id,
        source_video=source_video,
        frame_rate=extraction_result.frame_rate,
        max_persons=config.max_persons,
        enforce_exact_person_count=config.enforce_exact_person_count,
        id_policy=id_policy,
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
        description="Video inference pipeline (compress -> frames -> backend -> export).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run full video inference pipeline.")
    run_parser.add_argument("--video-a", required=True, type=str)
    run_parser.add_argument("--video-b", default=None, type=str)
    run_parser.add_argument("--output-dir", default="video_inference/output", type=str)
    run_parser.add_argument("--session-id", default=None, type=str)
    run_parser.add_argument("--checkpoint-path", default="", type=str)
    run_parser.add_argument("--mhr-path", default="", type=str)
    run_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    run_parser.add_argument(
        "--inference-backend",
        default="sam3d",
        choices=["sam3d", "ultralytics"],
    )
    run_parser.add_argument(
        "--inference-mode",
        default="auto",
        choices=["auto", "full", "body"],
    )
    run_parser.add_argument("--detector-name", default="sam3", type=str)
    run_parser.add_argument(
        "--ultralytics-model-path",
        default="yolo11m-pose.pt",
        type=str,
    )
    run_parser.add_argument(
        "--tracker-backend",
        default="internal",
        choices=["internal", "roboflow"],
    )
    run_parser.add_argument("--tracker-name", default="bytetrack", type=str)
    run_parser.add_argument("--max-persons", default=2, type=int)
    run_parser.add_argument(
        "--enforce-exact-person-count",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run_parser.add_argument(
        "--keep-empty-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run_parser.add_argument("--person-class-id", default=0, type=int)
    run_parser.add_argument("--bbox-thresh", default=0.3, type=float)
    run_parser.add_argument("--imgsz", default=640, type=int)
    run_parser.add_argument("--nms-iou-thresh", default=0.45, type=float)
    run_parser.add_argument("--batch-size", default=8, type=int)
    run_parser.add_argument("--max-images", default=None, type=int)
    run_parser.add_argument("--use-mask", action="store_true", default=False)
    run_parser.add_argument("--frame-rate", default=5.0, type=float)
    run_parser.add_argument("--max-width", default=1280, type=int)
    run_parser.add_argument("--crf", default=23, type=int)
    run_parser.add_argument("--preset", default="medium", type=str)
    run_parser.add_argument("--ffmpeg-bin", default="ffmpeg", type=str)
    run_parser.add_argument("--reuse-existing", action="store_true", default=False)
    run_parser.add_argument("--skip-compress", action="store_true", default=False)
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
            inference_backend=args.inference_backend,
            inference_mode=args.inference_mode,
            detector_name=args.detector_name,
            ultralytics_model_path=args.ultralytics_model_path,
            tracker_backend=args.tracker_backend,
            tracker_name=args.tracker_name,
            max_persons=args.max_persons,
            enforce_exact_person_count=args.enforce_exact_person_count,
            keep_empty_frames=args.keep_empty_frames,
            person_class_id=args.person_class_id,
            bbox_thresh=args.bbox_thresh,
            imgsz=args.imgsz,
            nms_iou_thresh=args.nms_iou_thresh,
            batch_size=args.batch_size,
            max_images=args.max_images,
            use_mask=args.use_mask,
            frame_rate=args.frame_rate,
            max_width=args.max_width,
            crf=args.crf,
            preset=args.preset,
            ffmpeg_bin=args.ffmpeg_bin,
            reuse_existing=args.reuse_existing,
            skip_compress=args.skip_compress,
            dry_run=args.dry_run,
        )
        run_pipeline(cfg)


if __name__ == "__main__":
    main()
