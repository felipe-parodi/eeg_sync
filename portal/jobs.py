"""Job planning and execution for the local processing portal."""

from __future__ import annotations

import json
import os
import shutil
import smtplib
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Iterable, Sequence

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4"}
CHUNK_CAMERA_IDS = {"camera_a", "camera_b"}


@dataclass
class SessionBlock:
    """A named analysis window in video time."""

    name: str
    start_s: float
    end_s: float
    color: str = "gray"


@dataclass
class ExclusionWindow:
    """A time window to exclude because an extra person is visible."""

    start_s: float
    end_s: float
    reason: str = ""


@dataclass
class CompressionSettings:
    """Compression settings used before inference."""

    target_fps: float = 8.0
    max_width: int = 854
    quality: int = 34
    cpu_preset: str = "ultrafast"
    size_threshold_mb: float = 50.0
    ffmpeg_bin: str = "ffmpeg"
    encoder_order: tuple[str, ...] = ("h264_nvenc",)


@dataclass
class CommandStep:
    """One subprocess command in a portal job."""

    name: str
    argv: list[str]


@dataclass
class PortalJobConfig:
    """Configuration for a submitted portal processing job."""

    job_id: str
    session_id: str
    email: str
    video_a: Path | None
    video_b: Path | None
    job_dir: Path
    output_root: Path
    session_config_path: Path
    blocks: list[SessionBlock]
    blocks_by_camera: dict[str, list[SessionBlock]] = field(default_factory=dict)
    include_gaze: bool = False
    frame_rate: float = 8.0
    image_width: int = 854
    image_height: int = 480
    max_persons: int = 4
    device: str = "auto"
    model_path: str = "yolo11m-pose.pt"
    overlay_seconds: float = 120.0

    @property
    def session_output_dir(self) -> Path:
        """Return the pipeline output directory for this session."""
        return self.output_root / self.session_id


def parse_timecode(value: str) -> float:
    """Parse seconds, MM:SS, or HH:MM:SS into seconds.

    Args:
        value: User-entered time string.

    Returns:
        Time in seconds.

    Raises:
        ValueError: If the value cannot be parsed.
    """
    text = value.strip()
    if not text:
        raise ValueError("time value cannot be empty")

    if ":" not in text:
        return float(text)

    parts = text.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    raise ValueError(f"cannot parse time value: {value!r}")


def format_seconds_for_cli(seconds: float) -> str:
    """Format seconds as a MM:SS string accepted by existing CLIs.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string.
    """
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    whole_minutes = int(seconds // 60)
    sec = seconds - whole_minutes * 60
    if abs(sec - round(sec)) < 1e-6:
        return f"{whole_minutes}:{int(round(sec)):02d}"
    return f"{whole_minutes}:{sec:05.2f}"


def split_blocks_for_exclusions(
    blocks: Sequence[SessionBlock],
    exclusions: Sequence[ExclusionWindow],
    min_duration_s: float = 0.1,
) -> list[SessionBlock]:
    """Remove exclusion windows from analysis blocks.

    Args:
        blocks: Original analysis blocks.
        exclusions: Windows where extra people are visible.
        min_duration_s: Minimum fragment duration to keep.

    Returns:
        New list of blocks with overlapping exclusion spans removed.
    """
    result: list[SessionBlock] = []
    sorted_exclusions = sorted(exclusions, key=lambda item: item.start_s)

    for block in blocks:
        fragments = [(block.start_s, block.end_s)]
        for exclusion in sorted_exclusions:
            next_fragments: list[tuple[float, float]] = []
            for start_s, end_s in fragments:
                overlap_start = max(start_s, exclusion.start_s)
                overlap_end = min(end_s, exclusion.end_s)
                if overlap_start >= overlap_end:
                    next_fragments.append((start_s, end_s))
                    continue
                if overlap_start - start_s >= min_duration_s:
                    next_fragments.append((start_s, overlap_start))
                if end_s - overlap_end >= min_duration_s:
                    next_fragments.append((overlap_end, end_s))
            fragments = next_fragments

        if len(fragments) == 1:
            start_s, end_s = fragments[0]
            result.append(
                SessionBlock(
                    name=block.name,
                    start_s=start_s,
                    end_s=end_s,
                    color=block.color,
                )
            )
            continue

        for index, (start_s, end_s) in enumerate(fragments, start=1):
            result.append(
                SessionBlock(
                    name=f"{block.name}_part{index}",
                    start_s=start_s,
                    end_s=end_s,
                    color=block.color,
                )
            )

    return result


def build_blocks_arg(blocks: Sequence[SessionBlock]) -> str:
    """Build the semicolon-delimited block string for track filtering.

    Args:
        blocks: Analysis blocks after exclusions are applied.

    Returns:
        CLI block specification string.
    """
    return ";".join(
        f"{block.name},{format_seconds_for_cli(block.start_s)},{format_seconds_for_cli(block.end_s)}"
        for block in blocks
    )


def active_camera_ids(config: PortalJobConfig) -> list[str]:
    """Return camera IDs that have an uploaded video.

    Args:
        config: Portal job configuration.

    Returns:
        Camera IDs in stable A/B order.

    Raises:
        ValueError: If no camera video is configured.
    """
    camera_ids = []
    if config.video_a is not None:
        camera_ids.append("camera_a")
    if config.video_b is not None:
        camera_ids.append("camera_b")
    if not camera_ids:
        raise ValueError("At least one GoPro video is required.")
    return camera_ids


def blocks_for_camera(config: PortalJobConfig, camera_id: str) -> list[SessionBlock]:
    """Return analysis blocks for one camera.

    Args:
        config: Portal job configuration.
        camera_id: Camera identifier.

    Returns:
        Camera-specific blocks, or the legacy shared block list.
    """
    camera_blocks = config.blocks_by_camera.get(camera_id)
    if camera_blocks is not None:
        return camera_blocks
    return config.blocks


def _all_config_blocks(config: PortalJobConfig) -> list[SessionBlock]:
    blocks: list[SessionBlock] = []
    seen = set()
    for camera_id in active_camera_ids(config):
        for block in blocks_for_camera(config, camera_id):
            key = (block.name, block.start_s, block.end_s, block.color)
            if key in seen:
                continue
            seen.add(key)
            blocks.append(block)
    return blocks


def validate_video_path(path: Path) -> None:
    """Validate one uploaded video path.

    Args:
        path: Path to validate.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If extension or size is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video extension for {path.name}. "
            f"Supported: {sorted(VIDEO_EXTENSIONS)}"
        )
    if path.stat().st_size <= 0:
        raise ValueError(f"Video file is empty: {path.name}")


def chunk_path_for(upload_dir: Path, camera_id: str, chunk_index: int) -> Path:
    """Return the canonical path for an uploaded file chunk.

    Args:
        upload_dir: Per-job upload directory.
        camera_id: Camera identifier, either ``camera_a`` or ``camera_b``.
        chunk_index: Zero-based chunk index.

    Returns:
        Path where the chunk should be stored.

    Raises:
        ValueError: If camera_id or chunk_index is invalid.
    """
    if camera_id not in CHUNK_CAMERA_IDS:
        raise ValueError("camera_id must be 'camera_a' or 'camera_b'")
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")
    return upload_dir / "chunks" / camera_id / f"chunk_{chunk_index:06d}.part"


def final_upload_path_for(
    upload_dir: Path, camera_id: str, original_filename: str
) -> Path:
    """Return the final uploaded video path for direct chunk writes.

    Args:
        upload_dir: Per-job upload directory.
        camera_id: Camera identifier, either ``camera_a`` or ``camera_b``.
        original_filename: Browser-provided filename used only for extension.

    Returns:
        Final video path for the camera.

    Raises:
        ValueError: If camera_id or extension is invalid.
    """
    if camera_id not in CHUNK_CAMERA_IDS:
        raise ValueError("camera_id must be 'camera_a' or 'camera_b'")
    suffix = Path(original_filename).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video extension for {original_filename}. "
            f"Supported: {sorted(VIDEO_EXTENSIONS)}"
        )
    return upload_dir / f"{camera_id}{suffix}"


def chunk_receipt_path_for(upload_dir: Path, camera_id: str, chunk_index: int) -> Path:
    """Return the marker path for a direct-written upload chunk."""
    if camera_id not in CHUNK_CAMERA_IDS:
        raise ValueError("camera_id must be 'camera_a' or 'camera_b'")
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")
    return upload_dir / "chunk_receipts" / camera_id / f"chunk_{chunk_index:06d}.json"


def mark_chunk_received(
    upload_dir: Path,
    camera_id: str,
    chunk_index: int,
    byte_count: int,
) -> Path:
    """Write a receipt marker after a chunk is written into the final upload.

    Args:
        upload_dir: Per-job upload directory.
        camera_id: Camera identifier, either ``camera_a`` or ``camera_b``.
        chunk_index: Zero-based chunk index.
        byte_count: Number of bytes written for the chunk.

    Returns:
        Receipt marker path.
    """
    receipt_path = chunk_receipt_path_for(upload_dir, camera_id, chunk_index)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps({"chunk_index": chunk_index, "byte_count": byte_count}, indent=2),
        encoding="utf-8",
    )
    return receipt_path


def validate_direct_chunked_upload(
    upload_dir: Path,
    camera_id: str,
    original_filename: str,
    total_chunks: int,
    expected_size_bytes: int,
) -> Path:
    """Validate a direct-written chunked upload and return its final video path.

    Args:
        upload_dir: Per-job upload directory.
        camera_id: Camera identifier, either ``camera_a`` or ``camera_b``.
        original_filename: Browser-provided filename used only for extension.
        total_chunks: Number of chunks expected from the browser.
        expected_size_bytes: Browser-reported final file size.

    Returns:
        Final assembled video path.

    Raises:
        FileNotFoundError: If any receipt or final file is missing.
        ValueError: If the final size does not match the browser metadata.
    """
    if total_chunks < 1:
        raise ValueError("total_chunks must be at least 1")
    output_path = final_upload_path_for(upload_dir, camera_id, original_filename)
    for index in range(total_chunks):
        receipt_path = chunk_receipt_path_for(upload_dir, camera_id, index)
        if not receipt_path.exists():
            raise FileNotFoundError(f"Missing upload chunk receipt: {receipt_path}")

    validate_video_path(output_path)
    actual_size = output_path.stat().st_size
    if actual_size != expected_size_bytes:
        raise ValueError(
            f"Uploaded {camera_id} size mismatch: expected "
            f"{expected_size_bytes} bytes, found {actual_size} bytes"
        )
    return output_path


def assemble_chunked_upload(
    upload_dir: Path,
    camera_id: str,
    original_filename: str,
    total_chunks: int,
) -> Path:
    """Assemble uploaded chunks into the final camera video file.

    Args:
        upload_dir: Per-job upload directory.
        camera_id: Camera identifier, either ``camera_a`` or ``camera_b``.
        original_filename: Browser-provided filename used only for extension.
        total_chunks: Number of chunks expected.

    Returns:
        Final assembled video path.

    Raises:
        FileNotFoundError: If any chunk is missing.
        ValueError: If the extension or chunk count is invalid.
    """
    if total_chunks < 1:
        raise ValueError("total_chunks must be at least 1")
    suffix = Path(original_filename).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video extension for {original_filename}. "
            f"Supported: {sorted(VIDEO_EXTENSIONS)}"
        )

    output_path = upload_dir / f"{camera_id}{suffix}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        for index in range(total_chunks):
            chunk_path = chunk_path_for(upload_dir, camera_id, index)
            if not chunk_path.exists():
                raise FileNotFoundError(f"Missing upload chunk: {chunk_path}")
            with chunk_path.open("rb") as chunk_file:
                shutil.copyfileobj(chunk_file, output_file)

    validate_video_path(output_path)
    return output_path


def write_session_config(config: PortalJobConfig) -> None:
    """Write the session_config.json consumed by downstream CLIs.

    Args:
        config: Portal job configuration.
    """
    camera_mappings = []
    for camera_id in active_camera_ids(config):
        camera_mappings.append(
            {"camera_id": camera_id, "parent_track_id": 0, "child_track_id": 1}
        )
    payload = {
        "session_id": config.session_id,
        "output_dir": str(config.session_output_dir),
        "image_width": config.image_width,
        "image_height": config.image_height,
        "frame_rate": config.frame_rate,
        "camera_mappings": camera_mappings,
        "session_blocks": [asdict(block) for block in _all_config_blocks(config)],
        "session_blocks_by_camera": {
            camera_id: [asdict(block) for block in blocks_for_camera(config, camera_id)]
            for camera_id in active_camera_ids(config)
        },
    }
    config.session_config_path.parent.mkdir(parents=True, exist_ok=True)
    config.session_config_path.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def build_ffmpeg_compression_command(
    input_path: Path,
    output_path: Path,
    settings: CompressionSettings,
    encoder: str,
) -> list[str]:
    """Build an ffmpeg compression command for one encoder.

    Args:
        input_path: Source video.
        output_path: Compressed destination path.
        settings: Compression settings.
        encoder: ffmpeg video encoder name.

    Returns:
        Command argument list.
    """
    vf = f"fps={settings.target_fps},scale='min({settings.max_width},iw)':-2"
    command = [
        settings.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    if encoder != "libx264":
        command.extend(["-hwaccel", "auto"])
    command.extend(["-i", str(input_path), "-vf", vf, "-c:v", encoder])

    quality = str(settings.quality)
    if encoder == "libx264":
        command.extend(["-preset", settings.cpu_preset, "-crf", quality])
    elif encoder == "h264_nvenc":
        command.extend(["-preset", "p1", "-cq", quality, "-b:v", "0"])
    elif encoder == "h264_qsv":
        command.extend(["-global_quality", quality])
    elif encoder == "h264_amf":
        command.extend(["-quality", "speed", "-qp_i", quality, "-qp_p", quality])
    else:
        command.extend(["-crf", quality])

    command.extend(["-an", str(output_path)])
    return command


def compress_video_if_needed(
    input_path: Path,
    output_path: Path,
    settings: CompressionSettings,
    force: bool = False,
) -> dict[str, Any]:
    """Compress one video when it exceeds the configured size threshold.

    Args:
        input_path: Source video.
        output_path: Destination for compressed video.
        settings: Compression settings.
        force: Compress even if the file is below the size threshold.

    Returns:
        Summary dictionary with selected output and command metadata.

    Raises:
        RuntimeError: If all encoder attempts fail.
    """
    validate_video_path(input_path)
    input_size_mb = input_path.stat().st_size / (1024 * 1024)
    should_compress = force or input_size_mb > settings.size_threshold_mb
    if not should_compress:
        return {
            "input_path": str(input_path),
            "output_path": str(input_path),
            "input_size_mb": input_size_mb,
            "output_size_mb": input_size_mb,
            "executed": False,
            "encoder": None,
            "command": [],
            "warning": None,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for encoder in settings.encoder_order:
        command = build_ffmpeg_compression_command(
            input_path=input_path,
            output_path=output_path,
            settings=settings,
            encoder=encoder,
        )
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as error:
            raise RuntimeError(
                f"'{settings.ffmpeg_bin}' not found. Install ffmpeg or set PORTAL_FFMPEG_BIN."
            ) from error
        except subprocess.CalledProcessError as error:
            stderr = error.stderr.strip() if error.stderr else "no stderr"
            errors.append(f"{encoder}: {stderr}")
            continue

        output_size_mb = output_path.stat().st_size / (1024 * 1024)
        warning = None
        if output_size_mb > settings.size_threshold_mb:
            warning = (
                f"Compressed file is {output_size_mb:.1f} MB, above the "
                f"{settings.size_threshold_mb:.1f} MB target."
            )
        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "input_size_mb": input_size_mb,
            "output_size_mb": output_size_mb,
            "executed": True,
            "encoder": encoder,
            "command": command,
            "warning": warning,
        }

    raise RuntimeError("All ffmpeg encoders failed: " + " | ".join(errors))


def build_processing_steps(config: PortalJobConfig) -> list[CommandStep]:
    """Build existing pipeline commands for a portal job.

    Args:
        config: Portal job configuration.

    Returns:
        Ordered command steps.
    """
    python = sys.executable
    session_dir = config.session_output_dir
    camera_ids = active_camera_ids(config)
    pose_argv = [
        python,
        "-m",
        "video_inference.pipeline",
        "run",
    ]
    if config.video_a is not None:
        pose_argv.extend(["--video-a", str(config.video_a)])
    if config.video_b is not None:
        pose_argv.extend(["--video-b", str(config.video_b)])
    pose_argv.extend(
        [
            "--inference-backend",
            "ultralytics",
            "--ultralytics-model-path",
            config.model_path,
            "--tracker-backend",
            "roboflow",
            "--tracker-name",
            "bytetrack",
            "--max-persons",
            str(config.max_persons),
            "--frame-rate",
            str(config.frame_rate),
        ]
    )
    for camera_id in camera_ids:
        camera_blocks = blocks_for_camera(config, camera_id)
        if not camera_blocks:
            raise ValueError(f"{camera_id} needs at least one analysis block.")
        flag = (
            "--analysis-windows-camera-a"
            if camera_id == "camera_a"
            else "--analysis-windows-camera-b"
        )
        pose_argv.extend([flag, build_blocks_arg(camera_blocks)])
    pose_argv.extend(
        [
            "--device",
            config.device,
            "--skip-compress",
            "--output-dir",
            str(config.output_root),
            "--session-id",
            config.session_id,
        ]
    )
    steps = [
        CommandStep(
            name="pose inference",
            argv=pose_argv,
        )
    ]

    for camera_id in camera_ids:
        camera_blocks = blocks_for_camera(config, camera_id)
        blocks_arg = build_blocks_arg(camera_blocks)
        camera_dir = session_dir / camera_id
        overlay_start = camera_blocks[0].start_s
        overlay_end = min(
            camera_blocks[0].end_s, overlay_start + config.overlay_seconds
        )

        steps.extend(
            [
                CommandStep(
                    name=f"filter tracks {camera_id}",
                    argv=[
                        python,
                        "-m",
                        "video_analysis.track_filter",
                        "--session-dir",
                        str(session_dir),
                        "--camera",
                        camera_id,
                        "--blocks",
                        blocks_arg,
                        "--source-fps",
                        str(config.frame_rate),
                        "--n-keep",
                        "2",
                    ],
                ),
                CommandStep(
                    name=f"smooth tracks {camera_id}",
                    argv=[
                        python,
                        "-m",
                        "video_analysis.temporal_smooth",
                        "--camera-dir",
                        str(camera_dir),
                        "--pose-input",
                        "pose_3d_filtered.csv",
                        "--tracks-input",
                        "tracks_2d_filtered.csv",
                        "--pose-output",
                        "pose_3d_smooth.csv",
                        "--tracks-output",
                        "tracks_2d_smooth.csv",
                    ],
                ),
                CommandStep(
                    name=f"pose metrics {camera_id}",
                    argv=[
                        python,
                        "-m",
                        "video_analysis.pose_metrics",
                        "--camera-dir",
                        str(camera_dir),
                        "--session-config",
                        str(config.session_config_path),
                        "--camera-id",
                        camera_id,
                        "--pose-input",
                        "pose_3d_smooth.csv",
                    ],
                ),
                CommandStep(
                    name=f"overlay video {camera_id}",
                    argv=[
                        python,
                        "-m",
                        "video_analysis.visualize_pose_tracks",
                        "--camera-dir",
                        str(camera_dir),
                        "--output-video",
                        str(camera_dir / "overlay_2min.mp4"),
                        "--pose-csv",
                        "pose_3d_smooth.csv",
                        "--tracks-csv",
                        "tracks_2d_smooth.csv",
                        "--start-time",
                        format_seconds_for_cli(overlay_start),
                        "--end-time",
                        format_seconds_for_cli(overlay_end),
                    ],
                ),
            ]
        )

        if config.include_gaze:
            steps.extend(
                [
                    CommandStep(
                        name=f"gaze inference {camera_id}",
                        argv=[
                            python,
                            "-m",
                            "gaze_analysis.gazelle_runner",
                            "--camera-dir",
                            str(camera_dir),
                            "--session-config",
                            str(config.session_config_path),
                            "--camera-id",
                            camera_id,
                            "--device",
                            config.device,
                            "--pose-input",
                            "pose_3d_smooth.csv",
                            "--tracks-input",
                            "tracks_2d_smooth.csv",
                        ],
                    ),
                    CommandStep(
                        name=f"gaze synchrony {camera_id}",
                        argv=[
                            python,
                            "-m",
                            "gaze_analysis.synchrony",
                            "--camera-dir",
                            str(camera_dir),
                            "--session-config",
                            str(config.session_config_path),
                            "--camera-id",
                            camera_id,
                            "--pose-input",
                            "pose_3d_smooth.csv",
                            "--tracks-input",
                            "tracks_2d_smooth.csv",
                        ],
                    ),
                    CommandStep(
                        name=f"gaze dashboard {camera_id}",
                        argv=[
                            python,
                            "-m",
                            "gaze_analysis.plotting",
                            "--camera-dir",
                            str(camera_dir),
                            "--session-config",
                            str(config.session_config_path),
                            "--camera-id",
                            camera_id,
                        ],
                    ),
                    CommandStep(
                        name=f"gaze metrics {camera_id}",
                        argv=[
                            python,
                            "-m",
                            "video_analysis.gaze_metrics",
                            "--camera-dir",
                            str(camera_dir),
                            "--session-config",
                            str(config.session_config_path),
                            "--camera-id",
                            camera_id,
                            "--pose-input",
                            "pose_3d_smooth.csv",
                            "--tracks-input",
                            "tracks_2d_smooth.csv",
                        ],
                    ),
                ]
            )

    return steps


def write_status(job_dir: Path, payload: dict[str, Any]) -> None:
    """Write the current status JSON for a job.

    Args:
        job_dir: Job working directory.
        payload: JSON-serializable status payload.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "status.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def read_status(job_dir: Path) -> dict[str, Any]:
    """Read a job status JSON file.

    Args:
        job_dir: Job working directory.

    Returns:
        Status payload, or a minimal missing status.
    """
    path = job_dir / "status.json"
    if not path.exists():
        return {"state": "missing", "message": "No status file found."}
    return json.loads(path.read_text(encoding="utf-8"))


def job_config_to_dict(config: PortalJobConfig) -> dict[str, Any]:
    """Convert a job config to a JSON-safe dictionary.

    Args:
        config: Portal job configuration.

    Returns:
        JSON-safe dictionary.
    """
    payload = asdict(config)
    for key in ("video_a", "video_b", "job_dir", "output_root", "session_config_path"):
        payload[key] = str(payload[key])
    return payload


def run_command_step(step: CommandStep, cwd: Path, log_path: Path) -> None:
    """Run one command step and append output to the job log.

    Args:
        step: Command step to run.
        cwd: Working directory.
        log_path: Log file path.
    """
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n\n== {step.name} ==\n")
        log_file.write(" ".join(step.argv) + "\n")
        log_file.flush()
        subprocess.run(
            step.argv,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )


def make_result_zip(
    session_output_dir: Path,
    zip_path: Path,
    extra_files: Iterable[tuple[Path, Path]] | None = None,
) -> None:
    """Create a zip archive of the session outputs.

    Args:
        session_output_dir: Directory to package.
        zip_path: Destination archive path.
        extra_files: Optional ``(source_path, archive_path)`` metadata files to include.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in session_output_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(session_output_dir.parent))
        for source_path, archive_path in extra_files or []:
            if source_path.exists():
                zf.write(source_path, archive_path)


def send_notification(
    to_address: str,
    subject: str,
    body: str,
    job_dir: Path,
) -> None:
    """Send an email notification or write a preview when SMTP is not configured.

    Args:
        to_address: Recipient email address.
        subject: Message subject.
        body: Message body.
        job_dir: Job directory for fallback preview output.
    """
    smtp_host = os.environ.get("PORTAL_SMTP_HOST")
    from_address = os.environ.get(
        "PORTAL_EMAIL_FROM", "video-processing-portal@localhost"
    )
    if not smtp_host:
        (job_dir / "email_preview.txt").write_text(
            f"To: {to_address}\nSubject: {subject}\n\n{body}\n",
            encoding="utf-8",
        )
        return

    message = EmailMessage()
    message["From"] = from_address
    message["To"] = to_address
    message["Subject"] = subject
    message.set_content(body)

    port = int(os.environ.get("PORTAL_SMTP_PORT", "587"))
    username = os.environ.get("PORTAL_SMTP_USER")
    password = os.environ.get("PORTAL_SMTP_PASSWORD")
    with smtplib.SMTP(smtp_host, port, timeout=30) as smtp:
        smtp.starttls()
        if username and password:
            smtp.login(username, password)
        smtp.send_message(message)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _iter_existing_artifacts(paths: Iterable[Path]) -> list[str]:
    return [str(path) for path in paths if path.exists()]


def run_portal_job(config: PortalJobConfig, repo_root: Path) -> None:
    """Run a submitted portal job end to end.

    Args:
        config: Job configuration.
        repo_root: Repository root used as subprocess working directory.
    """
    log_path = config.job_dir / "job.log"
    compression_settings = CompressionSettings(
        ffmpeg_bin=os.environ.get("PORTAL_FFMPEG_BIN", "ffmpeg")
    )
    try:
        write_status(
            config.job_dir,
            {
                "state": "running",
                "message": "Compressing uploaded videos if needed.",
                "current_step": "compression",
            },
        )

        compressed_dir = repo_root / "video_inference" / "compressed" / config.job_id
        compression_summaries = []
        for camera_name, source_path in (
            ("camera_a", config.video_a),
            ("camera_b", config.video_b),
        ):
            if source_path is None:
                continue
            summary = compress_video_if_needed(
                input_path=source_path,
                output_path=compressed_dir / f"{camera_name}_rapid.mp4",
                settings=compression_settings,
            )
            compression_summaries.append(summary)
            if camera_name == "camera_a":
                config.video_a = Path(summary["output_path"])
            else:
                config.video_b = Path(summary["output_path"])

        _write_json(
            config.job_dir / "rapid_compression_summary.json",
            {
                "config": asdict(compression_settings),
                "count": len(compression_summaries),
                "outputs": compression_summaries,
            },
        )
        write_session_config(config)
        _write_json(config.job_dir / "job_config.json", job_config_to_dict(config))

        steps = build_processing_steps(config)
        for index, step in enumerate(steps, start=1):
            write_status(
                config.job_dir,
                {
                    "state": "running",
                    "message": f"Running step {index} of {len(steps)}: {step.name}",
                    "current_step": step.name,
                },
            )
            run_command_step(step, cwd=repo_root, log_path=log_path)

        zip_path = config.job_dir / f"{config.session_id}_results.zip"
        metadata_root = Path(config.session_id) / "portal_metadata"
        make_result_zip(
            config.session_output_dir,
            zip_path,
            extra_files=[
                (
                    config.job_dir / "submission_metadata.json",
                    metadata_root / "submission_metadata.json",
                ),
                (config.job_dir / "job_config.json", metadata_root / "job_config.json"),
                (
                    config.job_dir / "rapid_compression_summary.json",
                    metadata_root / "rapid_compression_summary.json",
                ),
                (log_path, metadata_root / "job.log"),
            ],
        )
        artifacts = _iter_existing_artifacts(
            [
                zip_path,
                config.job_dir / "job.log",
                config.job_dir / "submission_metadata.json",
                config.job_dir / "job_config.json",
                config.job_dir / "rapid_compression_summary.json",
            ]
        )
        write_status(
            config.job_dir,
            {
                "state": "complete",
                "message": "Processing complete.",
                "current_step": None,
                "result_zip": str(zip_path),
                "artifacts": artifacts,
            },
        )
        send_notification(
            to_address=config.email,
            subject=f"Video processing portal results ready: {config.session_id}",
            body=(
                f"Results are ready for session {config.session_id}.\n\n"
                f"Result archive: {zip_path}\n"
                f"Job log: {log_path}\n"
            ),
            job_dir=config.job_dir,
        )
    except Exception as error:
        write_status(
            config.job_dir,
            {
                "state": "failed",
                "message": str(error),
                "current_step": None,
                "log_path": str(log_path),
            },
        )
        send_notification(
            to_address=config.email,
            subject=f"Video processing portal job failed: {config.session_id}",
            body=(
                f"Processing failed for session {config.session_id}.\n\n"
                f"Error: {error}\n"
                f"Job log: {log_path}\n"
            ),
            job_dir=config.job_dir,
        )
        raise


def copy_upload_to_path(source_path: Path, destination_path: Path) -> None:
    """Copy an uploaded temporary file into the portal workspace.

    Args:
        source_path: Temporary upload path.
        destination_path: Final destination.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination_path)
