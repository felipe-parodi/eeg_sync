"""Job planning and execution for the local processing portal."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4"}
CHUNK_CAMERA_IDS = {"camera_a", "camera_b"}
# 5 fps: every 2nd frame of the 10 fps portal-compressed videos, so frames are
# evenly spaced (6 fps from 10 fps alternated 100/200 ms steps, which put a
# 2 Hz artifact into every person's measured movement). The rate actually
# achieved for each video is read back after frame extraction.
PORTAL_DEFAULT_FRAME_RATE = 5.0
PORTAL_DEFAULT_IMAGE_WIDTH = 854
PORTAL_DEFAULT_IMAGE_HEIGHT = 480
# Two detections per frame: the parent and the child. Extra-person windows
# (e.g. an experimenter in the room) are excluded from the analysis blocks
# before inference, so a third tracked person is noise we do not want to keep.
PORTAL_DEFAULT_MAX_PERSONS = 2
# Shortest analysis part kept after extra-person windows are removed. Shorter
# pieces (e.g. a 4 s gap between two windows) hold too few frames for stable
# metrics, so they are dropped and listed on the job page instead.
MIN_PART_S = 10.0
# Domain-adapted pose model. ``yolo11l-pose-dense.pt`` is a yolo11l-pose
# fine-tuned on 3-teacher consensus pseudo-labels (yolo11x-pose + ViTPose-Huge +
# RTMPose, box-matched and majority-voted) sampled from the timing-windowed,
# 2-person-only sections of all 36 session videos -- same architecture and
# latency as stock yolo11l-pose, but better detection and keypoints on this
# footage (overhead, occlusion, young children). It supersedes the earlier
# ``yolo11l-pose-fused3.pt`` after winning a blind A/B on fresh (non-training)
# frames (dense 15, fused3 7, 9 ties of 31). Both are local artifacts
# (gitignored *.pt); we prefer dense, fall back to fused3, then to the stock
# model (downloaded on demand). See pose_training/ and AGENTS.md.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PORTAL_POSE_WEIGHTS = (
    _REPO_ROOT / "models" / "yolo11l-pose-dense.pt",
    _REPO_ROOT / "models" / "yolo11l-pose-fused3.pt",
)
PORTAL_DEFAULT_MODEL_PATH = next(
    (str(path) for path in _PORTAL_POSE_WEIGHTS if path.exists()),
    "yolo11l-pose.pt",
)
# Inference resolution. The fine-tuned students are trained at 960, but the
# pipeline default is 640 -- so without this the deployed model would run at a
# resolution it was not trained for. 960 matches training and measurably
# improves detection (at a modest GPU cost).
PORTAL_DEFAULT_INFER_IMGSZ = 960
# Length of each per-block QC overlay clip. One clip is rendered per analysis
# block (capped at this length), so the colleague sees tracking quality for
# every block rather than only the first.
PORTAL_DEFAULT_OVERLAY_SECONDS = 30.0
# Who research assistants contact when a job fails (``PORTAL_SUPPORT_CONTACT``).
DEFAULT_SUPPORT_CONTACT = "Felipe"
# Tracker used in phase 1. "offline" replaced BoT-SORT after a benchmark on 31
# camera views of past portal sessions (pose_training/bench.py): frames with
# both parent and child 0.52 -> 0.73, role accuracy on held-out anchor labels
# 0.78 -> 0.99. See docs/plans/pose-quality-overhaul_261009.md.
PORTAL_TRACKER_BACKEND = "offline"
PORTAL_TRACKER_NAME = "two_person_offline"
DETECTION_CACHE_RELPATH = Path("intermediate") / "detections.npz"


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

    # 10 fps divides evenly into the 5 fps analysis rate (8 fps could only
    # give an even 4 fps).
    target_fps: float = 10.0
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
    camera_id: str | None = None
    # GPU-bound steps never overlap across cameras: two at once only contend
    # (two ViTPose refinements took 110 s each in parallel vs 48 s alone).
    gpu_heavy: bool = False
    # A failing optional step (a QC check clip) is recorded as a note and the
    # job carries on; any other failing step stops the job.
    optional: bool = False


@dataclass
class PortalJobConfig:
    """Configuration for a submitted portal processing job."""

    job_id: str
    session_id: str
    video_a: Path | None
    video_b: Path | None
    job_dir: Path
    output_root: Path
    session_config_path: Path
    blocks: list[SessionBlock]
    blocks_by_camera: dict[str, list[SessionBlock]] = field(default_factory=dict)
    include_gaze: bool = False
    frame_rate: float = PORTAL_DEFAULT_FRAME_RATE
    image_width: int = PORTAL_DEFAULT_IMAGE_WIDTH
    image_height: int = PORTAL_DEFAULT_IMAGE_HEIGHT
    max_persons: int = PORTAL_DEFAULT_MAX_PERSONS
    device: str = "cuda"
    model_path: str = PORTAL_DEFAULT_MODEL_PATH
    infer_imgsz: int = PORTAL_DEFAULT_INFER_IMGSZ
    overlay_seconds: float = PORTAL_DEFAULT_OVERLAY_SECONDS
    refine_keypoints: bool = True
    # Gaze model: "page" (PaGE ViT-B; preferred over Gaze-LLE in a blind
    # side-by-side on lab footage and better on published child-gaze
    # benchmarks) or "gazelle" (Gaze-LLE ViT-L, the previous default).
    gaze_model: str = "page"
    # Process camera A and camera B at the same time (the GPU is mostly
    # idle during one camera's CPU-bound work): ~halves job wall time.
    parallel_cameras: bool = True

    @property
    def session_output_dir(self) -> Path:
        """Return the pipeline output directory for this session."""
        return self.output_root / self.session_id


_TIME_FORMS_HINT = "Use seconds (90), M:SS (1:30) or H:MM:SS (1:01:30)."
_SECONDS_RE = re.compile(r"^\d+(?:\.\d+)?$")
_CLOCK_RE = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")


def parse_timecode(value: str) -> float:
    """Parse seconds, M:SS, or H:MM:SS into seconds.

    Minutes and hours must be whole numbers; only the seconds field may have
    a decimal part. In H:MM:SS, minutes must be under 60; seconds must be
    under 60 whenever a colon is used. M:SS accepts any number of minutes
    (``75:30`` is 75 minutes 30 seconds).

    Args:
        value: User-entered time string.

    Returns:
        Time in seconds.

    Raises:
        ValueError: With a plain-language message when the value is empty,
            negative, not a number (``nan``, ``inf``, ``1e3``), malformed
            (``1.5:30``), or has 60 or more seconds/minutes in a field.
    """
    text = value.strip()
    if not text:
        raise ValueError("Time is empty.")
    if text.startswith("-") and (
        _SECONDS_RE.match(text[1:]) or _CLOCK_RE.match(text[1:])
    ):
        raise ValueError(
            f'"{text}" is negative. Times count from the start of the video.'
        )
    if _SECONDS_RE.match(text):
        return float(text)
    match = _CLOCK_RE.match(text)
    if not match:
        raise ValueError(f'"{text}" is not a time. {_TIME_FORMS_HINT}')
    hours_raw, minutes_raw, seconds_raw = match.groups()
    seconds = float(seconds_raw)
    minutes = int(minutes_raw)
    if seconds >= 60:
        raise ValueError(
            f'"{text}" has {seconds_raw.lstrip("0") or "0"} seconds. '
            "Seconds must be under 60."
        )
    if hours_raw is None:
        return minutes * 60 + seconds
    if minutes >= 60:
        raise ValueError(f'"{text}" has {minutes} minutes. Minutes must be under 60.')
    return int(hours_raw) * 3600 + minutes * 60 + seconds


def format_clock(seconds: float) -> str:
    """Format seconds as M:SS, or H:MM:SS from one hour on.

    Args:
        seconds: Non-negative time in seconds.

    Returns:
        Clock text such as ``"13:26"``, ``"13:06:00"`` or ``"1:05.5"`` (one
        decimal only when the time is not a whole second).
    """
    total = round(float(seconds), 1)
    whole = int(total)
    fraction = round(total - whole, 1)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    sec_text = f"{secs:02d}" if fraction == 0 else f"{secs + fraction:04.1f}"
    if hours:
        return f"{hours}:{minutes:02d}:{sec_text}"
    return f"{minutes}:{sec_text}"


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


def split_blocks_with_min_part(
    blocks: Sequence[SessionBlock],
    exclusions: Sequence[ExclusionWindow],
    min_part_s: float = MIN_PART_S,
) -> tuple[list[SessionBlock], list[SessionBlock]]:
    """Remove exclusion windows, then drop parts shorter than ``min_part_s``.

    Parts are numbered before short ones are dropped, so the kept names do
    not change when a neighbour is dropped (``grocery_part1`` and
    ``grocery_part3`` when ``grocery_part2`` is a 4 s sliver). The browser
    form mirrors this logic in ``portal/static/index.js`` for its preview.

    Args:
        blocks: Original analysis blocks.
        exclusions: Windows where extra people are visible.
        min_part_s: Shortest part to keep, in seconds.

    Returns:
        ``(kept, dropped)`` analysis blocks.
    """
    kept: list[SessionBlock] = []
    dropped: list[SessionBlock] = []
    # Any piece longer than zero is numbered, so the numbering never depends
    # on the minimum length.
    for part in split_blocks_for_exclusions(blocks, exclusions, min_duration_s=1e-9):
        if part.end_s - part.start_s >= min_part_s:
            kept.append(part)
        else:
            dropped.append(part)
    return kept, dropped


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


def achieved_session_frame_rate(config: PortalJobConfig) -> float:
    """Frame rate actually kept by frame extraction, shared by all cameras.

    Frame extraction keeps every k-th source frame, so the achieved rate is
    the source rate divided by k (5 fps from a 10 fps video, 4 fps from an
    8 fps one). Everything after extraction must use this rate.

    Args:
        config: Portal job configuration (after phase 1 has run).

    Returns:
        The achieved rate, or ``config.frame_rate`` when extraction did not
        record one (older runs).

    Raises:
        ValueError: If the cameras ended up at different rates.
    """
    from video_inference.frames import read_achieved_frame_rate

    rates = {
        camera_id: read_achieved_frame_rate(
            config.session_output_dir / camera_id / "frames", config.frame_rate
        )
        for camera_id in active_camera_ids(config)
    }
    distinct = sorted({round(rate, 6) for rate in rates.values()})
    if len(distinct) > 1:
        detail = ", ".join(
            f"Video {camera_id[-1].upper()} {rate:g}"
            for camera_id, rate in rates.items()
        )
        raise ValueError(
            "The two videos ended up at different frames per second "
            f"({detail}). Compress both videos with the portal compressor "
            "(10 frames per second) and submit the session again."
        )
    return float(distinct[0]) if distinct else config.frame_rate


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


def probe_video_path(path: Path, ffprobe_bin: str = "ffprobe") -> None:
    """Validate that ffprobe can parse at least one video stream.

    Args:
        path: Video path to probe.
        ffprobe_bin: ffprobe executable name or path.

    Raises:
        RuntimeError: If ffprobe is unavailable or cannot parse the video.
        ValueError: If the media has no video stream.
    """
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"'{ffprobe_bin}' not found. Install ffmpeg/ffprobe or set PORTAL_FFPROBE_BIN."
        ) from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip() if error.stderr else "no stderr"
        raise RuntimeError(f"ffprobe could not parse {path.name}: {stderr}") from error

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as error:
        raise RuntimeError(f"ffprobe returned invalid JSON for {path.name}") from error
    if not payload.get("streams"):
        raise ValueError(f"No video stream found in uploaded file: {path.name}")


def probe_video_duration(path: Path, ffprobe_bin: str = "ffprobe") -> float:
    """Return a video's length in seconds as reported by ffprobe.

    Args:
        path: Video path to probe.
        ffprobe_bin: ffprobe executable name or path.

    Returns:
        Duration in seconds.

    Raises:
        RuntimeError: If ffprobe is unavailable, fails, or reports no
            usable duration.
    """
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"'{ffprobe_bin}' not found. Install ffmpeg/ffprobe or set PORTAL_FFPROBE_BIN."
        ) from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip() if error.stderr else "no stderr"
        raise RuntimeError(f"ffprobe could not read {path.name}: {stderr}") from error
    try:
        duration = float(json.loads(result.stdout or "{}")["format"]["duration"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Could not read the length of {path.name}.") from error
    if not duration > 0 or duration == float("inf"):
        raise RuntimeError(f"Could not read the length of {path.name}.")
    return duration


def validate_video_path(
    path: Path,
    *,
    probe: bool = False,
    ffprobe_bin: str = "ffprobe",
) -> None:
    """Validate one uploaded video path.

    Args:
        path: Path to validate.
        probe: Run ffprobe to verify parseable video content.
        ffprobe_bin: ffprobe executable name or path when ``probe`` is true.

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
    if probe:
        probe_video_path(path, ffprobe_bin=ffprobe_bin)


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
    probe: bool = False,
    ffprobe_bin: str = "ffprobe",
) -> Path:
    """Validate a direct-written chunked upload and return its final video path.

    Args:
        upload_dir: Per-job upload directory.
        camera_id: Camera identifier, either ``camera_a`` or ``camera_b``.
        original_filename: Browser-provided filename used only for extension.
        total_chunks: Number of chunks expected from the browser.
        expected_size_bytes: Browser-reported final file size.
        probe: Run ffprobe to verify parseable video content.
        ffprobe_bin: ffprobe executable name or path when ``probe`` is true.

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

    validate_video_path(output_path, probe=probe, ffprobe_bin=ffprobe_bin)
    actual_size = output_path.stat().st_size
    if actual_size != expected_size_bytes:
        raise ValueError(
            f"Uploaded {camera_id} size mismatch: expected "
            f"{expected_size_bytes} bytes, found {actual_size} bytes"
        )
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


def build_processing_steps(
    config: PortalJobConfig,
    anchors_json_by_camera: dict[str, Path] | None = None,
) -> list[CommandStep]:
    """Build the full pipeline as a flat list of command steps.

    Returns phase 1 (pose inference) followed by phase 2 (filter, smooth,
    metrics, overlay, optional gaze). Kept as a thin wrapper so existing
    callers and tests still work; production uses the phase-split entry
    points (``build_phase1_steps`` + ``build_phase2_steps``) so the
    portal can pause between them for human annotation.
    """
    return build_phase1_steps(config) + build_phase2_steps(
        config, anchors_json_by_camera=anchors_json_by_camera
    )


def build_phase1_steps(config: PortalJobConfig) -> list[CommandStep]:
    """Build pre-annotation commands: pose inference + tracking only.

    Phase 1 ends with ``tracks_2d.csv`` and ``pose_3d.csv`` on disk per
    camera. The portal then extracts anchor frames from those tracks and
    waits for the colleague to label them before phase 2 runs.
    """
    python = sys.executable
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
            # Offline two-person association: caches every detection (down
            # to conf 0.05) with an appearance embedding, then solves parent
            # and child over the whole recording. Phase 2 re-solves it with
            # the colleague's anchor labels. See video_inference/offline_assoc.
            "--tracker-backend",
            PORTAL_TRACKER_BACKEND,
            "--max-persons",
            str(config.max_persons),
            "--frame-rate",
            str(config.frame_rate),
            "--imgsz",
            str(config.infer_imgsz),
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
    if config.parallel_cameras:
        pose_argv.append("--parallel-cameras")
    return [CommandStep(name="pose inference", argv=pose_argv)]


GAZE_RUNNER_MODULES = {
    "page": "gaze_analysis.page_runner",
    "gazelle": "gaze_analysis.gazelle_runner",
}


def build_phase2_steps(
    config: PortalJobConfig,
    anchors_json_by_camera: dict[str, Path] | None = None,
) -> list[CommandStep]:
    """Build post-annotation commands: filter -> smooth -> metrics -> overlay.

    Args:
        config: Portal job configuration.
        anchors_json_by_camera: Optional ``{camera_id: anchors.json path}``.
            When provided for a camera, the filter step gets
            ``--anchors-json <path>`` so role assignment uses user labels
            instead of size heuristics.
    """
    python = sys.executable
    session_dir = config.session_output_dir
    camera_ids = active_camera_ids(config)
    anchors_map = anchors_json_by_camera or {}
    steps: list[CommandStep] = []
    camera_starts: list[tuple[str, int]] = []

    for camera_id in camera_ids:
        camera_blocks = blocks_for_camera(config, camera_id)
        blocks_arg = build_blocks_arg(camera_blocks)
        camera_dir = session_dir / camera_id
        camera_starts.append((camera_id, len(steps)))

        offline = (camera_dir / DETECTION_CACHE_RELPATH).exists()
        refine = offline and config.refine_keypoints
        pose_after_roles = "pose_3d_refined.csv" if refine else "pose_3d_filtered.csv"
        if offline:
            # Re-solve parent/child over the whole recording with the anchors.
            filter_argv = [
                python,
                "-m",
                "video_inference.offline_assoc",
                "--camera-dir",
                str(camera_dir),
            ]
        else:
            # Jobs from before the offline tracker: block filter on raw tracks.
            filter_argv = [
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
            ]
        anchors_path = anchors_map.get(camera_id)
        if anchors_path is not None:
            filter_argv.extend(["--anchors-json", str(anchors_path)])

        steps.extend(
            [
                CommandStep(
                    name=f"filter tracks {camera_id}",
                    argv=filter_argv,
                ),
            ]
        )
        if refine:
            # Re-estimate each person's keypoints with ViTPose-Large from their
            # box (~24% less keypoint jitter on past sessions); falls back
            # to the original keypoints where ViTPose disagrees on who it is.
            steps.append(
                CommandStep(
                    gpu_heavy=True,
                    name=f"refine keypoints {camera_id}",
                    argv=[
                        python,
                        "-m",
                        "video_inference.topdown_refine",
                        "--camera-dir",
                        str(camera_dir),
                        "--device",
                        config.device,
                    ],
                )
            )
        steps.extend(
            [
                # Fill short detection gaps before smoothing: reindex each
                # track onto a regular frame grid and linearly interpolate the
                # bbox and pose across gaps. The interpolator caps the fill at
                # 3x the median frame step, so brief drop-outs (the flicker)
                # are bridged but long occlusions/exits stay empty. Smoothing
                # then runs on the gap-filled series.
                CommandStep(
                    name=f"interpolate tracks {camera_id}",
                    argv=[
                        python,
                        "-m",
                        "video_analysis.interpolate",
                        "--camera-dir",
                        str(camera_dir),
                        "--target-fps",
                        str(config.frame_rate),
                        "--tracks-input",
                        "tracks_2d_filtered.csv",
                        "--pose-input",
                        pose_after_roles,
                        "--tracks-output",
                        "tracks_2d_interpolated.csv",
                        "--pose-output",
                        "pose_3d_interpolated.csv",
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
                        "pose_3d_interpolated.csv",
                        "--tracks-input",
                        "tracks_2d_interpolated.csv",
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
            ]
        )

        # Per analysis block, render a clip of the first config.overlay_seconds
        # and a clip of the last config.overlay_seconds, so the colleague can
        # eyeball tracking quality at both the start and the end of every block
        # (free_play, clap-synchrony, storybook, ...). For a block shorter than
        # one window the two clips would be identical, so only the first
        # (full-block) clip is rendered.
        for block in camera_blocks:
            window = config.overlay_seconds
            overlay_windows = [
                (
                    f"overlay_{block.name}.mp4",
                    "",
                    block.start_s,
                    min(block.end_s, block.start_s + window),
                )
            ]
            if block.end_s - block.start_s > window:
                overlay_windows.append(
                    (
                        f"overlay_{block.name}_end.mp4",
                        " (end)",
                        block.end_s - window,
                        block.end_s,
                    )
                )
            for out_name, name_suffix, win_start, win_end in overlay_windows:
                steps.append(
                    CommandStep(
                        optional=True,
                        name=f"overlay video {camera_id} {block.name}{name_suffix}",
                        argv=[
                            python,
                            "-m",
                            "video_analysis.visualize_pose_tracks",
                            "--camera-dir",
                            str(camera_dir),
                            "--output-video",
                            str(camera_dir / out_name),
                            "--pose-csv",
                            "pose_3d_smooth.csv",
                            "--tracks-csv",
                            "tracks_2d_smooth.csv",
                            "--start-time",
                            format_seconds_for_cli(win_start),
                            "--end-time",
                            format_seconds_for_cli(win_end),
                        ],
                    )
                )

        if config.include_gaze:
            steps.extend(
                [
                    CommandStep(
                        gpu_heavy=True,
                        name=f"gaze inference {camera_id}",
                        argv=[
                            python,
                            "-m",
                            GAZE_RUNNER_MODULES[config.gaze_model],
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

    bounds = camera_starts + [(None, len(steps))]
    for (camera_id, start), (_, end) in zip(bounds, bounds[1:]):
        for step in steps[start:end]:
            step.camera_id = camera_id
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Job status must be a JSON object: {path}")
    return payload


# Stages shown on the job page, in order: (key, label).
JOB_STAGES: tuple[tuple[str, str], ...] = (
    ("upload", "Upload"),
    ("prepare", "Prepare video"),
    ("find_people", "Find people"),
    ("label_parent", "Label parent (you)"),
    ("track_refine", "Track and refine"),
    ("measure", "Measure"),
    ("report", "Report"),
)
JOB_STAGE_KEYS = tuple(key for key, _ in JOB_STAGES)

# The one table that maps internal step names to a stage and plain words.
# Matched by prefix against ``current_step`` / ``CommandStep.name`` (the
# camera id and block name that follow the prefix only name the video).
# Each row: (step-name prefix, stage key, what is happening, failure phrase).
STEP_STAGE_TABLE: tuple[tuple[str, str, str, str], ...] = (
    ("uploading", "upload", "Receiving the videos.", "receiving the videos"),
    ("queued", "upload", "Waiting for another job to finish.", "waiting its turn"),
    (
        "finalize uploads",
        "upload",
        "Putting the uploaded videos together.",
        "putting the uploaded videos together",
    ),
    (
        "compression",
        "prepare",
        "Making the videos smaller for processing.",
        "preparing the videos",
    ),
    (
        "pose inference",
        "find_people",
        "Finding the people in every frame. This is the longest step.",
        "finding the people in the videos",
    ),
    (
        "extract anchors",
        "find_people",
        "Picking a few frames for you to label.",
        "picking frames for you to label",
    ),
    (
        "awaiting annotation",
        "label_parent",
        "Waiting for you to mark the parent in a few frames.",
        "waiting for your labels",
    ),
    (
        "phase 2 queued",
        "track_refine",
        "Labels received. Waiting to continue.",
        "starting after your labels",
    ),
    (
        "filter tracks",
        "track_refine",
        "Following the parent and the child through the video.",
        "following the parent and the child",
    ),
    (
        "refine keypoints",
        "track_refine",
        "Refining body positions.",
        "refining body positions",
    ),
    (
        "interpolate tracks",
        "track_refine",
        "Filling short gaps in tracking.",
        "filling short gaps in tracking",
    ),
    ("smooth tracks", "track_refine", "Smoothing movement.", "smoothing movement"),
    ("pose metrics", "measure", "Measuring movement.", "measuring movement"),
    (
        "overlay video",
        "measure",
        "Making short check clips.",
        "making the check clips",
    ),
    ("gaze", "measure", "Estimating where each person looks.", "estimating gaze"),
    (
        "session report",
        "report",
        "Writing the session report.",
        "writing the session report",
    ),
    ("results zip", "report", "Packing the results.", "packing the results"),
)

CAMERA_LABELS = {"camera_a": "Video A", "camera_b": "Video B"}


def describe_step(step_name: str | None) -> dict[str, str]:
    """Translate an internal step name into a stage and plain words.

    Args:
        step_name: Internal step name such as ``"smooth tracks camera_a"``.

    Returns:
        Dict with ``stage`` (a ``JOB_STAGE_KEYS`` entry, or ``""`` when the
        step is unknown), ``message`` (what is happening, in plain words,
        naming the video when the step belongs to one camera) and
        ``failure`` (a phrase for "the job stopped while ...").
    """
    name = (step_name or "").strip()
    camera = next((cid for cid in CAMERA_LABELS if cid in name.split()), None)
    for prefix, stage, message, failure in STEP_STAGE_TABLE:
        if name == prefix or name.startswith(prefix + " "):
            if camera is not None:
                video = CAMERA_LABELS[camera]
                message = f"{video}: {message}"
                failure = f"{failure} for {video}"
            return {"stage": stage, "message": message, "failure": failure}
    return {"stage": "", "message": "Working.", "failure": "processing"}


def running_status(
    step_name: str, phase: int | None = None, **extra: Any
) -> dict[str, Any]:
    """Build a ``running`` status payload for one internal step.

    Args:
        step_name: Internal step name (kept as ``current_step`` for the log).
        phase: Pipeline phase (1 before labeling, 2 after).
        **extra: Further fields (e.g. ``cameras``).

    Returns:
        Status dict with ``state``, plain ``message``, ``stage`` and
        ``current_step``.
    """
    info = describe_step(step_name)
    payload: dict[str, Any] = {
        "state": "running",
        "message": info["message"],
        "stage": info["stage"],
        "current_step": step_name,
    }
    if phase is not None:
        payload["phase"] = phase
    payload.update(extra)
    return payload


def support_contact() -> str:
    """Return who to contact about a failed job.

    Returns:
        ``PORTAL_SUPPORT_CONTACT`` when set and not blank, else
        ``DEFAULT_SUPPORT_CONTACT``.
    """
    return os.environ.get("PORTAL_SUPPORT_CONTACT", "").strip() or (
        DEFAULT_SUPPORT_CONTACT
    )


def friendly_failure_message(step_name: str | None, error: BaseException) -> str:
    """Return a short plain message saying where a job stopped.

    A raw command line (``Command '[...]' returned non-zero exit status``)
    is never part of it. ``ValueError`` messages are written for people
    (e.g. "The two videos ended up at different frames per second ..."),
    so they follow the sentence; any other error stays in the details.

    Args:
        step_name: Internal name of the step that was running.
        error: The exception that stopped the job.

    Returns:
        A short message for the job page.
    """
    phrase = describe_step(step_name)["failure"]
    if isinstance(error, ValueError) and str(error):
        return f"The job stopped while {phrase}. {error}"
    return (
        f"The job stopped while {phrase}. Open the technical details below "
        f"and send them to {support_contact()}."
    )


def failed_status(
    step_name: str | None, error: BaseException, **extra: Any
) -> dict[str, Any]:
    """Build a ``failed`` status payload with a plain message.

    Args:
        step_name: Internal name of the step that was running.
        error: The exception that stopped the job (kept in ``error``).
        **extra: Further fields (e.g. ``phase``, ``log_path``).

    Returns:
        Status dict with ``state``, plain ``message``, ``stage``,
        ``failed_step`` and the raw ``error`` text.
    """
    payload: dict[str, Any] = {
        "state": "failed",
        "message": friendly_failure_message(step_name, error),
        "stage": describe_step(step_name)["stage"],
        "current_step": None,
        "failed_step": step_name,
        "error": f"{type(error).__name__}: {error}",
    }
    payload.update(extra)
    return payload


def failing_step_name(error: BaseException, default: str | None) -> str | None:
    """Return the step a failure came from (tagged by ``run_steps``).

    Args:
        error: The exception raised by a step.
        default: Step name to use when the error carries none.

    Returns:
        The failing step's internal name, or ``default``.
    """
    return getattr(error, "portal_step_name", None) or default


JOB_NOTES_FILENAME = "notes.json"


def read_job_notes(job_dir: Path) -> list[str]:
    """Return the job's plain-language notes (empty when there are none).

    Args:
        job_dir: Job working directory.

    Returns:
        Notes in the order they were added.
    """
    path = job_dir / JOB_NOTES_FILENAME
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in payload] if isinstance(payload, list) else []


def add_job_note(job_dir: Path, text: str) -> None:
    """Append one plain-language note to the job (listed on the job page).

    Args:
        job_dir: Job working directory.
        text: The note.
    """
    notes = read_job_notes(job_dir)
    notes.append(text)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / JOB_NOTES_FILENAME).write_text(
        json.dumps(notes, indent=2), encoding="utf-8"
    )


def overlay_skip_note(step: CommandStep) -> str:
    """Plain note for a check clip that could not be made.

    Args:
        step: The overlay step (name ``overlay video <camera> <block>[ (end)]``).

    Returns:
        A sentence naming the video and block.
    """
    parts = step.name.split(" ", 3)
    block = parts[3] if len(parts) > 3 else step.name
    if block.endswith(" (end)"):
        block = block[: -len(" (end)")] + " end clip"
    video = CAMERA_LABELS.get(step.camera_id or "", "")
    where = f"{video}, {block}" if video else block
    return (
        f"Check clip not made ({where}). This usually means the block runs "
        "past the end of the video. The measurements are not affected."
    )


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
    from portal import run_log

    started = run_log.now()
    ok = False
    try:
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
        ok = True
    finally:
        run_log.record_step(
            Path(log_path).parent, step.name, step.camera_id, started, run_log.now(), ok
        )


def build_session_report_step(config: PortalJobConfig) -> CommandStep:
    """Build the step that writes the session table, report, and study table.

    Runs after every camera's chain: it reads all cameras' metric summaries
    and rebuilds ``all_sessions_metrics.csv`` across every processed session.

    Args:
        config: Portal job configuration.

    Returns:
        The session report command.
    """
    return CommandStep(
        name="session report",
        argv=[
            sys.executable,
            "-m",
            "video_analysis.session_report",
            "--session-dir",
            str(config.session_output_dir),
            "--all-sessions-root",
            str(config.output_root),
        ],
    )


def run_steps(
    steps: Sequence[CommandStep],
    cwd: Path,
    log_path: Path,
    on_start: Any = None,
    parallel_cameras: bool = False,
    on_optional_failure: Any = None,
) -> None:
    """Run command steps, optionally one thread per camera.

    Steps of one camera always run in order. With ``parallel_cameras`` the
    cameras' chains run concurrently; each writes its own log section, which
    is appended to ``log_path`` (camera A first) when the chains finish, so the
    job log stays readable.

    Args:
        steps: Steps to run (``camera_id`` groups them).
        cwd: Working directory for the subprocesses.
        log_path: Job log.
        on_start: Optional callback ``(step_number, total, step)`` called as
            each step starts.
        parallel_cameras: Run different cameras' chains concurrently.
        on_optional_failure: Optional callback ``(step, error)`` called when
            an ``optional`` step fails; the chain then continues.

    Raises:
        subprocess.CalledProcessError: The first failing required step (after
            every chain has stopped). The error carries the step's name as
            ``portal_step_name``.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    groups: dict[str | None, list[CommandStep]] = {}
    for step in steps:
        groups.setdefault(step.camera_id, []).append(step)
    total = len(steps)
    started = 0
    lock = threading.Lock()

    def notify(step: CommandStep) -> None:
        nonlocal started
        with lock:
            started += 1
            if on_start is not None:
                on_start(started, total, step)

    def run_one(step: CommandStep, step_log: Path) -> None:
        try:
            run_command_step(step, cwd=cwd, log_path=step_log)
        except Exception as error:
            if step.optional:
                if on_optional_failure is not None:
                    on_optional_failure(step, error)
                return
            if getattr(error, "portal_step_name", None) is None:
                try:
                    error.portal_step_name = step.name  # type: ignore[attr-defined]
                except AttributeError:
                    pass
            raise

    if not parallel_cameras or len(groups) <= 1:
        for step in steps:
            notify(step)
            run_one(step, log_path)
        return

    part_logs = {
        key: log_path.with_name(f"{log_path.stem}.{key or 'session'}{log_path.suffix}")
        for key in groups
    }

    gpu_lock = threading.Lock()

    def run_chain(key: str | None) -> None:
        for step in groups[key]:
            notify(step)
            if step.gpu_heavy:
                with gpu_lock:
                    run_one(step, part_logs[key])
            else:
                run_one(step, part_logs[key])

    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        futures = [pool.submit(run_chain, key) for key in groups]
        for future in futures:
            try:
                future.result()
            except BaseException as error:  # noqa: BLE001 - re-raised below
                errors.append(error)
    with log_path.open("a", encoding="utf-8") as log_file:
        for key, part in part_logs.items():
            if part.exists():
                log_file.write(part.read_text(encoding="utf-8", errors="replace"))
                part.unlink()
    if errors:
        raise errors[0]


RESULT_ZIP_EXCLUDED_DIRS = frozenset({"frames", "intermediate"})


def make_result_zip(
    session_output_dir: Path,
    zip_path: Path,
    extra_files: Iterable[tuple[Path, Path]] | None = None,
) -> None:
    """Create a zip archive of the session outputs.

    Skips intermediate artifacts the colleague never opens: extracted
    JPEG frames under any ``frames/`` directory and the raw inference
    JSON under any ``intermediate/`` directory. These can be hundreds
    of MB at 5 fps over a 10-minute block; the per-frame information
    is already encoded in the CSVs that *are* included.

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
            if not path.is_file():
                continue
            rel = path.relative_to(session_output_dir)
            if any(part in RESULT_ZIP_EXCLUDED_DIRS for part in rel.parts):
                continue
            zf.write(path, path.relative_to(session_output_dir.parent))
        for source_path, archive_path in extra_files or []:
            if source_path.exists():
                zf.write(source_path, archive_path)


def build_results_readme(config: PortalJobConfig) -> str:
    """Build the README.md that ships at the top of the results zip.

    The README documents every file in the archive, its schema, units, and
    how to load it, written so that both a non-technical colleague and an
    LLM/analyst can act on it without reading the pipeline source.

    Args:
        config: Portal job configuration for the completed session.

    Returns:
        Markdown text for ``<session_id>/README.md``.
    """
    camera_ids = active_camera_ids(config)
    tree_lines = []
    block_lines = []
    for camera_id in camera_ids:
        block_names = ", ".join(
            block.name for block in blocks_for_camera(config, camera_id)
        )
        tree_lines.append(
            f"  {camera_id + '/':<25}<- pose, tracks, metrics, figures, overlays"
        )
        block_lines.append(f"- `{camera_id}/` analysis blocks: {block_names}")
    camera_tree = "\n".join(tree_lines)
    camera_blocks_md = "\n".join(block_lines)
    fps = config.frame_rate

    return f"""# {config.session_id} — parent-child interaction analysis

This archive holds automated 2D pose tracking and interaction metrics for one
recording session. A person-detection + pose model (`{config.model_path}`) ran
on each video frame, a multi-object tracker linked detections across frames,
and you labeled which tracked person is the **parent** on a handful of anchor
frames. Everything here derives from that.

Generated by the Video Processing Portal. Session id: `{config.session_id}`.
Frame sampling rate: **{fps} fps** (one row per tracked person per sampled
frame).

## How to read this archive

```
{config.session_id}/
  README.md                <- this file
  report.html              <- START HERE: one-page summary (open in a browser)
  session_metrics.csv      <- all headline numbers, one row per camera x block
  task_metrics.csv         <- the same, one row per camera x task (parts rejoined)
  session_summary.json     <- run-level config + validation status
{camera_tree}
  portal_metadata/         <- provenance: what you submitted + the run log
```

Each `camera_*/` folder is one camera angle and is self-contained:

{camera_blocks_md}

## Start here

Open **`report.html`** in a browser. It shows, for every camera and block,
how much of the block had both people tracked, the headline numbers, and
links to the overlay clips and figures.

**`session_metrics.csv`** holds the same numbers as a table (one row per
camera and block) for stats software. **`task_metrics.csv`** has one row per
camera and task: when extra-person windows split a task into `_part1`,
`_part2`, ..., the parts are rejoined by pooling their frames (distance, gaze)
and their 20-second windows (movement synchrony). The portal also keeps a combined table
for every processed session; download it from the portal's home page
("Download all sessions table").

Key columns:

| column | meaning |
|---|---|
| `block_start_s`, `block_end_s`, `block_frames` | The block on this camera's own clock, and how many frames it spans. |
| `frames_with_parent`, `frames_with_child`, `frames_with_both` | Frames in the block where each person (or both) was tracked. |
| `coverage_both` | `frames_with_both / block_frames`. All dyad metrics use only those frames. |
| `low_coverage` | `True` if `coverage_both` is below 0.5. Treat that block's metrics with care and check the overlay clips. |
| `proximity_*`, `xcorr_*`, `lag_*` | Pose metrics (see `pose_metrics_summary.json` below). |
| `gaze_*` | Gaze metrics per block, when gaze analysis was requested. |
| `tracking_*` | Whole-camera tracking stats from `two_person_summary.json`. |

## Which files to analyze

Use the **`*_smooth.csv`** files. They are the final, analysis-ready outputs.
The other CSVs are earlier stages kept for transparency and debugging:

| Stage | tracks file | pose file | What it is |
|---|---|---|---|
| 1. raw | `tracks_2d.csv` | `pose_3d.csv` | Every detection from every tracked person, before role assignment. |
| 2. filtered | `tracks_2d_filtered.csv` | `pose_3d_filtered.csv` | The parent and the child, identified across the whole recording using your anchor labels. |
| 2b. refined | | `pose_3d_refined.csv` | Same people, keypoints re-estimated by a stronger pose model (ViTPose-Large) for steadier joints. Present for new jobs. |
| 3. interpolated | `tracks_2d_interpolated.csv` | `pose_3d_interpolated.csv` | Short detection gaps (1 s or less, inside one block) linearly filled so boxes/joints do not flicker. Longer gaps are left completely empty. Only real video frame times are filled. Filled rows are flagged in the `interpolated` column. |
| 4. smooth | `tracks_2d_smooth.csv` | `pose_3d_smooth.csv` | Stage 3 + temporal EMA smoothing. **Use these.** |

After stage 2, `track_id` is canonical: **`0` = parent, `1` = child**
(`track_label` column says so explicitly).

From stage 3 on, the `interpolated` column is `True` for any row whose
position was filled (gap-bridged or low-confidence-infilled) rather than
directly observed — useful if you want to analyze only observed frames.

## File schemas

### `tracks_2d*.csv` — one row per person per frame (bounding boxes)

| column | type | meaning |
|---|---|---|
| `frame_idx` | int | Frame index on the {fps} fps sampling grid. |
| `timestamp_s` | float | Time in seconds from the start of the video. |
| `track_id` | int | Person id. After filtering: 0 = parent, 1 = child. |
| `track_label` | str | `parent` or `child` (or `unknown` before filtering). |
| `bbox_x1,bbox_y1,bbox_x2,bbox_y2` | float | Bounding box corners in **image pixels** (top-left origin). |
| `track_confidence` | float | Detection confidence, 0–1. |
| `interpolated` | bool | `True` if this row was gap-filled rather than observed (stage 3+ only). |

### `pose_3d*.csv` — one row per person per keypoint per frame

| column | type | meaning |
|---|---|---|
| `frame_idx`, `timestamp_s`, `track_id`, `track_label` | | Same as above. |
| `keypoint_name` | str | `kp_000`…`kp_016`, the 17 COCO body keypoints (see below). |
| `x_m,y_m` | float | Keypoint position in **image pixels** (top-left origin). The `_m` suffix is a naming artifact; these are 2D pixel coordinates, not meters. |
| `z_m` | float | Always `0`. The pose model is 2D; this is a placeholder column for pipeline compatibility, not a depth estimate. |
| `keypoint_confidence` | float | Per-keypoint confidence, 0–1. Low-confidence joints are less reliable. |
| `interpolated` | bool | `True` if this keypoint was gap-filled or low-confidence-infilled rather than observed (stage 3+ only). |

COCO-17 keypoint order (`kp_000`→nose, increasing index):

```
00 nose          01 left_eye      02 right_eye     03 left_ear      04 right_ear
05 left_shoulder 06 right_shoulder 07 left_elbow    08 right_elbow   09 left_wrist
10 right_wrist   11 left_hip      12 right_hip     13 left_knee     14 right_knee
15 left_ankle    16 right_ankle
```

### `pose_metrics_proximity.csv` — parent-child closeness over time

| column | type | meaning |
|---|---|---|
| `frame_idx`, `timestamp_s` | | Frame timing. |
| `torso_distance_px` | float | Distance between the parent and child torso centroids, in **image pixels**. Smaller = closer. Pixel scale is camera-dependent; compare within a session, not across cameras. |
| `block` | str | Which analysis block this frame belongs to. |

### `pose_metrics_sync.csv` — movement synchrony (per 20 s window)

| column | type | meaning |
|---|---|---|
| `segment_id` | int | Which unbroken stretch of the block (both people tracked, no gap) the window comes from. |
| `window_start_s,window_end_s` | float | The 20 s window analyzed. |
| `sync_r` | float | Correlation between the parent's and the child's whole-body movement in that window, -1 to 1. Compare with the block's chance level (`sync_chance_mean` in the summary). |
| `block` | str | Analysis block. |

### `pose_metrics_xcorr.csv` — superseded torso-speed synchrony (per 5 s window)

| column | type | meaning |
|---|---|---|
| `segment_id` | int | Which unbroken stretch of the block (both people tracked, no gap) the window comes from. Windows never cross a gap. |
| `window_start_s,window_end_s` | float | The 5 s window analyzed. |
| `peak_xcorr` | float | Peak cross-correlation of parent vs child movement in that window, 0…1 (negative values are shown as 0). Higher = more synchronized motion. |
| `peak_lag_s` | float | Lag at the peak, in seconds. **Positive = child follows parent**; negative = child leads. |
| `block` | str | Analysis block. |

### `pose_metrics_summary.json` — per-block roll-up

A list of objects, one per block, with `proximity_mean/std/median/n_frames`
and `sync_mean/std`, `sync_n_windows`, `sync_chance_mean`, `sync_minus_chance`
(movement synchrony; only the part above chance reflects the interaction),
the superseded `xcorr_*` / `lag_*` columns, plus the tracking coverage columns
described under "Start here".

### Figures and video

- `pose_proximity.png` / `.svg` — torso distance over time and per-block comparison.
- `pose_synchrony.png` / `.svg` — synchrony strength and lead/follow per block.
- `overlay_<block>.mp4` and `overlay_<block>_end.mp4` — ~{int(config.overlay_seconds)} s clips of the **start** and **end** of each block, with boxes and skeletons drawn, to eyeball tracking quality at both ends. (A block shorter than ~{int(config.overlay_seconds)} s yields a single full-block `overlay_<block>.mp4`.) Every keypoint is drawn; keypoints that were gap-filled or low-confidence-infilled (`interpolated = True`) are shown muted (hollow dot, thin dimmed bone) so you can tell them from directly observed joints.
- SVGs are editable vector versions of the PNGs.

### Gaze (when requested)

A gaze model (PaGE) estimates, for each person in each frame, the point in the
image they are looking at (`gaze_heatmap.csv`: `gaze_peak_x/y` as fractions of
image width/height; `inout_score` = probability the target is inside the
image). Each frame is then given one category (`gaze_metrics_categories.csv`):

| category | rule |
|---|---|
| `mutual_gaze` | each person's gaze lands on the other's face |
| `parent_watching` / `child_watching` | one person's gaze lands on the other (face or body); the other's does not land back |
| `both_watching` | each looks at the other, not face to face |
| `joint_attention` | both look at something in view (not each other), gaze points within 1.5 head-heights |
| `independent` | both look at different things in view |
| `unknown` | someone's gaze is out of view, or a box is missing |

One-frame flickers are absorbed into the surrounding category.
`gaze_metrics_summary.json` gives each category's share per block
(`<category>_pct`, summing to 100) and its **chance level**
(`<category>_chance_pct`): the share with the child's gaze shifted 60 s in time,
which keeps seating positions but breaks the moment-to-moment pairing. Treat a
category as meaningful only when it clearly exceeds its chance level.
`convergence_mean` is the overlap of the two people's gaze maps (0 = disjoint,
1 = identical), with `convergence_chance` computed the same way.

Gaze estimates have not been validated against human coding on this lab's
videos; a blind visual comparison of two models on 50 random frames was used
to choose PaGE.

### `session_summary.json` and `portal_metadata/`

`session_summary.json` records the run configuration and a per-block validity
flag. Each camera folder also has `two_person_summary.json` (how many frames
contain the parent, the child, or both) and `keypoint_refinement_summary.json`
(share of person-frames whose keypoints were refined). `portal_metadata/` holds `submission_metadata.json` (exactly what was
submitted: blocks, exclusion windows), `job_config.json`, and `job.log` (the
full processing log) for provenance.

## Loading the data (Python)

```python
import pandas as pd

cam = "{camera_ids[0]}"  # one of: {", ".join(camera_ids)}
pose = pd.read_csv(f"{config.session_id}/{{cam}}/pose_3d_smooth.csv")
tracks = pd.read_csv(f"{config.session_id}/{{cam}}/tracks_2d_smooth.csv")
prox = pd.read_csv(f"{config.session_id}/{{cam}}/pose_metrics_proximity.csv")

parent = pose[pose.track_label == "parent"]          # track_id 0
child = pose[pose.track_label == "child"]            # track_id 1
nose_parent = parent[parent.keypoint_name == "kp_000"]   # nose over time

# Mean torso distance in the free_play block:
prox[prox.block == "free_play"]["torso_distance_px"].mean()
```

## Notes and caveats

- Coordinates use an **image / top-left origin**: x grows right, y grows down.
- Metrics are computed **only inside the analysis blocks** you defined; time
  outside those blocks (and any extra-person exclusion windows) is not analyzed.
- Pixel-based distances depend on camera placement and zoom; interpret them
  within a session, not as absolute physical distance.
- Interpolation fills only short gaps and only when the tracker keeps the same
  id across the gap; longer drop-outs remain empty rather than invented.
"""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _iter_existing_artifacts(paths: Iterable[Path]) -> list[str]:
    return [str(path) for path in paths if path.exists()]


PORTAL_ANCHOR_FRAMES_PER_CAMERA = 10


def extract_job_anchor_frames(
    config: PortalJobConfig,
    n_anchors: int = PORTAL_ANCHOR_FRAMES_PER_CAMERA,
) -> dict[str, Any]:
    """Render annotation anchor frames for every active camera.

    For each active camera, picks N evenly time-spaced frames from the
    tracked output, renders each one with bbox overlays, and writes them
    under ``<job_dir>/anchors/<camera_id>/`` along with a ``manifest.json``
    the portal UI consumes.

    Returns:
        Per-camera summary ``{camera_id: {"count": int, "dir": str}}``.
    """
    from video_analysis.anchor_frames import extract_anchor_frames

    anchors_root = config.job_dir / "anchors"
    anchors_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for camera_id in active_camera_ids(config):
        tracks_csv = config.session_output_dir / camera_id / "tracks_2d.csv"
        if not tracks_csv.exists():
            summary[camera_id] = {"count": 0, "error": "tracks_2d.csv missing"}
            continue
        source_video = config.video_a if camera_id == "camera_a" else config.video_b
        if source_video is None:
            summary[camera_id] = {"count": 0, "error": "source video missing"}
            continue
        out_dir = anchors_root / camera_id
        metadata = extract_anchor_frames(
            tracks_csv=tracks_csv,
            source_video=Path(source_video),
            output_dir=out_dir,
            n_anchors=n_anchors,
            blocks=[
                (block.name, block.start_s, block.end_s)
                for block in blocks_for_camera(config, camera_id)
            ],
        )
        summary[camera_id] = {
            "count": len(metadata),
            "dir": str(out_dir),
        }
    return summary


def resolve_anchors_json_paths(
    config: PortalJobConfig,
) -> dict[str, Path]:
    """Map each active camera to its anchors.json file under the job dir.

    The file is the user-submitted annotation set converted into the
    schema ``video-filter-tracks --anchors-json`` reads.
    """
    paths: dict[str, Path] = {}
    for camera_id in active_camera_ids(config):
        candidate = config.job_dir / "anchors" / camera_id / "anchors.json"
        if candidate.exists():
            paths[camera_id] = candidate
    return paths


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
    current_step = "compression"
    try:
        write_status(config.job_dir, running_status(current_step))

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

        steps = build_phase1_steps(config)
        for step in steps:
            current_step = step.name
            write_status(config.job_dir, running_status(step.name, phase=1))
            run_command_step(step, cwd=repo_root, log_path=log_path)

        # Record the real frame size (uploads are often 960x540, not the
        # 854x480 default); gaze head boxes are normalized by it.
        from gaze_analysis.config import frame_image_size

        config.image_width, config.image_height = frame_image_size(
            config.session_output_dir / active_camera_ids(config)[0],
            (config.image_width, config.image_height),
        )
        # Use the frame rate extraction actually achieved from here on.
        config.frame_rate = achieved_session_frame_rate(config)
        write_session_config(config)
        _write_json(config.job_dir / "job_config.json", job_config_to_dict(config))

        # After pose+tracking, extract anchor frames per camera and pause
        # for the colleague to label which detection is the parent. Phase 2
        # resumes via resume_portal_job_phase2 once annotations arrive.
        current_step = "extract anchors"
        write_status(config.job_dir, running_status(current_step, phase=1))
        anchor_summary = extract_job_anchor_frames(config)
        write_status(
            config.job_dir,
            {
                "state": "awaiting_annotation",
                "message": (
                    "Ready for you: mark which person is the parent in a few "
                    "frames so processing can continue."
                ),
                "stage": "label_parent",
                "current_step": "awaiting annotation",
                "phase": 1,
                "annotate_url": f"/jobs/{config.job_id}/annotate",
                "anchor_summary": anchor_summary,
            },
        )
    except Exception as error:
        write_status(
            config.job_dir,
            failed_status(
                failing_step_name(error, current_step),
                error,
                phase=1,
                log_path=str(log_path),
            ),
        )
        from portal import run_log

        run_log.write_run_record(config, "failed", error=error)
        raise


def resume_portal_job_phase2(config: PortalJobConfig, repo_root: Path) -> None:
    """Continue a portal job after the colleague has submitted annotations.

    Expects ``<job_dir>/anchors/<camera>/anchors.json`` to exist for every
    active camera that should use anchor-based role assignment. Runs the
    filter -> smooth -> metrics -> overlay chain, then bundles the result
    zip.
    """
    import threading

    log_path = config.job_dir / "job.log"
    current_step = "phase 2 queued"
    # Latest plain message per video, so both cameras show while they run
    # side by side (on_start is called under run_steps' lock).
    cameras: dict[str, str] = {}
    notes_lock = threading.Lock()
    try:
        # The config is rebuilt from the upload, so re-read the frame rate
        # that frame extraction actually achieved in phase 1.
        config.frame_rate = achieved_session_frame_rate(config)
        anchors_map = resolve_anchors_json_paths(config)
        steps = build_phase2_steps(config, anchors_json_by_camera=anchors_map)

        def on_start(index: int, total: int, step: CommandStep) -> None:
            nonlocal current_step
            current_step = step.name
            if step.camera_id:
                cameras[step.camera_id] = describe_step(step.name)["message"]
            write_status(
                config.job_dir,
                running_status(
                    step.name,
                    phase=2,
                    cameras=dict(sorted(cameras.items())) if step.camera_id else {},
                ),
            )

        def on_optional_failure(step: CommandStep, error: BaseException) -> None:
            with notes_lock:
                add_job_note(config.job_dir, overlay_skip_note(step))

        run_steps(
            steps,
            cwd=repo_root,
            log_path=log_path,
            on_start=on_start,
            parallel_cameras=config.parallel_cameras,
            on_optional_failure=on_optional_failure,
        )
        report_step = build_session_report_step(config)
        on_start(len(steps) + 1, len(steps) + 1, report_step)
        run_command_step(report_step, cwd=repo_root, log_path=log_path)
        current_step = "results zip"
        write_status(config.job_dir, running_status(current_step, phase=2))

        # Run record (timings, versions, outcome); written before the zip so
        # the session's copy ships with the results.
        from portal import run_log

        run_log.write_run_record(config, "complete")
        zip_path = config.job_dir / f"{config.session_id}_results.zip"
        metadata_root = Path(config.session_id) / "portal_metadata"
        readme_path = config.job_dir / "README.md"
        readme_path.write_text(build_results_readme(config), encoding="utf-8")
        make_result_zip(
            config.session_output_dir,
            zip_path,
            extra_files=[
                (readme_path, Path(config.session_id) / "README.md"),
                (
                    config.job_dir / "submission_metadata.json",
                    metadata_root / "submission_metadata.json",
                ),
                (
                    config.job_dir / "job_config.json",
                    metadata_root / "job_config.json",
                ),
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
                "stage": "report",
                "current_step": None,
                "result_zip": str(zip_path),
                "artifacts": artifacts,
            },
        )
    except Exception as error:
        write_status(
            config.job_dir,
            failed_status(
                failing_step_name(error, current_step),
                error,
                phase=2,
                log_path=str(log_path),
            ),
        )
        from portal import run_log

        run_log.write_run_record(config, "failed", error=error)
        raise
