"""Frame extraction helpers for video inference."""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence


@dataclass(frozen=True)
class AnalysisWindow:
    """A source-video time window to extract for inference.

    Args:
        name: Human-readable block name.
        start_s: Start time in source-video seconds.
        end_s: End time in source-video seconds.
    """

    name: str
    start_s: float
    end_s: float


@dataclass
class FrameExtractionResult:
    """Frame extraction stage result metadata."""

    video_path: Path
    frames_dir: Path
    frame_rate: float
    frame_paths: List[Path]
    command: List[str]
    executed: bool
    commands: List[List[str]] = field(default_factory=list)


def _parse_timecode(value: str) -> float:
    """Parse seconds, MM:SS, or HH:MM:SS to seconds."""
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

    raise ValueError(f"Cannot parse time value: {value!r}")


def parse_analysis_windows(windows: str) -> List[AnalysisWindow]:
    """Parse a semicolon-delimited analysis-window CLI string.

    Args:
        windows: Window string such as ``name,start,end;name,start,end``.

    Returns:
        Parsed analysis windows in input order.

    Raises:
        ValueError: If any entry is malformed or has non-positive duration.
    """
    parsed: List[AnalysisWindow] = []
    for entry in windows.split(";"):
        text = entry.strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid analysis window: {entry!r}. "
                "Expected format: name,start,end"
            )
        name, start_raw, end_raw = parts
        if not name:
            raise ValueError(f"Invalid analysis window name: {entry!r}")
        start_s = _parse_timecode(start_raw)
        end_s = _parse_timecode(end_raw)
        if end_s <= start_s:
            raise ValueError(
                f"Analysis window {name!r} end must be after start "
                f"({start_raw} -> {end_raw})"
            )
        parsed.append(AnalysisWindow(name=name, start_s=start_s, end_s=end_s))
    return parsed


# ---------------------------------------------------------------------------
# Even frame sampling on a shared time grid
# ---------------------------------------------------------------------------

EXTRACTION_INFO_JSON = "extraction_info.json"
_SHOWINFO_PTS = re.compile(r"Parsed_showinfo.*?pts_time:\s*([-0-9.eE+]+)")


def _ffprobe_bin(ffmpeg_bin: str) -> str:
    """Return the ffprobe executable that sits next to ``ffmpeg_bin``."""
    path = Path(ffmpeg_bin)
    if path.name.lower().startswith("ffmpeg"):
        return str(path.with_name("ffprobe" + path.name[len("ffmpeg") :]))
    return "ffprobe"


def _parse_rate(text: str) -> float:
    """Parse an ffprobe rate such as ``10/1`` or ``30000/1001``."""
    if "/" in text:
        num, den = text.split("/", 1)
        return float(num) / float(den) if float(den) else 0.0
    return float(text)


def probe_source_frame_rate(source: Path, ffmpeg_bin: str = "ffmpeg") -> float:
    """Read a video's frame rate with ffprobe.

    Args:
        source: Video file.
        ffmpeg_bin: ffmpeg executable; ffprobe is looked up next to it.

    Returns:
        Frames per second of the first video stream.

    Raises:
        RuntimeError: If ffprobe is missing or reports no usable rate.
    """
    command = [
        _ffprobe_bin(ffmpeg_bin),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate",
        "-of",
        "json",
        str(source),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "ffprobe was not found next to ffmpeg. Install ffmpeg (it includes ffprobe)."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Could not read the frame rate of {source}: {(error.stderr or '').strip()}"
        ) from error
    streams = json.loads(completed.stdout or "{}").get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {source}.")
    nominal = _parse_rate(str(streams[0].get("r_frame_rate", "0/0")))
    average = _parse_rate(str(streams[0].get("avg_frame_rate", "0/0")))
    rate = nominal or average
    if rate <= 0:
        raise RuntimeError(f"Could not read the frame rate of {source}.")
    if average and abs(average - nominal) / rate > 0.01:
        print(
            f"[frames] warning: {source.name} has a variable frame rate "
            f"({nominal:.3f} nominal vs {average:.3f} average); frame times are "
            "taken from the video itself."
        )
    return rate


def frame_step(source_fps: float, target_fps: float) -> int:
    """Keep every k-th source frame so the spacing is even and near the target.

    Args:
        source_fps: Source video frame rate.
        target_fps: Desired analysis rate.

    Returns:
        k >= 1. The achieved rate is ``source_fps / k``.
    """
    if target_fps <= 0:
        raise ValueError("frame_rate must be > 0")
    return max(1, int(round(source_fps / target_fps)))


def _first_source_frame(start_s: float, source_fps: float) -> int:
    """Index of the first source frame at or after ``start_s``."""
    return int(math.ceil(start_s * source_fps - 1e-6))


def _select_filter(start_s: float, source_fps: float, step: int) -> str:
    """ffmpeg filter keeping frames whose video-wide number is a multiple of k.

    After seeking, ffmpeg numbers frames from 0 at the first decoded frame, so
    the offset lines the selection up with frame numbers counted from the start
    of the video. Every block then lands on the same time grid.
    """
    offset = _first_source_frame(start_s, source_fps)
    return f"select=not(mod(n+{offset}\\,{step})),showinfo"


def _segment_times(
    start_s: float,
    count: int,
    source_fps: float,
    step: int,
    stderr: str | None,
) -> List[float]:
    """True source times of the frames ffmpeg wrote for one segment.

    Uses the times ffmpeg reports (showinfo, relative to the seek point) when
    it reports one per frame; otherwise computes them from frame numbers.
    Either way each time is snapped to the nearest source frame.

    Args:
        start_s: Seek point in source seconds.
        count: Frames written for this segment.
        source_fps: Source frame rate.
        step: Keep-every-k value.
        stderr: ffmpeg stderr (may be None).

    Returns:
        One source time per written frame, in order.
    """
    reported = [float(m) for m in _SHOWINFO_PTS.findall(stderr or "")]
    if len(reported) == count:
        raw = [start_s + t for t in reported]
    else:
        first = _first_source_frame(start_s, source_fps)
        first_kept = first + ((-first) % step)
        raw = [(first_kept + i * step) / source_fps for i in range(count)]
    return [round(t * source_fps) / source_fps for t in raw]


def _write_extraction_info(
    destination: Path, source_fps: float, step: int, target_fps: float
) -> float:
    """Record the source rate, k, and the achieved rate next to the frames."""
    achieved = source_fps / step
    info = {
        "source_fps": source_fps,
        "frame_step": step,
        "requested_frame_rate": target_fps,
        "achieved_frame_rate": achieved,
        "note": "Every k-th source frame, counted from the start of the video; "
        "timestamp_s in frame_index.csv is each frame's true source time.",
    }
    (destination / EXTRACTION_INFO_JSON).write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    return achieved


def read_achieved_frame_rate(frames_dir: Path, default: float) -> float:
    """Return the achieved extraction rate recorded next to the frames.

    Args:
        frames_dir: Folder holding the extracted frames.
        default: Value used when no extraction info exists (older runs).

    Returns:
        Frames per second actually kept.
    """
    path = Path(frames_dir) / EXTRACTION_INFO_JSON
    if not path.exists():
        return default
    return float(json.loads(path.read_text(encoding="utf-8"))["achieved_frame_rate"])


def _sorted_frame_paths(frames_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in frames_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )


def _write_frame_index(
    frame_paths: List[Path], timestamps: Sequence[float], output_csv: Path
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_idx", "timestamp_s", "image_name"])
        for idx, (path, timestamp_s) in enumerate(zip(frame_paths, timestamps)):
            writer.writerow([idx, f"{timestamp_s:.6f}", path.name])


def _write_segmented_frame_index(
    rows: List[dict[str, str]],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "frame_idx",
            "timestamp_s",
            "image_name",
            "window_name",
            "window_start_s",
            "window_end_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_analysis_windows(
    analysis_windows: Sequence[AnalysisWindow] | str | None,
) -> List[AnalysisWindow]:
    """Normalize optional CLI or typed analysis windows to a list."""
    if analysis_windows is None:
        return []
    if isinstance(analysis_windows, str):
        return parse_analysis_windows(analysis_windows)
    return list(analysis_windows)


def _clear_existing_extracted_frames(frames_dir: Path) -> None:
    """Remove extracted frames and their timestamp index before re-extraction."""
    for path in _sorted_frame_paths(frames_dir):
        path.unlink()
    index_path = frames_dir / "frame_index.csv"
    if index_path.exists():
        index_path.unlink()


def _build_segment_command(
    source: Path,
    pattern: Path,
    window: AnalysisWindow,
    source_fps: float,
    step: int,
    start_number: int,
    ffmpeg_bin: str,
    overwrite: bool,
) -> List[str]:
    """Build an ffmpeg command for one half-open analysis window."""
    duration_s = window.end_s - window.start_s
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "info",
        "-y" if overwrite else "-n",
        "-ss",
        f"{window.start_s:.6f}",
        "-t",
        f"{duration_s:.6f}",
        "-i",
        str(source),
        "-vf",
        _select_filter(window.start_s, source_fps, step),
        "-fps_mode",
        "passthrough",
        "-start_number",
        str(start_number),
        "-q:v",
        "2",
        str(pattern),
    ]


def _ffmpeg_failure_message(
    source: Path,
    stderr: str | None,
    context: str,
) -> str:
    """Return a user-facing ffmpeg failure message with useful stderr."""
    stderr_text = (stderr or "").strip()
    if not stderr_text:
        stderr_text = "ffmpeg did not report stderr"
    if len(stderr_text) > 2000:
        stderr_text = stderr_text[-2000:]
    return f"ffmpeg frame extraction failed for {source} ({context}): {stderr_text}"


def _extract_segmented_frames(
    source: Path,
    destination: Path,
    frame_rate: float,
    analysis_windows: Sequence[AnalysisWindow],
    ffmpeg_bin: str,
    overwrite: bool,
    dry_run: bool,
) -> FrameExtractionResult:
    """Extract evenly spaced frames from only the requested source-video windows.

    Keeps every k-th source frame (k = round(source fps / frame_rate)), counted
    from the start of the video, and records each frame's true source time.
    The returned ``frame_rate`` is the achieved rate, source fps / k.
    """
    if overwrite:
        _clear_existing_extracted_frames(destination)

    source_fps = probe_source_frame_rate(source, ffmpeg_bin)
    step = frame_step(source_fps, frame_rate)
    achieved = source_fps / step

    pattern = destination / "frame_%06d.jpg"
    commands: List[List[str]] = []
    segment_counts: List[int] = []
    segment_stderr: List[str | None] = []
    next_start_number = 1
    for window in analysis_windows:
        command = _build_segment_command(
            source=source,
            pattern=pattern,
            window=window,
            source_fps=source_fps,
            step=step,
            start_number=next_start_number,
            ffmpeg_bin=ffmpeg_bin,
            overwrite=overwrite,
        )
        commands.append(command)
        if not dry_run:
            # The before/after diff relies on each segment using a unique
            # ffmpeg -start_number so existing frame names are never reused.
            before = set(_sorted_frame_paths(destination))
            try:
                completed = subprocess.run(
                    command, check=True, capture_output=True, text=True
                )
            except FileNotFoundError as error:
                raise RuntimeError(
                    f"'{ffmpeg_bin}' not found. Install ffmpeg or set --ffmpeg-bin."
                ) from error
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    _ffmpeg_failure_message(
                        source=source,
                        stderr=error.stderr,
                        context=f"window {window.name}",
                    )
                ) from error
            after = set(_sorted_frame_paths(destination))
            new_count = len(after - before)
            segment_counts.append(new_count)
            segment_stderr.append(getattr(completed, "stderr", None))
            next_start_number += new_count

    if dry_run:
        return FrameExtractionResult(
            video_path=source,
            frames_dir=destination,
            frame_rate=achieved,
            frame_paths=[],
            command=commands[0] if commands else [],
            commands=commands,
            executed=False,
        )

    frame_paths = _sorted_frame_paths(destination)
    rows: List[dict[str, str]] = []
    offset = 0
    for window, expected_count, stderr in zip(
        analysis_windows, segment_counts, segment_stderr
    ):
        segment_paths = frame_paths[offset : offset + expected_count]
        times = _segment_times(
            window.start_s, len(segment_paths), source_fps, step, stderr
        )
        for path, timestamp_s in zip(segment_paths, times):
            rows.append(
                {
                    "frame_idx": str(len(rows)),
                    "timestamp_s": f"{timestamp_s:.6f}",
                    "image_name": path.name,
                    "window_name": window.name,
                    "window_start_s": f"{window.start_s:.6f}",
                    "window_end_s": f"{window.end_s:.6f}",
                }
            )
        offset += expected_count

    if len(rows) != len(frame_paths):
        raise RuntimeError(
            "Segmented frame extraction produced an inconsistent frame index "
            f"({len(rows)} timestamp rows for {len(frame_paths)} frame files)."
        )
    _write_segmented_frame_index(rows, destination / "frame_index.csv")
    _write_extraction_info(destination, source_fps, step, frame_rate)

    return FrameExtractionResult(
        video_path=source,
        frames_dir=destination,
        frame_rate=achieved,
        frame_paths=frame_paths,
        command=commands[0] if commands else [],
        commands=commands,
        executed=True,
    )


def extract_frames_ffmpeg(
    video_path: Path | str,
    frames_dir: Path | str,
    frame_rate: float = 15.0,
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = True,
    dry_run: bool = False,
    analysis_windows: Sequence[AnalysisWindow] | str | None = None,
) -> FrameExtractionResult:
    """
    Extract image frames from a video via ffmpeg.

    Writes:
    - frame_000001.jpg, frame_000002.jpg, ... (ffmpeg's %06d default is
      1-indexed; downstream code that joins ``frame_idx`` to a filename
      must add 1, e.g. ``frame_{frame_idx + 1:06d}.jpg``).
    - frame_index.csv mapping the 0-indexed ``frame_idx`` to its
      ``timestamp_s`` and ``image_name``.
    """
    source = Path(video_path)
    destination = Path(frames_dir)
    destination.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    parsed_windows = _normalize_analysis_windows(analysis_windows)
    if parsed_windows:
        return _extract_segmented_frames(
            source=source,
            destination=destination,
            frame_rate=frame_rate,
            analysis_windows=parsed_windows,
            ffmpeg_bin=ffmpeg_bin,
            overwrite=overwrite,
            dry_run=dry_run,
        )

    source_fps = probe_source_frame_rate(source, ffmpeg_bin)
    step = frame_step(source_fps, frame_rate)
    achieved = source_fps / step

    pattern = destination / "frame_%06d.jpg"
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "info",
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-vf",
        _select_filter(0.0, source_fps, step),
        "-fps_mode",
        "passthrough",
        "-q:v",
        "2",
        str(pattern),
    ]

    if dry_run:
        return FrameExtractionResult(
            video_path=source,
            frames_dir=destination,
            frame_rate=achieved,
            frame_paths=[],
            command=command,
            commands=[command],
            executed=False,
        )

    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"'{ffmpeg_bin}' not found. Install ffmpeg or set --ffmpeg-bin."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            _ffmpeg_failure_message(
                source=source,
                stderr=error.stderr,
                context="full video",
            )
        ) from error

    frame_paths = _sorted_frame_paths(destination)
    times = _segment_times(
        0.0, len(frame_paths), source_fps, step, getattr(completed, "stderr", None)
    )
    _write_frame_index(frame_paths, times, destination / "frame_index.csv")
    _write_extraction_info(destination, source_fps, step, frame_rate)

    return FrameExtractionResult(
        video_path=source,
        frames_dir=destination,
        frame_rate=achieved,
        frame_paths=frame_paths,
        command=command,
        commands=[command],
        executed=True,
    )
