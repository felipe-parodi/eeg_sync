"""Frame extraction helpers for video inference."""

from __future__ import annotations

import csv
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


def _sorted_frame_paths(frames_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in frames_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )


def _write_frame_index(
    frame_paths: List[Path], frame_rate: float, output_csv: Path
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_idx", "timestamp_s", "image_name"])
        for idx, path in enumerate(frame_paths):
            writer.writerow([idx, f"{idx / frame_rate:.6f}", path.name])


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
    if analysis_windows is None:
        return []
    if isinstance(analysis_windows, str):
        return parse_analysis_windows(analysis_windows)
    return list(analysis_windows)


def _clear_existing_extracted_frames(frames_dir: Path) -> None:
    for path in _sorted_frame_paths(frames_dir):
        path.unlink()
    index_path = frames_dir / "frame_index.csv"
    if index_path.exists():
        index_path.unlink()


def _build_segment_command(
    source: Path,
    pattern: Path,
    window: AnalysisWindow,
    frame_rate: float,
    start_number: int,
    ffmpeg_bin: str,
    overwrite: bool,
) -> List[str]:
    duration_s = window.end_s - window.start_s
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{window.start_s:.6f}",
        "-t",
        f"{duration_s:.6f}",
        "-i",
        str(source),
        "-vf",
        f"fps={frame_rate}",
        "-start_number",
        str(start_number),
        "-q:v",
        "2",
        str(pattern),
    ]


def _extract_segmented_frames(
    source: Path,
    destination: Path,
    frame_rate: float,
    analysis_windows: Sequence[AnalysisWindow],
    ffmpeg_bin: str,
    overwrite: bool,
    dry_run: bool,
) -> FrameExtractionResult:
    if overwrite:
        _clear_existing_extracted_frames(destination)

    pattern = destination / "frame_%06d.jpg"
    commands: List[List[str]] = []
    segment_counts: List[int] = []
    next_start_number = 1
    for window in analysis_windows:
        command = _build_segment_command(
            source=source,
            pattern=pattern,
            window=window,
            frame_rate=frame_rate,
            start_number=next_start_number,
            ffmpeg_bin=ffmpeg_bin,
            overwrite=overwrite,
        )
        commands.append(command)
        if not dry_run:
            before = set(_sorted_frame_paths(destination))
            try:
                subprocess.run(command, check=True)
            except FileNotFoundError as error:
                raise RuntimeError(
                    f"'{ffmpeg_bin}' not found. Install ffmpeg or set --ffmpeg-bin."
                ) from error
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    f"ffmpeg frame extraction failed for {source} "
                    f"window {window.name}"
                ) from error
            after = set(_sorted_frame_paths(destination))
            new_count = len(after - before)
            segment_counts.append(new_count)
            next_start_number += new_count

    if dry_run:
        return FrameExtractionResult(
            video_path=source,
            frames_dir=destination,
            frame_rate=frame_rate,
            frame_paths=[],
            command=commands[0] if commands else [],
            commands=commands,
            executed=False,
        )

    frame_paths = _sorted_frame_paths(destination)
    rows: List[dict[str, str]] = []
    offset = 0
    for window, expected_count in zip(analysis_windows, segment_counts):
        segment_paths = frame_paths[offset : offset + expected_count]
        for local_idx, path in enumerate(segment_paths):
            timestamp_s = window.start_s + local_idx / frame_rate
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

    return FrameExtractionResult(
        video_path=source,
        frames_dir=destination,
        frame_rate=frame_rate,
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

    pattern = destination / "frame_%06d.jpg"
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-vf",
        f"fps={frame_rate}",
        "-q:v",
        "2",
        str(pattern),
    ]

    if dry_run:
        return FrameExtractionResult(
            video_path=source,
            frames_dir=destination,
            frame_rate=frame_rate,
            frame_paths=[],
            command=command,
            commands=[command],
            executed=False,
        )

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"'{ffmpeg_bin}' not found. Install ffmpeg or set --ffmpeg-bin."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"ffmpeg frame extraction failed for {source}") from error

    frame_paths = _sorted_frame_paths(destination)
    _write_frame_index(frame_paths, frame_rate, destination / "frame_index.csv")

    return FrameExtractionResult(
        video_path=source,
        frames_dir=destination,
        frame_rate=frame_rate,
        frame_paths=frame_paths,
        command=command,
        commands=[command],
        executed=True,
    )
