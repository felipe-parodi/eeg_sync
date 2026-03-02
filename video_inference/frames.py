"""Frame extraction helpers for video inference."""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class FrameExtractionResult:
    """Frame extraction stage result metadata."""

    video_path: Path
    frames_dir: Path
    frame_rate: float
    frame_paths: List[Path]
    command: List[str]
    executed: bool


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


def extract_frames_ffmpeg(
    video_path: Path | str,
    frames_dir: Path | str,
    frame_rate: float = 15.0,
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = True,
    dry_run: bool = False,
) -> FrameExtractionResult:
    """
    Extract image frames from a video via ffmpeg.

    Writes:
    - frame_000000.jpg, frame_000001.jpg, ...
    - frame_index.csv (frame_idx -> timestamp)
    """
    source = Path(video_path)
    destination = Path(frames_dir)
    destination.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

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
        executed=True,
    )
