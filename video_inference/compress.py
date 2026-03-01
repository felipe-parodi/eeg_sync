"""Video compression helpers for the inference pipeline."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class CompressionResult:
    """Compression stage result metadata."""

    input_path: Path
    output_path: Path
    command: List[str]
    executed: bool


def compress_video(
    input_path: Path | str,
    output_path: Path | str,
    target_fps: float = 15.0,
    max_width: int = 1280,
    crf: int = 23,
    preset: str = "medium",
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = True,
    dry_run: bool = False,
) -> CompressionResult:
    """
    Compress a video for faster frame extraction and inference.

    Produces H.264 video with normalized frame rate and capped width.
    """
    source = Path(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(f"Input video not found: {source}")

    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-vf",
        f"fps={target_fps},scale='min({max_width},iw)':-2",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-an",
        str(destination),
    ]

    if dry_run:
        return CompressionResult(source, destination, command, executed=False)

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"'{ffmpeg_bin}' not found. Install ffmpeg or set --ffmpeg-bin."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"ffmpeg compression failed for {source}") from error

    return CompressionResult(source, destination, command, executed=True)
