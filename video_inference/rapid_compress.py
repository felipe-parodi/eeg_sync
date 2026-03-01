"""Rapid batch video compression CLI for faster inference."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .compress import CompressionResult, compress_video

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass
class RapidCompressionConfig:
    """Configuration for rapid compression runs."""

    videos: List[str]
    input_dir: Optional[str]
    output_dir: str
    suffix: str = "_rapid"
    target_fps: float = 10.0
    max_width: int = 960
    crf: int = 30
    preset: str = "veryfast"
    ffmpeg_bin: str = "ffmpeg"
    overwrite: bool = True
    dry_run: bool = False


def discover_videos(videos: List[str], input_dir: Optional[str]) -> List[Path]:
    """Resolve and validate the input video list."""
    discovered = [Path(path) for path in videos]

    if input_dir:
        input_dir_path = Path(input_dir)
        if not input_dir_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir_path}")
        discovered.extend(
            sorted(
                [
                    path
                    for path in input_dir_path.iterdir()
                    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
                ]
            )
        )

    unique_paths = []
    seen = set()
    for path in discovered:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)

    if not unique_paths:
        raise ValueError("No input videos found. Provide --video and/or --input-dir.")

    for path in unique_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input video not found: {path}")
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video extension for {path}. "
                f"Supported: {sorted(VIDEO_EXTENSIONS)}"
            )

    return unique_paths


def _output_path_for_video(video_path: Path, output_dir: Path, suffix: str) -> Path:
    return output_dir / f"{video_path.stem}{suffix}.mp4"


def run_rapid_compression(
    config: RapidCompressionConfig,
    compress_fn: Callable[..., CompressionResult] = compress_video,
) -> Dict[str, object]:
    """Compress one or more videos using rapid defaults."""
    inputs = discover_videos(config.videos, config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[CompressionResult] = []
    for input_video in inputs:
        output_video = _output_path_for_video(input_video, output_dir, config.suffix)
        result = compress_fn(
            input_path=input_video,
            output_path=output_video,
            target_fps=config.target_fps,
            max_width=config.max_width,
            crf=config.crf,
            preset=config.preset,
            ffmpeg_bin=config.ffmpeg_bin,
            overwrite=config.overwrite,
            dry_run=config.dry_run,
        )
        results.append(result)

    summary = {
        "config": asdict(config),
        "count": len(results),
        "outputs": [
            {
                "input_path": str(result.input_path),
                "output_path": str(result.output_path),
                "executed": result.executed,
                "command": result.command,
            }
            for result in results
        ],
    }

    summary_path = output_dir / "rapid_compression_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-compress-rapid",
        description="Rapidly compress videos before inference.",
    )
    parser.add_argument("--video", action="append", default=[], help="Input video path.")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory of videos to compress (non-recursive).",
    )
    parser.add_argument(
        "--output-dir", default="video_inference/compressed", help="Output directory."
    )
    parser.add_argument("--suffix", default="_rapid", help="Suffix for output files.")
    parser.add_argument("--target-fps", default=10.0, type=float)
    parser.add_argument("--max-width", default=960, type=int)
    parser.add_argument("--crf", default=30, type=int)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--no-overwrite", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = RapidCompressionConfig(
        videos=args.video,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        suffix=args.suffix,
        target_fps=args.target_fps,
        max_width=args.max_width,
        crf=args.crf,
        preset=args.preset,
        ffmpeg_bin=args.ffmpeg_bin,
        overwrite=not args.no_overwrite,
        dry_run=args.dry_run,
    )
    run_rapid_compression(config)


if __name__ == "__main__":
    main()
