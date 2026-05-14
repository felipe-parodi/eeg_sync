"""Compress GoPro videos before upload to the Video Processing Portal.

Purpose:
    Use this script on a collaborator laptop before uploading videos through the
    web portal. It makes much smaller MP4 files that are faster to upload and
    still suitable for pose/gaze processing.

What you need:
    1. Python 3.10 or newer.
    2. ffmpeg installed and available on PATH, or an ffmpeg.exe file placed in
       the same folder as this script.

Examples:
    Compress one video:
        python compress_videos_for_portal.py --video "C:\\Videos\\session_A.mov"

    Compress two videos:
        python compress_videos_for_portal.py --video "A.mov" --video "B.mov"

    Compress every video in a folder:
        python compress_videos_for_portal.py --input-dir "C:\\Videos"

Outputs:
    Compressed files are written to a "portal_compressed" folder by default.
    The script also writes compression_summary.json with the input files,
    output files, encoder used, and file sizes.

Notes:
    The script uses only Python's standard library. It tries fast hardware H.264
    encoders first, then falls back to CPU H.264 if hardware encoding is not
    available. The defaults match the portal's expected rapid-compression
    profile: 8 FPS, max width 854 pixels, no audio, aggressive compression.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4"}
ENCODER_PRESETS = {
    "auto": ("h264_nvenc", "h264_qsv", "h264_amf", "libx264"),
    "nvenc": ("h264_nvenc",),
    "qsv": ("h264_qsv",),
    "amf": ("h264_amf",),
    "cpu": ("libx264",),
}


@dataclass
class CompressionSettings:
    """Settings for portal-compatible video compression."""

    target_fps: float = 8.0
    max_width: int = 854
    quality: int = 34
    cpu_preset: str = "ultrafast"
    ffmpeg_bin: str = "ffmpeg"
    overwrite: bool = True
    target_mb: float = 50.0


def resolve_ffmpeg(script_dir: Path, requested_bin: str) -> str:
    """Resolve the ffmpeg executable.

    Args:
        script_dir: Folder containing this script.
        requested_bin: User-provided ffmpeg executable name or path.

    Returns:
        Executable path or command name to pass to subprocess.
    """
    requested_path = Path(requested_bin)
    if requested_path.exists():
        return str(requested_path)

    local_ffmpeg = script_dir / (
        "ffmpeg.exe" if requested_bin == "ffmpeg" else requested_bin
    )
    if local_ffmpeg.exists():
        return str(local_ffmpeg)

    found = shutil.which(requested_bin)
    return found or requested_bin


def discover_videos(
    videos: Sequence[Path | str],
    input_dir: Path | str | None,
    recursive: bool = False,
) -> list[Path]:
    """Find input videos from explicit files and/or a folder.

    Args:
        videos: Explicit video file paths.
        input_dir: Optional folder containing videos.
        recursive: Whether to search subfolders.

    Returns:
        Unique input video paths in deterministic order.

    Raises:
        FileNotFoundError: If a supplied file or folder does not exist.
        ValueError: If no videos are found or a file extension is unsupported.
    """
    discovered = [Path(path) for path in videos if str(path)]

    if input_dir:
        folder = Path(input_dir)
        if not folder.exists():
            raise FileNotFoundError(f"Input folder not found: {folder}")
        iterator = folder.rglob("*") if recursive else folder.iterdir()
        discovered.extend(
            sorted(
                path
                for path in iterator
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            )
        )

    unique: list[Path] = []
    seen = set()
    for path in discovered:
        if not path.exists():
            raise FileNotFoundError(f"Input video not found: {path}")
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video extension for {path.name}. "
                f"Supported: {sorted(VIDEO_EXTENSIONS)}"
            )
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)

    if not unique:
        raise ValueError("No videos found. Use --video or --input-dir.")
    return unique


def output_path_for(input_path: Path, output_dir: Path, suffix: str) -> Path:
    """Return the compressed output path for one input video."""
    return output_dir / f"{input_path.stem}{suffix}.mp4"


def build_ffmpeg_command(
    input_path: Path,
    output_path: Path,
    settings: CompressionSettings,
    encoder: str,
) -> list[str]:
    """Build an ffmpeg command for one encoder attempt.

    Args:
        input_path: Source video.
        output_path: Destination MP4 path.
        settings: Compression settings.
        encoder: ffmpeg H.264 encoder name.

    Returns:
        Command argument list.
    """
    vf = f"fps={settings.target_fps},scale='min({settings.max_width},iw)':-2"
    command = [
        settings.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if settings.overwrite else "-n",
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


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def compress_one_video(
    input_path: Path,
    output_path: Path,
    settings: CompressionSettings,
    encoders: Sequence[str],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Compress one video and return a JSON-serializable summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for encoder in encoders:
        command = build_ffmpeg_command(input_path, output_path, settings, encoder)
        print(f"Compressing {input_path.name} with {encoder}...")
        if dry_run:
            return {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "encoder": encoder,
                "executed": False,
                "command": command,
            }
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            stderr = error.stderr.strip() if error.stderr else "no stderr"
            errors.append(f"{encoder}: {stderr}")
            if output_path.exists():
                output_path.unlink()
            continue

        input_mb = _size_mb(input_path)
        output_mb = _size_mb(output_path)
        warning = None
        if output_mb > settings.target_mb:
            warning = (
                f"Output is {output_mb:.1f} MB, above the "
                f"{settings.target_mb:.1f} MB target."
            )
        print(f"  Done: {output_mb:.1f} MB from {input_mb:.1f} MB")
        if warning:
            print(f"  Warning: {warning}")
        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "encoder": encoder,
            "input_size_mb": input_mb,
            "output_size_mb": output_mb,
            "warning": warning,
            "executed": True,
            "command": command,
        }

    raise RuntimeError(
        f"All encoder attempts failed for {input_path}: {' | '.join(errors)}"
    )


def run_compression(args: argparse.Namespace) -> dict[str, Any]:
    """Run compression from parsed CLI arguments."""
    script_dir = Path(__file__).resolve().parent
    ffmpeg_bin = resolve_ffmpeg(script_dir, args.ffmpeg_bin)
    settings = CompressionSettings(
        target_fps=args.target_fps,
        max_width=args.max_width,
        quality=args.quality,
        cpu_preset=args.cpu_preset,
        ffmpeg_bin=ffmpeg_bin,
        overwrite=not args.no_overwrite,
        target_mb=args.target_mb,
    )
    inputs = discover_videos(args.video, args.input_dir, args.recursive)
    output_dir = Path(args.output_dir)
    encoders = ENCODER_PRESETS[args.encoder]
    results = [
        compress_one_video(
            input_path=path,
            output_path=output_path_for(path, output_dir, args.suffix),
            settings=settings,
            encoders=encoders,
            dry_run=args.dry_run,
        )
        for path in inputs
    ]
    summary = {
        "settings": asdict(settings),
        "encoder_mode": args.encoder,
        "count": len(results),
        "outputs": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "compression_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to: {summary_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Compress videos for upload to the Video Processing Portal."
    )
    parser.add_argument("--video", action="append", default=[], help="Video file path.")
    parser.add_argument("--input-dir", default=None, help="Folder of videos.")
    parser.add_argument(
        "--recursive", action="store_true", help="Search input folder recursively."
    )
    parser.add_argument(
        "--output-dir", default="portal_compressed", help="Output folder."
    )
    parser.add_argument("--suffix", default="_portal", help="Output filename suffix.")
    parser.add_argument("--target-fps", default=8.0, type=float)
    parser.add_argument("--max-width", default=854, type=int)
    parser.add_argument("--quality", default=34, type=int)
    parser.add_argument("--cpu-preset", default="ultrafast")
    parser.add_argument("--target-mb", default=50.0, type=float)
    parser.add_argument(
        "--encoder",
        choices=sorted(ENCODER_PRESETS),
        default="auto",
        help="Encoder mode. auto tries GPU first, then CPU.",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--no-overwrite", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        run_compression(args)
    except Exception as error:
        print(f"ERROR: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
