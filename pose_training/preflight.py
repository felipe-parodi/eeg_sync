"""Pre-flight environment checks for the pose-training pipeline.

Verifies that the things every later phase depends on are present: a CUDA
capable torch, at least one GPU, ffmpeg/ffprobe, the teacher/student weights,
and the compressed video source directory. Prints a plain-language ready /
not-ready summary so a non-technical operator can act on it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .coco import validate_flip_idx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = REPO_ROOT / "compressed_video"
DEFAULT_STUDENT_WEIGHTS = REPO_ROOT / "models" / "yolo11l-pose.pt"


@dataclass
class CheckResult:
    """Outcome of one pre-flight check."""

    name: str
    ok: bool
    detail: str


@dataclass
class PreflightReport:
    """Collected pre-flight results."""

    checks: List[CheckResult] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return all(check.ok for check in self.checks)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append(CheckResult(name=name, ok=ok, detail=detail))


def _check_cuda(report: PreflightReport) -> None:
    try:
        import torch
    except ImportError as error:
        report.add("torch", False, f"torch not importable: {error}")
        return
    if not torch.cuda.is_available():
        report.add(
            "cuda",
            False,
            f"torch {torch.__version__} has no CUDA. Use the project's .venv "
            "(uv run), not the system python.",
        )
        return
    gpus = [
        f"{torch.cuda.get_device_name(i)} "
        f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.0f} GB)"
        for i in range(torch.cuda.device_count())
    ]
    report.add("cuda", True, f"torch {torch.__version__}; GPUs: {', '.join(gpus)}")


def _check_binary(report: PreflightReport, name: str) -> None:
    path = shutil.which(name)
    if path is None:
        report.add(name, False, f"{name} not found on PATH")
        return
    try:
        proc = subprocess.run(
            [name, "-version"], capture_output=True, text=True, timeout=15
        )
        first_line = proc.stdout.splitlines()[0] if proc.stdout else path
    except (subprocess.SubprocessError, OSError) as error:
        report.add(name, False, f"{name} failed to run: {error}")
        return
    report.add(name, True, first_line)


def _check_weights(report: PreflightReport, student_weights: Path) -> None:
    if student_weights.exists():
        size_mb = student_weights.stat().st_size / 1e6
        report.add(
            "student_weights", True, f"{student_weights.name} ({size_mb:.0f} MB)"
        )
    else:
        report.add(
            "student_weights",
            False,
            f"{student_weights} not found (expected the current portal model)",
        )
    report.add(
        "teacher_weights",
        True,
        "yolo11x-pose.pt downloads automatically on first use",
    )


def _check_videos(report: PreflightReport, video_dir: Path) -> None:
    if not video_dir.exists():
        report.add("videos", False, f"{video_dir} not found")
        return
    clips = sorted(video_dir.glob("*.mp4"))
    report.add(
        "videos",
        bool(clips),
        f"{len(clips)} mp4 clip(s) in {video_dir.name}",
    )


def _check_flip_idx(report: PreflightReport) -> None:
    try:
        validate_flip_idx()
    except ValueError as error:
        report.add("flip_idx", False, str(error))
        return
    report.add("flip_idx", True, "COCO-17 flip index is a valid involution")


def run_preflight(video_dir: Path, student_weights: Path) -> PreflightReport:
    """Run every pre-flight check and return the collected report."""
    report = PreflightReport()
    _check_cuda(report)
    _check_binary(report, "ffmpeg")
    _check_binary(report, "ffprobe")
    _check_weights(report, student_weights)
    _check_videos(report, video_dir)
    _check_flip_idx(report)
    return report


def format_report(report: PreflightReport) -> str:
    """Render a report as a human-readable multi-line string."""
    lines = []
    for check in report.checks:
        mark = "OK " if check.ok else "XX "
        lines.append(f"  [{mark}] {check.name}: {check.detail}")
    verdict = "READY" if report.ready else "NOT READY (fix the XX lines above)"
    lines.append(f"\nPre-flight: {verdict}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pose-training environment pre-flight."
    )
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR), type=str)
    parser.add_argument(
        "--student-weights", default=str(DEFAULT_STUDENT_WEIGHTS), type=str
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_preflight(Path(args.video_dir), Path(args.student_weights))
    print(format_report(report))
    raise SystemExit(0 if report.ready else 1)


if __name__ == "__main__":
    main()
