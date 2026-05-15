import csv
import subprocess
from pathlib import Path

import pytest

from video_inference.frames import (
    AnalysisWindow,
    extract_frames_ffmpeg,
    parse_analysis_windows,
)


def test_parse_analysis_windows_accepts_seconds_and_timecodes() -> None:
    windows = parse_analysis_windows("free,1:00,2:00;storybook,90,95.5")

    assert [(item.name, item.start_s, item.end_s) for item in windows] == [
        ("free", 60.0, 120.0),
        ("storybook", 90.0, 95.5),
    ]


def test_extract_frames_ffmpeg_writes_segmented_source_timestamps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_text("fake video", encoding="utf-8")
    frames_dir = tmp_path / "frames"
    commands: list[list[str]] = []
    frames_per_segment = [2, 1]

    def fake_run(command: list[str], check: bool, **kwargs) -> None:
        assert check is True
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        commands.append(command)
        start_number = int(command[command.index("-start_number") + 1])
        count = frames_per_segment[len(commands) - 1]
        output_pattern = Path(command[-1])
        for offset in range(count):
            image_path = (
                output_pattern.parent / f"frame_{start_number + offset:06d}.jpg"
            )
            image_path.write_text("x", encoding="utf-8")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = extract_frames_ffmpeg(
        video_path=source,
        frames_dir=frames_dir,
        frame_rate=2.0,
        analysis_windows=[
            AnalysisWindow(name="free", start_s=10.0, end_s=11.0),
            AnalysisWindow(name="storybook", start_s=20.0, end_s=20.5),
        ],
    )

    with (frames_dir / "frame_index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(commands) == 2
    assert commands[0][commands[0].index("-ss") + 1] == "10.000000"
    assert commands[0][commands[0].index("-t") + 1] == "1.000000"
    assert commands[0][commands[0].index("-start_number") + 1] == "1"
    assert commands[1][commands[1].index("-ss") + 1] == "20.000000"
    assert commands[1][commands[1].index("-start_number") + 1] == "3"
    assert [path.name for path in result.frame_paths] == [
        "frame_000001.jpg",
        "frame_000002.jpg",
        "frame_000003.jpg",
    ]
    assert [
        (
            int(row["frame_idx"]),
            float(row["timestamp_s"]),
            row["image_name"],
            row["window_name"],
        )
        for row in rows
    ] == [
        (0, 10.0, "frame_000001.jpg", "free"),
        (1, 10.5, "frame_000002.jpg", "free"),
        (2, 20.0, "frame_000003.jpg", "storybook"),
    ]


def test_extract_frames_ffmpeg_allows_window_with_zero_frames(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_text("fake video", encoding="utf-8")
    frames_dir = tmp_path / "frames"

    def fake_run(command: list[str], check: bool, **kwargs) -> None:
        assert check is True

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = extract_frames_ffmpeg(
        video_path=source,
        frames_dir=frames_dir,
        frame_rate=2.0,
        analysis_windows=[AnalysisWindow(name="past_eof", start_s=999.0, end_s=1000.0)],
    )

    with (frames_dir / "frame_index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert result.frame_paths == []
    assert rows == []


def test_extract_frames_ffmpeg_surfaces_stderr_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_text("fake video", encoding="utf-8")

    def fake_run(command: list[str], check: bool, **kwargs) -> None:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=command,
            stderr="Invalid data found when processing input",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Invalid data found"):
        extract_frames_ffmpeg(
            video_path=source,
            frames_dir=tmp_path / "frames",
            frame_rate=2.0,
            analysis_windows=[AnalysisWindow(name="bad", start_s=10.0, end_s=11.0)],
        )
