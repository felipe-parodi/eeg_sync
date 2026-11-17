import csv
import json
import subprocess
from pathlib import Path

import pytest

import video_inference.frames as frames_module
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
    monkeypatch.setattr(frames_module, "probe_source_frame_rate", lambda *a: 10.0)

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
    monkeypatch.setattr(frames_module, "probe_source_frame_rate", lambda *a: 10.0)

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
    monkeypatch.setattr(frames_module, "probe_source_frame_rate", lambda *a: 10.0)

    with pytest.raises(RuntimeError, match="Invalid data found"):
        extract_frames_ffmpeg(
            video_path=source,
            frames_dir=tmp_path / "frames",
            frame_rate=2.0,
            analysis_windows=[AnalysisWindow(name="bad", start_s=10.0, end_s=11.0)],
        )


# ---- Even frame spacing on a shared grid (metrics-validity plan, phase 0) ----


def _fake_ffmpeg(
    frames_per_segment: list[int],
    commands: list[list[str]],
    pts_times: list[list[float]] | None = None,
):
    """Fake ffmpeg that writes frame files and, optionally, showinfo lines."""

    def fake_run(command: list[str], check: bool, **kwargs):
        commands.append(command)
        segment = len(commands) - 1
        start_number = int(command[command.index("-start_number") + 1])
        output_pattern = Path(command[-1])
        for offset in range(frames_per_segment[segment]):
            (
                output_pattern.parent / f"frame_{start_number + offset:06d}.jpg"
            ).write_text("x", encoding="utf-8")
        stderr = ""
        if pts_times is not None:
            stderr = "\n".join(
                f"[Parsed_showinfo_1 @ 0000] n:{i} pts:{int(t * 10240)} "
                f"pts_time:{t:g} duration:1024"
                for i, t in enumerate(pts_times[segment])
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr=stderr)

    return fake_run


def _index_times(frames_dir: Path) -> list[float]:
    with (frames_dir / "frame_index.csv").open(encoding="utf-8", newline="") as handle:
        return [float(row["timestamp_s"]) for row in csv.DictReader(handle)]


def test_extraction_keeps_every_kth_source_frame(tmp_path: Path, monkeypatch) -> None:
    """10 fps source at 5 fps: every 2nd source frame, counted from video start."""
    source = tmp_path / "source.mp4"
    source.write_text("fake", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_ffmpeg([5], commands))
    monkeypatch.setattr(frames_module, "probe_source_frame_rate", lambda *a: 10.0)

    result = extract_frames_ffmpeg(
        source,
        tmp_path / "frames",
        frame_rate=5.0,
        analysis_windows=[AnalysisWindow("g", 838.25, 839.25)],
    )

    vf = commands[0][commands[0].index("-vf") + 1]
    # First decoded source frame is 8383 (838.3 s); keep frames where
    # (n + 8383) is a multiple of 2.
    assert vf.startswith(r"select=not(mod(n+8383\,2))")
    assert "showinfo" in vf
    assert result.frame_rate == 5.0
    times = _index_times(tmp_path / "frames")
    assert times == pytest.approx([838.4, 838.6, 838.8, 839.0, 839.2])


def test_extraction_frames_sit_on_one_grid_across_blocks(
    tmp_path: Path, monkeypatch
) -> None:
    """Blocks with different start offsets still share one 0.2 s grid."""
    source = tmp_path / "source.mp4"
    source.write_text("fake", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_ffmpeg([3, 3], commands))
    monkeypatch.setattr(frames_module, "probe_source_frame_rate", lambda *a: 10.0)

    extract_frames_ffmpeg(
        source,
        tmp_path / "frames",
        frame_rate=5.0,
        analysis_windows=[
            AnalysisWindow("a", 10.0, 10.6),
            AnalysisWindow("b", 100.31, 100.91),
        ],
    )

    grid = [t * 5.0 for t in _index_times(tmp_path / "frames")]
    assert grid == pytest.approx([round(g) for g in grid], abs=1e-6)


def test_extraction_uses_true_times_reported_by_ffmpeg(
    tmp_path: Path, monkeypatch
) -> None:
    """showinfo times (relative to the seek point) win over the calculation."""
    source = tmp_path / "source.mp4"
    source.write_text("fake", encoding="utf-8")
    commands: list[list[str]] = []
    # ffmpeg reports frames 0.15 s after the 838.25 seek point, every 0.2 s.
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_ffmpeg([3], commands, pts_times=[[0.15, 0.35, 0.55]]),
    )
    monkeypatch.setattr(frames_module, "probe_source_frame_rate", lambda *a: 10.0)

    extract_frames_ffmpeg(
        source,
        tmp_path / "frames",
        frame_rate=5.0,
        analysis_windows=[AnalysisWindow("g", 838.25, 838.85)],
    )

    assert _index_times(tmp_path / "frames") == pytest.approx([838.4, 838.6, 838.8])


@pytest.mark.parametrize(
    ("source_fps", "expected_rate"),
    [(10.0, 5.0), (8.0, 4.0), (15.0, 5.0), (30.0, 5.0), (59.94, 4.995)],
)
def test_extraction_reports_achieved_rate(
    tmp_path: Path, monkeypatch, source_fps: float, expected_rate: float
) -> None:
    """The achieved rate is source fps / k and is written to extraction_info.json."""
    source = tmp_path / "source.mp4"
    source.write_text("fake", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_ffmpeg([2], commands))
    monkeypatch.setattr(frames_module, "probe_source_frame_rate", lambda *a: source_fps)

    result = extract_frames_ffmpeg(
        source,
        tmp_path / "frames",
        frame_rate=5.0,
        analysis_windows=[AnalysisWindow("g", 0.0, 1.0)],
    )

    assert result.frame_rate == pytest.approx(expected_rate)
    info = json.loads((tmp_path / "frames" / "extraction_info.json").read_text())
    assert info["source_fps"] == pytest.approx(source_fps)
    assert info["achieved_frame_rate"] == pytest.approx(expected_rate)
    assert info["frame_step"] == round(source_fps / 5.0)
