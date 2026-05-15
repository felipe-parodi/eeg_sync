import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "compress_videos_for_portal.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "compress_videos_for_portal", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load compressor script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_discover_videos_supports_file_and_directory(tmp_path: Path) -> None:
    module = _load_script_module()
    direct = tmp_path / "direct.mov"
    folder = tmp_path / "folder"
    folder.mkdir()
    nested_video = folder / "nested.mp4"
    ignored = folder / "notes.txt"
    direct.write_text("x", encoding="utf-8")
    nested_video.write_text("x", encoding="utf-8")
    ignored.write_text("x", encoding="utf-8")

    discovered = module.discover_videos(
        videos=[direct],
        input_dir=folder,
        recursive=False,
    )

    assert discovered == [direct, nested_video]


def test_build_ffmpeg_command_prefers_nvenc_profile(tmp_path: Path) -> None:
    module = _load_script_module()
    settings = module.CompressionSettings()
    command = module.build_ffmpeg_command(
        input_path=tmp_path / "raw.mov",
        output_path=tmp_path / "raw_portal.mp4",
        settings=settings,
        encoder="h264_nvenc",
    )

    assert command[:5] == ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert "fps=8.0,scale='min(854,iw)':-2" in command
    assert "-an" in command


def test_build_ffmpeg_command_supports_cpu_fallback(tmp_path: Path) -> None:
    module = _load_script_module()
    settings = module.CompressionSettings()
    command = module.build_ffmpeg_command(
        input_path=tmp_path / "raw.mov",
        output_path=tmp_path / "raw_portal.mp4",
        settings=settings,
        encoder="libx264",
    )

    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-preset") + 1] == "ultrafast"
    assert command[command.index("-crf") + 1] == "34"


def test_validate_unique_output_paths_rejects_stem_collisions(tmp_path: Path) -> None:
    module = _load_script_module()
    input_a = tmp_path / "a" / "session.mov"
    input_b = tmp_path / "b" / "session.mp4"
    output_dir = tmp_path / "out"
    output_paths = [
        module.output_path_for(input_a, output_dir, "_portal"),
        module.output_path_for(input_b, output_dir, "_portal"),
    ]

    with pytest.raises(ValueError, match="same compressed output"):
        module.validate_unique_output_paths([input_a, input_b], output_paths)


def test_compress_one_video_falls_back_to_cpu_encoder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module()
    input_path = tmp_path / "raw.mov"
    output_path = tmp_path / "raw_portal.mp4"
    input_path.write_bytes(b"x" * 1024)
    attempts: list[str] = []

    def fake_run(command, check, capture_output, text):
        encoder = command[command.index("-c:v") + 1]
        attempts.append(encoder)
        if encoder == "h264_nvenc":
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=command,
                stderr="No capable devices found",
            )
        output_path.write_bytes(b"compressed")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = module.compress_one_video(
        input_path=input_path,
        output_path=output_path,
        settings=module.CompressionSettings(),
        encoders=("h264_nvenc", "libx264"),
    )

    assert attempts == ["h264_nvenc", "libx264"]
    assert result["encoder"] == "libx264"
    assert result["executed"] is True
