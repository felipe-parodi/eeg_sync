import importlib.util
import sys
from pathlib import Path

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
