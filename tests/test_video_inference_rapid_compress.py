import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_inference.compress import CompressionResult  # noqa: E402
from video_inference.rapid_compress import (  # noqa: E402
    RapidCompressionConfig,
    discover_videos,
    run_rapid_compression,
)


def test_discover_videos_collects_from_list_and_directory(tmp_path: Path):
    listed = tmp_path / "listed.mov"
    listed.write_text("x", encoding="utf-8")

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "a.mp4").write_text("x", encoding="utf-8")
    (input_dir / "b.txt").write_text("x", encoding="utf-8")

    found = discover_videos(videos=[str(listed)], input_dir=str(input_dir))
    assert {path.name for path in found} == {"listed.mov", "a.mp4"}


def test_run_rapid_compression_writes_summary_and_expected_outputs(tmp_path: Path):
    video_a = tmp_path / "a.mov"
    video_b = tmp_path / "b.mp4"
    video_a.write_text("a", encoding="utf-8")
    video_b.write_text("b", encoding="utf-8")

    calls = []

    def fake_compress_fn(**kwargs):
        calls.append(kwargs)
        return CompressionResult(
            input_path=Path(kwargs["input_path"]),
            output_path=Path(kwargs["output_path"]),
            command=["ffmpeg"],
            executed=False,
        )

    config = RapidCompressionConfig(
        videos=[str(video_a), str(video_b)],
        input_dir=None,
        output_dir=str(tmp_path / "compressed"),
        dry_run=True,
    )
    summary = run_rapid_compression(config, compress_fn=fake_compress_fn)

    assert summary["count"] == 2
    output_names = {Path(item["output_path"]).name for item in summary["outputs"]}
    assert output_names == {"a_rapid.mp4", "b_rapid.mp4"}
    assert len(calls) == 2

    summary_path = Path(summary["summary_path"])
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["count"] == 2
