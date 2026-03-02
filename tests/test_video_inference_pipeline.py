import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_inference.compress import CompressionResult  # noqa: E402
from video_inference.frames import FrameExtractionResult  # noqa: E402
from video_inference.pipeline import (  # noqa: E402
    CameraPipelineSummary,
    PipelineConfig,
    run_camera_pipeline,
    run_pipeline,
)
from video_inference.schema import validate_session_output  # noqa: E402


def _fake_payload(frame_count: int = 3):
    frames = []
    for frame_idx in range(frame_count):
        frames.append(
            {
                "frame_idx": frame_idx,
                "image_name": f"frame_{frame_idx:06d}.jpg",
                "persons": [
                    {
                        "track_id": 0,
                        "track_label": "parent",
                        "bbox_xyxy": [10.0, 20.0, 100.0, 220.0],
                        "confidence": 0.95,
                        "keypoints_3d": [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
                    },
                    {
                        "track_id": 1,
                        "track_label": "child",
                        "bbox_xyxy": [140.0, 40.0, 220.0, 180.0],
                        "confidence": 0.93,
                        "keypoints_3d": [[0.4, 0.5, 0.6], [0.5, 0.6, 0.7]],
                    },
                ],
            }
        )
    return {"frames": frames}


def test_run_camera_pipeline_writes_schema_valid_outputs(tmp_path: Path):
    source_video = tmp_path / "input.mp4"
    source_video.write_text("fake", encoding="utf-8")

    config = PipelineConfig(
        video_a=str(source_video),
        video_b=None,
        output_dir=str(tmp_path / "out"),
        checkpoint_path="fake.ckpt",
        mhr_path="fake_mhr.pt",
        dry_run=True,
    )

    def fake_extract_fn(**kwargs):
        frames_dir = Path(kwargs["frames_dir"])
        frames_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(3):
            (frames_dir / f"frame_{idx:06d}.jpg").write_text("x", encoding="utf-8")
        (frames_dir / "frame_index.csv").write_text(
            "frame_idx,timestamp_s,image_name\n0,0.0,frame_000000.jpg\n",
            encoding="utf-8",
        )
        return FrameExtractionResult(
            video_path=Path(kwargs["video_path"]),
            frames_dir=frames_dir,
            frame_rate=float(kwargs["frame_rate"]),
            frame_paths=sorted(frames_dir.glob("frame_*.jpg")),
            command=[],
            executed=False,
        )

    def fake_infer_fn(runner_config):
        payload = _fake_payload(frame_count=3)
        Path(runner_config.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(runner_config.output_json).write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return payload

    summary = run_camera_pipeline(
        camera_id="camera_a",
        source_video=str(source_video),
        session_dir=tmp_path / "session",
        config=config,
        compress_fn=lambda **kwargs: CompressionResult(
            input_path=Path(kwargs["input_path"]),
            output_path=Path(kwargs["output_path"]),
            command=[],
            executed=False,
        ),
        extract_fn=fake_extract_fn,
        infer_fn=fake_infer_fn,
    )

    assert summary.schema_valid
    validation = validate_session_output(tmp_path / "session" / "camera_a")
    assert validation.is_valid, validation.errors


def test_run_pipeline_supports_two_cameras_with_mocks(tmp_path: Path):
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_text("a", encoding="utf-8")
    video_b.write_text("b", encoding="utf-8")

    config = PipelineConfig(
        video_a=str(video_a),
        video_b=str(video_b),
        output_dir=str(tmp_path / "pipeline_out"),
        checkpoint_path="fake.ckpt",
        mhr_path="fake_mhr.pt",
        dry_run=True,
    )

    from video_inference import pipeline as pipeline_module

    original_run_camera = pipeline_module.run_camera_pipeline

    def fake_run_camera_pipeline(camera_id, source_video, session_dir, config):
        camera_dir = Path(session_dir) / camera_id
        camera_dir.mkdir(parents=True, exist_ok=True)
        return CameraPipelineSummary(
            camera_id=camera_id,
            source_video=source_video,
            compressed_video=str(camera_dir / "intermediate" / "compressed.mp4"),
            frames_dir=str(camera_dir / "frames"),
            raw_output_json=str(camera_dir / "intermediate" / "inference_raw.json"),
            schema_valid=True,
            frame_count=2,
            errors=[],
        )

    try:
        pipeline_module.run_camera_pipeline = fake_run_camera_pipeline
        summary = run_pipeline(config)
    finally:
        pipeline_module.run_camera_pipeline = original_run_camera

    assert summary["overall_valid"]
    assert len(summary["cameras"]) == 2
    assert {item["camera_id"] for item in summary["cameras"]} == {
        "camera_a",
        "camera_b",
    }


def test_run_camera_pipeline_can_skip_compression(tmp_path: Path):
    source_video = tmp_path / "already_compressed.mp4"
    source_video.write_text("fake", encoding="utf-8")

    config = PipelineConfig(
        video_a=str(source_video),
        video_b=None,
        output_dir=str(tmp_path / "out"),
        checkpoint_path="fake.ckpt",
        mhr_path="fake_mhr.pt",
        skip_compress=True,
        dry_run=True,
    )

    def fake_extract_fn(**kwargs):
        assert Path(kwargs["video_path"]) == source_video
        frames_dir = Path(kwargs["frames_dir"])
        frames_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(2):
            (frames_dir / f"frame_{idx:06d}.jpg").write_text("x", encoding="utf-8")
        return FrameExtractionResult(
            video_path=Path(kwargs["video_path"]),
            frames_dir=frames_dir,
            frame_rate=float(kwargs["frame_rate"]),
            frame_paths=sorted(frames_dir.glob("frame_*.jpg")),
            command=[],
            executed=False,
        )

    def fake_infer_fn(runner_config):
        payload = _fake_payload(frame_count=2)
        Path(runner_config.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(runner_config.output_json).write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return payload

    summary = run_camera_pipeline(
        camera_id="camera_a",
        source_video=str(source_video),
        session_dir=tmp_path / "session",
        config=config,
        compress_fn=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("compress_fn should not be called when skip_compress=True")
        ),
        extract_fn=fake_extract_fn,
        infer_fn=fake_infer_fn,
    )

    assert summary.schema_valid


def test_run_camera_pipeline_uses_ultralytics_backend_config(tmp_path: Path):
    source_video = tmp_path / "input.mp4"
    source_video.write_text("fake", encoding="utf-8")

    config = PipelineConfig(
        video_a=str(source_video),
        video_b=None,
        output_dir=str(tmp_path / "out"),
        checkpoint_path="",
        mhr_path="",
        inference_backend="ultralytics",
        ultralytics_model_path="yolo11n-pose.pt",
        tracker_backend="internal",
        tracker_name="bytetrack",
        dry_run=True,
    )

    def fake_extract_fn(**kwargs):
        frames_dir = Path(kwargs["frames_dir"])
        frames_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(2):
            (frames_dir / f"frame_{idx:06d}.jpg").write_text("x", encoding="utf-8")
        return FrameExtractionResult(
            video_path=Path(kwargs["video_path"]),
            frames_dir=frames_dir,
            frame_rate=float(kwargs["frame_rate"]),
            frame_paths=sorted(frames_dir.glob("frame_*.jpg")),
            command=[],
            executed=False,
        )

    seen = {}

    def fake_infer_fn(runner_config):
        seen["runner_class"] = runner_config.__class__.__name__
        seen["model_path"] = runner_config.model_path
        payload = _fake_payload(frame_count=2)
        Path(runner_config.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(runner_config.output_json).write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return payload

    summary = run_camera_pipeline(
        camera_id="camera_a",
        source_video=str(source_video),
        session_dir=tmp_path / "session",
        config=config,
        compress_fn=lambda **kwargs: CompressionResult(
            input_path=Path(kwargs["input_path"]),
            output_path=Path(kwargs["output_path"]),
            command=[],
            executed=False,
        ),
        extract_fn=fake_extract_fn,
        infer_fn=fake_infer_fn,
    )

    assert summary.schema_valid
    assert seen["runner_class"] == "UltraRunnerConfig"
    assert seen["model_path"] == "yolo11n-pose.pt"
