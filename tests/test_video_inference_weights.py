"""Model weights live in models/; bare names resolve there."""

from __future__ import annotations

from pathlib import Path

from video_inference import weights


def test_bare_name_resolves_to_models_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert Path(weights.resolve_weights("yolo11m-pose.pt")) == (
        weights.MODELS_DIR / "yolo11m-pose.pt"
    )


def test_explicit_path_is_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "custom" / "my.pt"
    assert weights.resolve_weights(path) == str(path)


def test_file_in_current_folder_is_still_found(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local.pt").write_bytes(b"x")
    assert weights.resolve_weights("local.pt") == "local.pt"


def test_portal_and_bench_point_into_models_dir() -> None:
    from portal import jobs
    from pose_training import bench

    assert bench.DEFAULT_MODEL.parent == weights.MODELS_DIR
    assert Path(jobs.PORTAL_DEFAULT_MODEL_PATH).parent in (
        weights.MODELS_DIR,
        Path("."),
    )
