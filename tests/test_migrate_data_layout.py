"""Tests for scripts/migrate_data_layout.py (one-time move to data/, models/, logs/)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_data_layout.py"
spec = importlib.util.spec_from_file_location("migrate_data_layout", SCRIPT)
migrate = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(migrate)


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _old_layout(root: Path) -> None:
    _touch(root / "video_inference/output/1019v3/session_metrics.csv")
    _touch(root / "video_inference/output/portal_jobs/job1/status.json")
    _touch(root / "video_inference/data/portal_uploads/job1/camera_a.mp4")
    _touch(root / "video_inference/data/raw_session.mov")
    _touch(root / "video_inference/compressed/a_rapid.mp4")
    _touch(root / "pose_training/data/bench_cache/x.npz")
    _touch(root / "pose_training/runs/student/weights.pt")
    _touch(root / "compressed_video/1019_FULL_SHORT_compressed.mp4")
    _touch(root / "runs/pose/train/args.yaml")
    _touch(root / "yolo11l-pose.pt", "weights")
    _touch(root / "models/yolo11l-pose.pt", "weights")  # identical copy already moved
    _touch(root / "yolo26n.pt", "other")
    _touch(root / "portal_server.log")
    _touch(root / ".portal_tunnel.err.log")
    _touch(root / "video_inference/__pycache__/x.pyc")


def test_migration_moves_everything_into_new_layout(tmp_path: Path) -> None:
    _old_layout(tmp_path)

    report = migrate.migrate(tmp_path)

    expected = [
        "data/output/1019v3/session_metrics.csv",
        "data/output/portal_jobs/job1/status.json",
        "data/uploads/job1/camera_a.mp4",
        "data/videos/raw_session.mov",
        "data/compressed/a_rapid.mp4",
        "data/training/bench_cache/x.npz",
        "data/runs/student/weights.pt",
        "data/videos/1019_FULL_SHORT_compressed.mp4",
        "data/runs/pose/train/args.yaml",
        "models/yolo11l-pose.pt",
        "models/yolo26n.pt",
        "logs/portal_server.log",
        "logs/.portal_tunnel.err.log",
    ]
    for rel in expected:
        assert (tmp_path / rel).exists(), rel
    # Old locations are gone (leftover folders with only caches removed).
    for rel in (
        "video_inference",
        "pose_training",
        "compressed_video",
        "runs",
        "yolo11l-pose.pt",
        "yolo26n.pt",
        "portal_server.log",
    ):
        assert not (tmp_path / rel).exists(), rel
    assert report["conflicts"] == []


def test_migration_is_idempotent(tmp_path: Path) -> None:
    _old_layout(tmp_path)
    migrate.migrate(tmp_path)

    report = migrate.migrate(tmp_path)

    assert report["moved"] == [] and report["conflicts"] == []


def test_migration_never_overwrites(tmp_path: Path) -> None:
    _touch(tmp_path / "video_inference/output/1019v3/report.html", "old")
    _touch(tmp_path / "data/output/1019v3/report.html", "new")

    report = migrate.migrate(tmp_path)

    assert (tmp_path / "data/output/1019v3/report.html").read_text() == "new"
    assert (tmp_path / "video_inference/output/1019v3/report.html").read_text() == "old"
    assert any("report.html" in c for c in report["conflicts"])


def test_dry_run_changes_nothing(tmp_path: Path) -> None:
    _old_layout(tmp_path)

    report = migrate.migrate(tmp_path, dry_run=True)

    assert report["moved"]
    assert (tmp_path / "video_inference/output/1019v3/session_metrics.csv").exists()
    assert not (tmp_path / "data").exists()


def test_differing_root_weights_are_not_deleted(tmp_path: Path) -> None:
    _touch(tmp_path / "yolo11l-pose.pt", "root version")
    _touch(tmp_path / "models/yolo11l-pose.pt", "models version")

    report = migrate.migrate(tmp_path)

    assert (tmp_path / "yolo11l-pose.pt").exists()
    assert any("yolo11l-pose.pt" in c for c in report["conflicts"])
