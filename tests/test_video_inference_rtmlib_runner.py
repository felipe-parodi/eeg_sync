import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_inference.rtmlib_runner import (
    RtmlibRunnerConfig,
    _bbox_from_keypoints_2d,
    _resolve_rtmlib_device,
    run_rtmlib_pose_on_images,
)

# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


def test_bbox_from_keypoints_2d_basic():
    kpts = np.array([[10, 20], [30, 40], [50, 60]], dtype=float)
    bbox = _bbox_from_keypoints_2d(kpts, expansion=1.0)
    assert bbox == pytest.approx([10.0, 20.0, 50.0, 60.0])


def test_bbox_from_keypoints_2d_with_expansion():
    kpts = np.array([[10, 20], [30, 40], [50, 60]], dtype=float)
    bbox = _bbox_from_keypoints_2d(kpts, expansion=1.1)
    cx, cy = 30.0, 40.0
    half_w = 20.0 * 1.1
    half_h = 20.0 * 1.1
    assert bbox == pytest.approx([cx - half_w, cy - half_h, cx + half_w, cy + half_h])


def test_resolve_rtmlib_device_cuda(monkeypatch):
    monkeypatch.setattr(
        "video_inference.rtmlib_runner.resolve_device", lambda _: "cuda"
    )
    assert _resolve_rtmlib_device("auto") == "cuda"


def test_resolve_rtmlib_device_cpu(monkeypatch):
    monkeypatch.setattr("video_inference.rtmlib_runner.resolve_device", lambda _: "cpu")
    assert _resolve_rtmlib_device("auto") == "cpu"


def test_resolve_rtmlib_device_mps_maps_to_cuda(monkeypatch):
    monkeypatch.setattr("video_inference.rtmlib_runner.resolve_device", lambda _: "mps")
    assert _resolve_rtmlib_device("auto") == "cuda"


# ---------------------------------------------------------------------------
# Fake rtmlib mocks
# ---------------------------------------------------------------------------

N_WHOLEBODY_KPS = 133
N_COCO17_KPS = 17


def _make_fake_3d_tracker(n_persons: int = 2):
    """Return a callable that mimics PoseTracker(Wholebody3d) output."""

    def fake_call(frame):
        h, w = frame.shape[:2]
        kpts_3d = np.random.rand(n_persons, N_WHOLEBODY_KPS, 3).astype(np.float32)
        scores = np.random.rand(n_persons, N_WHOLEBODY_KPS).astype(np.float32)
        scores[:] = np.clip(scores, 0.5, 1.0)
        kpts_simcc = np.random.rand(n_persons, N_WHOLEBODY_KPS, 3).astype(np.float32)
        kpts_2d = np.random.rand(n_persons, N_WHOLEBODY_KPS, 2).astype(np.float32)
        kpts_2d[..., 0] *= w
        kpts_2d[..., 1] *= h
        return kpts_3d, scores, kpts_simcc, kpts_2d

    return fake_call


def _make_fake_2d_tracker(n_persons: int = 2):
    """Return a callable that mimics PoseTracker(Body) output."""

    def fake_call(frame):
        h, w = frame.shape[:2]
        kpts_2d = np.random.rand(n_persons, N_COCO17_KPS, 2).astype(np.float32)
        kpts_2d[..., 0] *= w
        kpts_2d[..., 1] *= h
        scores = np.random.rand(n_persons, N_COCO17_KPS).astype(np.float32)
        scores[:] = np.clip(scores, 0.5, 1.0)
        return kpts_2d, scores

    return fake_call


def _setup_fake_rtmlib(monkeypatch, mode_3d: bool, n_persons: int = 2):
    """Inject a fake rtmlib module with mock Wholebody3d / Body classes."""
    if mode_3d:
        fake_model_instance = MagicMock(side_effect=_make_fake_3d_tracker(n_persons))
    else:
        fake_model_instance = MagicMock(side_effect=_make_fake_2d_tracker(n_persons))

    # The runner calls Wholebody3d(...) or Body(...) to get a model, then
    # calls model(frame).  We make the class constructor return our fake.
    fake_wholebody3d_cls = MagicMock(return_value=fake_model_instance)
    fake_body_cls = MagicMock(return_value=fake_model_instance)

    fake_rtmlib = types.ModuleType("rtmlib")
    fake_rtmlib.Wholebody3d = fake_wholebody3d_cls
    fake_rtmlib.Body = fake_body_cls

    monkeypatch.setitem(sys.modules, "rtmlib", fake_rtmlib)
    return fake_model_instance


def _create_fake_frames(tmp_path, n_frames: int = 4):
    """Create minimal JPEG-like files to iterate over."""
    image_folder = tmp_path / "frames"
    image_folder.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        # Write a minimal valid image (1x1 black pixel).
        import cv2

        img = np.zeros((100, 160, 3), dtype=np.uint8)
        cv2.imwrite(str(image_folder / f"frame_{i:06d}.jpg"), img)
    return image_folder


# ---------------------------------------------------------------------------
# Integration tests: 3D mode (Wholebody3d)
# ---------------------------------------------------------------------------


def test_run_rtmlib_3d_produces_valid_payload(tmp_path, monkeypatch):
    monkeypatch.setattr("video_inference.rtmlib_runner.resolve_device", lambda _: "cpu")
    _setup_fake_rtmlib(monkeypatch, mode_3d=True, n_persons=2)
    image_folder = _create_fake_frames(tmp_path, n_frames=3)
    output_json = tmp_path / "output.json"

    cfg = RtmlibRunnerConfig(
        image_folder=str(image_folder),
        output_json=str(output_json),
        device="cpu",
        mode_3d=True,
        max_persons=2,
    )
    payload = run_rtmlib_pose_on_images(cfg)

    assert payload["backend"] == "rtmlib_wholebody3d"
    assert len(payload["frames"]) == 3

    for frame in payload["frames"]:
        persons = frame["persons"]
        assert len(persons) == 2
        for person in persons:
            assert person["track_id"] in {0, 1}
            assert person["track_label"] in {"parent", "child"}
            kps = person["keypoints_3d"]
            assert len(kps) == N_COCO17_KPS
            for kp in kps:
                assert len(kp) == 4  # [x, y, z, conf]


def test_run_rtmlib_3d_keypoints_use_pixel_coords(tmp_path, monkeypatch):
    """Verify keypoint x,y come from 2D pixel space, not 3D model space.

    The fake frames are 160x100. kpts_2d values are rand*w (0-160) and
    rand*h (0-100). kpts_3d values are rand(0,1). If the bug is present,
    x,y would be < 1.0; with the fix, they should be in pixel range.
    """
    monkeypatch.setattr("video_inference.rtmlib_runner.resolve_device", lambda _: "cpu")
    _setup_fake_rtmlib(monkeypatch, mode_3d=True, n_persons=1)
    image_folder = _create_fake_frames(tmp_path, n_frames=1)
    output_json = tmp_path / "output.json"

    cfg = RtmlibRunnerConfig(
        image_folder=str(image_folder),
        output_json=str(output_json),
        device="cpu",
        mode_3d=True,
        max_persons=2,
    )
    payload = run_rtmlib_pose_on_images(cfg)

    for frame in payload["frames"]:
        for person in frame["persons"]:
            bbox = person["bbox_xyxy"]
            for kp in person["keypoints_3d"]:
                x, y = kp[0], kp[1]
                # Pixel coords should be in image range, not model-space (0-1)
                assert x >= 0.0, f"kp x={x} below 0"
                assert y >= 0.0, f"kp y={y} below 0"
                # x,y should be within bbox range (same magnitude)
                # With random data, at least some kps should exceed 1.0
            xs = [kp[0] for kp in person["keypoints_3d"]]
            assert max(xs) > 1.0, (
                f"Max kp x={max(xs):.2f} is too small — likely model-space, "
                f"not pixel coords (bbox x range: {bbox[0]:.0f}-{bbox[2]:.0f})"
            )


def test_run_rtmlib_3d_keypoints_are_coco17_subset(tmp_path, monkeypatch):
    """Verify that 133 wholebody keypoints are subsetted to 17."""
    monkeypatch.setattr("video_inference.rtmlib_runner.resolve_device", lambda _: "cpu")
    _setup_fake_rtmlib(monkeypatch, mode_3d=True, n_persons=1)
    image_folder = _create_fake_frames(tmp_path, n_frames=1)
    output_json = tmp_path / "output.json"

    cfg = RtmlibRunnerConfig(
        image_folder=str(image_folder),
        output_json=str(output_json),
        device="cpu",
        mode_3d=True,
        max_persons=2,
    )
    payload = run_rtmlib_pose_on_images(cfg)

    for frame in payload["frames"]:
        for person in frame["persons"]:
            assert len(person["keypoints_3d"]) == N_COCO17_KPS


# ---------------------------------------------------------------------------
# Integration tests: 2D mode (Body)
# ---------------------------------------------------------------------------


def test_run_rtmlib_2d_produces_valid_payload(tmp_path, monkeypatch):
    monkeypatch.setattr("video_inference.rtmlib_runner.resolve_device", lambda _: "cpu")
    _setup_fake_rtmlib(monkeypatch, mode_3d=False, n_persons=2)
    image_folder = _create_fake_frames(tmp_path, n_frames=3)
    output_json = tmp_path / "output.json"

    cfg = RtmlibRunnerConfig(
        image_folder=str(image_folder),
        output_json=str(output_json),
        device="cpu",
        mode_3d=False,
        max_persons=2,
    )
    payload = run_rtmlib_pose_on_images(cfg)

    assert payload["backend"] == "rtmlib_body"
    assert len(payload["frames"]) == 3

    for frame in payload["frames"]:
        persons = frame["persons"]
        assert len(persons) == 2
        for person in persons:
            kps = person["keypoints_3d"]
            assert len(kps) == N_COCO17_KPS
            for kp in kps:
                assert len(kp) == 4
                assert kp[2] == 0.0  # z=0 for pseudo-3D


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_run_rtmlib_missing_image_folder(tmp_path):
    cfg = RtmlibRunnerConfig(
        image_folder=str(tmp_path / "nonexistent"),
        output_json=str(tmp_path / "output.json"),
    )
    with pytest.raises(FileNotFoundError):
        run_rtmlib_pose_on_images(cfg)


def test_run_rtmlib_empty_frames_produces_empty_payload(tmp_path, monkeypatch):
    monkeypatch.setattr("video_inference.rtmlib_runner.resolve_device", lambda _: "cpu")
    _setup_fake_rtmlib(monkeypatch, mode_3d=True, n_persons=0)
    image_folder = tmp_path / "frames"
    image_folder.mkdir(parents=True, exist_ok=True)
    output_json = tmp_path / "output.json"

    cfg = RtmlibRunnerConfig(
        image_folder=str(image_folder),
        output_json=str(output_json),
        device="cpu",
    )
    payload = run_rtmlib_pose_on_images(cfg)
    assert payload["frames"] == []


def test_run_rtmlib_tracking_assigns_parent_child(tmp_path, monkeypatch):
    """Parent should get track_id=0 (larger bbox), child track_id=1."""
    monkeypatch.setattr("video_inference.rtmlib_runner.resolve_device", lambda _: "cpu")
    _setup_fake_rtmlib(monkeypatch, mode_3d=True, n_persons=2)
    image_folder = _create_fake_frames(tmp_path, n_frames=2)
    output_json = tmp_path / "output.json"

    cfg = RtmlibRunnerConfig(
        image_folder=str(image_folder),
        output_json=str(output_json),
        device="cpu",
        mode_3d=True,
        max_persons=2,
    )
    payload = run_rtmlib_pose_on_images(cfg)

    for frame in payload["frames"]:
        labels = {p["track_label"] for p in frame["persons"]}
        assert labels == {"parent", "child"}
        ids = {p["track_id"] for p in frame["persons"]}
        assert ids == {0, 1}


def test_run_rtmlib_writes_json_file(tmp_path, monkeypatch):
    monkeypatch.setattr("video_inference.rtmlib_runner.resolve_device", lambda _: "cpu")
    _setup_fake_rtmlib(monkeypatch, mode_3d=True, n_persons=2)
    image_folder = _create_fake_frames(tmp_path, n_frames=1)
    output_json = tmp_path / "output.json"

    cfg = RtmlibRunnerConfig(
        image_folder=str(image_folder),
        output_json=str(output_json),
        device="cpu",
    )
    run_rtmlib_pose_on_images(cfg)

    import json

    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert "frames" in data
    assert "runner_config" in data
