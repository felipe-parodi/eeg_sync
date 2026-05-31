"""Tests for the student training wrapper (no real GPU training)."""

from __future__ import annotations

from pathlib import Path

from pose_training import train


class _FakeTrainer:
    def __init__(self, save_dir: Path) -> None:
        self.save_dir = save_dir


class _FakeModel:
    def __init__(self, save_dir: Path) -> None:
        self.save_dir = save_dir
        self.train_kwargs = None
        self.trainer = None

    def train(self, **kwargs):  # noqa: ANN003
        self.train_kwargs = kwargs
        weights = self.save_dir / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "best.pt").write_bytes(b"fake")
        self.trainer = _FakeTrainer(self.save_dir)
        return _FakeTrainer(self.save_dir)


def test_train_student_passes_core_args_and_returns_best(tmp_path: Path) -> None:
    save_dir = tmp_path / "run"
    model = _FakeModel(save_dir)
    config = train.TrainConfig(
        data_yaml=tmp_path / "data.yaml",
        base_weights="yolo11l-pose.pt",
        epochs=3,
        imgsz=960,
        batch=8,
        device="0",
        project=tmp_path / "runs",
        name="student",
        augment=False,
    )
    best = train.train_student(config, model=model)
    assert model.train_kwargs["data"] == str(tmp_path / "data.yaml")
    assert model.train_kwargs["epochs"] == 3
    assert model.train_kwargs["imgsz"] == 960
    assert model.train_kwargs["batch"] == 8
    # Fine-tuning safety recipe + augmentation knobs are forwarded.
    assert model.train_kwargs["freeze"] == 10
    assert model.train_kwargs["mosaic"] == 0.0
    assert model.train_kwargs["lr0"] == 0.001
    assert model.train_kwargs["degrees"] == 8.0
    assert model.train_kwargs["perspective"] == 0.0005
    assert best == save_dir / "weights" / "best.pt"
    assert best.exists()


def test_augmentation_transforms_are_pixel_only() -> None:
    import albumentations as A  # noqa: N812
    import numpy as np

    transforms = train.augmentation_transforms()
    assert len(transforms) >= 5
    image = (np.random.default_rng(0).random((540, 960, 3)) * 255).astype("uint8")
    out = A.Compose(transforms)(image=image)["image"]
    # Pixel-only transforms never change the image size (so keypoints stay valid).
    assert out.shape == image.shape
