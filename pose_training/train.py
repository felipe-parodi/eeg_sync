"""Fine-tune a YOLO11 pose student on the teacher-labeled dataset.

Thin wrapper around Ultralytics training that starts from the COCO-pretrained
``yolo11l-pose.pt`` checkpoint (domain fine-tuning, not training from scratch)
and keeps outputs under the gitignored ``pose_training/runs`` directory. The
flip augmentation Ultralytics applies reads ``flip_idx`` from the dataset's
data.yaml, so the COCO-17 left/right swap is handled correctly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from video_inference.weights import resolve_weights

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = REPO_ROOT / "pose_training" / "data" / "yolo" / "data.yaml"
DEFAULT_PROJECT = REPO_ROOT / "pose_training" / "runs"
DEFAULT_BASE_WEIGHTS = "yolo11l-pose.pt"


def augmentation_transforms() -> list:
    """Domain-matched, pixel-only Albumentations for compressed GoPro footage.

    Every transform here is pixel-level: it changes appearance but never moves a
    keypoint. That is required because Ultralytics applies Albumentations to the
    image without keypoint params, so a spatial transform would desync the
    labels. These match the deployment domain (low-res, compressed, blurred).
    """
    import albumentations as A  # noqa: N812

    return [
        A.Downscale(scale_range=(0.25, 0.75), p=0.3),  # low-res / upscaling
        A.ImageCompression(quality_range=(30, 80), p=0.3),  # JPEG/codec artifacts
        A.MotionBlur(blur_limit=7, p=0.2),  # motion smear
        A.ToGray(p=0.1),  # lighting/white-balance
        A.ChannelDropout(p=0.05),  # drop a colour channel
        A.GaussNoise(p=0.1),  # sensor noise
    ]


def install_custom_albumentations() -> None:
    """Patch Ultralytics so training uses augmentation_transforms() by default.

    Ultralytics constructs ``Albumentations(p=1.0)`` (transforms=None -> its
    built-in list). We wrap ``__init__`` to substitute ours, reusing all of
    Ultralytics' own Compose/version handling. Idempotent.
    """
    import ultralytics.data.augment as aug

    original_init = aug.Albumentations.__init__
    if getattr(original_init, "_eeg_sync_custom", False):
        return
    custom = augmentation_transforms()

    def patched_init(self, p: float = 1.0, transforms: list | None = None) -> None:
        original_init(
            self, p=p, transforms=transforms if transforms is not None else custom
        )

    patched_init._eeg_sync_custom = True
    aug.Albumentations.__init__ = patched_init


@dataclass
class TrainConfig:
    """Configuration for student fine-tuning."""

    data_yaml: Path = DEFAULT_DATA_YAML
    base_weights: str = DEFAULT_BASE_WEIGHTS
    epochs: int = 40
    imgsz: int = 960
    batch: int = 16
    device: str = "0"
    project: Path = DEFAULT_PROJECT
    name: str = "student_l"
    patience: int = 15
    workers: int = 8
    seed: int = 0
    # Fine-tuning safety recipe. The first auto-only run (100 epochs, full
    # unfreeze, default mosaic) degraded the keypoint head below the COCO
    # baseline. These defaults adapt gently instead: freeze the pretrained
    # backbone, disable mosaic (it mangles close-up two-person scenes), and use
    # a low base LR so the model stays near its strong starting point.
    freeze: int = 10
    mosaic: float = 0.0
    lr0: float = 0.001
    # Domain augmentation. Pose-safe geometric knobs (small rotation +
    # perspective; flip/scale/translate/erasing keep Ultralytics defaults) plus
    # custom pixel-only Albumentations matched to compressed GoPro footage.
    augment: bool = True
    degrees: float = 8.0
    perspective: float = 0.0005


def _resolve_save_dir(model: Any, results: Any) -> Path:
    """Find the run output directory from the trainer or results object."""
    for source in (results, getattr(model, "trainer", None)):
        save_dir = getattr(source, "save_dir", None)
        if save_dir is not None:
            return Path(save_dir)
    raise RuntimeError("Could not resolve the training save directory")


def train_student(config: TrainConfig, model: Optional[Any] = None) -> Path:
    """Fine-tune the student model and return the path to ``best.pt``.

    Args:
        config: Training configuration.
        model: Optional pre-built model exposing ``.train(**kwargs)``. When
            None, a YOLO model is loaded from ``config.base_weights``.

    Returns:
        Path to the best checkpoint produced by training.
    """
    if model is None:
        from ultralytics import YOLO

        model = YOLO(resolve_weights(config.base_weights))

    if config.augment:
        install_custom_albumentations()

    results = model.train(
        data=str(config.data_yaml),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device,
        project=str(config.project),
        name=config.name,
        patience=config.patience,
        workers=config.workers,
        seed=config.seed,
        freeze=config.freeze,
        mosaic=config.mosaic,
        lr0=config.lr0,
        degrees=config.degrees,
        perspective=config.perspective,
        pretrained=True,
        exist_ok=True,
        verbose=True,
    )
    save_dir = _resolve_save_dir(model, results)
    return save_dir / "weights" / "best.pt"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLO11 pose student on teacher pseudo-labels."
    )
    parser.add_argument("--data-yaml", default=str(DEFAULT_DATA_YAML), type=str)
    parser.add_argument("--base-weights", default=DEFAULT_BASE_WEIGHTS, type=str)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--imgsz", default=960, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--device", default="0", type=str)
    parser.add_argument("--project", default=str(DEFAULT_PROJECT), type=str)
    parser.add_argument("--name", default="student_l", type=str)
    parser.add_argument("--patience", default=15, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--freeze", default=10, type=int, help="Freeze the first N layers (backbone)."
    )
    parser.add_argument("--mosaic", default=0.0, type=float)
    parser.add_argument("--lr0", default=0.001, type=float)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = TrainConfig(
        data_yaml=Path(args.data_yaml),
        base_weights=args.base_weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=Path(args.project),
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        freeze=args.freeze,
        mosaic=args.mosaic,
        lr0=args.lr0,
    )
    best = train_student(config)
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
