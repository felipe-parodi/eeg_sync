"""Evaluate the trained student against the current portal model.

Produces the three acceptance artifacts: (1) pose metrics on the held-out
test participants via Ultralytics ``val``, (2) a confidence-distribution
comparison of the old vs new model on held-out frames, and (3) side-by-side
overlay clips for human review. Metrics on the test split measure agreement
with the teacher on unseen participants, not ground-truth correctness; the
human overlay review is the real gate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from video_inference.weights import resolve_weights

from .coco import SKELETON_EDGES
from .pseudo_label import _result_to_arrays, extract_persons

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = REPO_ROOT / "pose_training" / "data" / "yolo" / "data.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "pose_training" / "data" / "eval"
STOCK_WEIGHTS = "yolo11l-pose.pt"


@dataclass
class EvalConfig:
    """Configuration for the evaluation stage."""

    new_weights: str
    old_weights: str = STOCK_WEIGHTS
    data_yaml: Path = DEFAULT_DATA_YAML
    out_dir: Path = DEFAULT_OUT_DIR
    split: str = "test"
    imgsz: int = 960
    device: str = "0"
    conf: float = 0.25
    max_persons: int = 6


def collect_pred_confidences(
    results: List[Any], max_persons: int
) -> Dict[str, List[float]]:
    """Collect instance and head/limb keypoint confidences from predictions."""
    instance_conf: List[float] = []
    head_conf: List[float] = []
    limb_conf: List[float] = []
    for result in results:
        boxes_xyxy, boxes_conf, kpts_xy, kpts_conf = _result_to_arrays(result)
        persons = extract_persons(
            boxes_xyxy, boxes_conf, kpts_xy, kpts_conf, max_persons
        )
        for person in persons:
            instance_conf.append(person["box_conf"])
            for index, (_, _, conf) in enumerate(person["keypoints"]):
                (head_conf if index < 5 else limb_conf).append(conf)
    return {
        "instance_conf": instance_conf,
        "head_kpt_conf": head_conf,
        "limb_kpt_conf": limb_conf,
    }


def model_pred_confidences(
    weights: str,
    frame_paths: List[str],
    imgsz: int,
    device: str,
    conf: float,
    max_persons: int,
    batch_size: int = 16,
    model: Optional[Any] = None,
) -> Dict[str, List[float]]:
    """Run a model over frames and collect its prediction confidences."""
    if model is None:
        from ultralytics import YOLO

        model = YOLO(resolve_weights(weights))
    collected: Dict[str, List[float]] = {
        "instance_conf": [],
        "head_kpt_conf": [],
        "limb_kpt_conf": [],
    }
    for start in range(0, len(frame_paths), batch_size):
        chunk = frame_paths[start : start + batch_size]
        results = model(chunk, imgsz=imgsz, conf=conf, device=device, verbose=False)
        stats = collect_pred_confidences(results, max_persons)
        for key in collected:
            collected[key].extend(stats[key])
    return collected


def validate_on_split(
    weights: str,
    data_yaml: Path,
    split: str,
    imgsz: int,
    device: str,
    model: Optional[Any] = None,
) -> Dict[str, float]:
    """Run Ultralytics val on a split and return key pose/box metrics."""
    if model is None:
        from ultralytics import YOLO

        model = YOLO(resolve_weights(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    pose = getattr(metrics, "pose", None)
    box = getattr(metrics, "box", None)
    return {
        "pose_map50_95": float(getattr(pose, "map", float("nan"))),
        "pose_map50": float(getattr(pose, "map50", float("nan"))),
        "box_map50_95": float(getattr(box, "map", float("nan"))),
        "box_map50": float(getattr(box, "map50", float("nan"))),
    }


def _hstack_labeled(
    left: np.ndarray, right: np.ndarray, left_label: str, right_label: str
) -> np.ndarray:
    """Stack two frames horizontally with a labeled banner on top."""
    height = min(left.shape[0], right.shape[0])

    def _fit(image: np.ndarray) -> np.ndarray:
        scale = height / image.shape[0]
        return cv2.resize(image, (int(image.shape[1] * scale), height))

    left_fit, right_fit = _fit(left), _fit(right)
    body = np.hstack([left_fit, right_fit])
    banner = np.full((34, body.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(
        banner,
        left_label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        banner,
        right_label,
        (left_fit.shape[1] + 10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return np.vstack([banner, body])


def _draw_pose(image: np.ndarray, persons: List[Dict[str, Any]]) -> np.ndarray:
    """Draw boxes and COCO-17 skeletons for a list of person records."""
    colors = [(0, 128, 255), (255, 160, 0), (0, 220, 0), (180, 0, 255)]
    out = image.copy()
    for index, person in enumerate(persons):
        color = colors[index % len(colors)]
        x1, y1, x2, y2 = [int(v) for v in person["bbox_xyxy"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        kpts = person["keypoints"]
        for x, y, conf in kpts:
            if conf >= 0.3:
                cv2.circle(out, (int(x), int(y)), 3, color, -1)
        for a, b in SKELETON_EDGES:
            if kpts[a][2] >= 0.3 and kpts[b][2] >= 0.3:
                cv2.line(
                    out,
                    (int(kpts[a][0]), int(kpts[a][1])),
                    (int(kpts[b][0]), int(kpts[b][1])),
                    color,
                    2,
                )
    return out


def render_overlay_comparison(
    config: EvalConfig,
    video_path: Path,
    start_s: float,
    seconds: float,
    out_path: Path,
    out_fps: float = 6.0,
    old_model: Optional[Any] = None,
    new_model: Optional[Any] = None,
) -> Path:
    """Render an old-vs-new side-by-side overlay clip from a held-out video.

    Returns:
        Path to the written MP4.
    """
    if old_model is None or new_model is None:
        from ultralytics import YOLO

        old_model = old_model or YOLO(resolve_weights(config.old_weights))
        new_model = new_model or YOLO(resolve_weights(config.new_weights))

    capture = cv2.VideoCapture(str(video_path))
    src_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / out_fps)))
    start_frame = int(start_s * src_fps)
    end_frame = int((start_s + seconds) * src_fps)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: Optional[cv2.VideoWriter] = None
    frame_idx = start_frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while frame_idx < end_frame:
        ok, frame = capture.read()
        if not ok:
            break
        if (frame_idx - start_frame) % step == 0:
            old_persons = _predict_persons(old_model, frame, config)
            new_persons = _predict_persons(new_model, frame, config)
            panel = _hstack_labeled(
                _draw_pose(frame, old_persons),
                _draw_pose(frame, new_persons),
                "OLD (stock yolo11l-pose)",
                "NEW (trained student)",
            )
            if writer is None:
                height, width = panel.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    out_fps,
                    (width, height),
                )
            writer.write(panel)
        frame_idx += 1
    capture.release()
    if writer is not None:
        writer.release()
    return out_path


def _predict_persons(model: Any, frame: np.ndarray, config: EvalConfig):
    """Predict persons on one BGR frame and return person records."""
    results = model(
        frame,
        imgsz=config.imgsz,
        conf=config.conf,
        device=config.device,
        verbose=False,
    )
    boxes_xyxy, boxes_conf, kpts_xy, kpts_conf = _result_to_arrays(results[0])
    return extract_persons(
        boxes_xyxy, boxes_conf, kpts_xy, kpts_conf, config.max_persons
    )


def plot_confidence_comparison(
    old_stats: Dict[str, List[float]],
    new_stats: Dict[str, List[float]],
    out_dir: Path,
) -> Path:
    """Plot old-vs-new instance and keypoint confidence distributions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 12, "axes.labelsize": 14, "figure.dpi": 300, "savefig.dpi": 300}
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(
        [old_stats["instance_conf"], new_stats["instance_conf"]],
        bins=30,
        range=(0, 1),
        label=["old", "new"],
        color=["#888888", "#1f77b4"],
    )
    axes[0].set_title("Instance confidence (held-out)")
    axes[0].set_xlabel("confidence")
    axes[0].legend()
    old_kpt = old_stats["head_kpt_conf"] + old_stats["limb_kpt_conf"]
    new_kpt = new_stats["head_kpt_conf"] + new_stats["limb_kpt_conf"]
    axes[1].hist(
        [old_kpt, new_kpt],
        bins=30,
        range=(0, 1),
        label=["old", "new"],
        color=["#888888", "#117733"],
    )
    axes[1].set_title("Keypoint confidence (held-out)")
    axes[1].set_xlabel("confidence")
    axes[1].legend()
    fig.tight_layout()
    png_path = out_dir / "confidence_comparison.png"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(out_dir / "confidence_comparison.svg", bbox_inches="tight")
    plt.close(fig)
    return png_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained student vs the stock portal model."
    )
    parser.add_argument("--new-weights", required=True, type=str)
    parser.add_argument("--old-weights", default=STOCK_WEIGHTS, type=str)
    parser.add_argument("--data-yaml", default=str(DEFAULT_DATA_YAML), type=str)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--imgsz", default=960, type=int)
    parser.add_argument("--device", default="0", type=str)
    parser.add_argument(
        "--overlay-video",
        action="append",
        default=None,
        type=str,
        help="Held-out video to render an old-vs-new overlay clip from. "
        "Repeatable for multiple clips.",
    )
    parser.add_argument(
        "--overlay-start-frac",
        default=0.4,
        type=float,
        help="Fraction into each video to start the overlay clip.",
    )
    parser.add_argument("--overlay-seconds", default=20.0, type=float)
    parser.add_argument(
        "--no-confidence-plot",
        action="store_true",
        help="Skip the old-vs-new confidence-distribution plot.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = EvalConfig(
        new_weights=args.new_weights,
        old_weights=args.old_weights,
        data_yaml=Path(args.data_yaml),
        out_dir=Path(args.out_dir),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "old": validate_on_split(
            config.old_weights,
            config.data_yaml,
            config.split,
            config.imgsz,
            config.device,
        ),
        "new": validate_on_split(
            config.new_weights,
            config.data_yaml,
            config.split,
            config.imgsz,
            config.device,
        ),
    }
    (config.out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("Metrics (test split, agreement with teacher on held-out participants):")
    print(json.dumps(metrics, indent=2))

    if not args.no_confidence_plot:
        test_images = sorted(
            (config.data_yaml.parent / "images" / config.split).glob("*.jpg")
        )
        frame_paths = [str(path) for path in test_images]
        if frame_paths:
            old_stats = model_pred_confidences(
                config.old_weights,
                frame_paths,
                config.imgsz,
                config.device,
                config.conf,
                config.max_persons,
            )
            new_stats = model_pred_confidences(
                config.new_weights,
                frame_paths,
                config.imgsz,
                config.device,
                config.conf,
                config.max_persons,
            )
            png = plot_confidence_comparison(old_stats, new_stats, config.out_dir)
            print(f"Wrote confidence comparison: {png}")

    for video in args.overlay_video or []:
        video_path = Path(video)
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        src_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        capture.release()
        start_s = (frame_count / src_fps) * args.overlay_start_frac
        out_clip = config.out_dir / f"overlay_{video_path.stem}.mp4"
        render_overlay_comparison(
            config, video_path, start_s, args.overlay_seconds, out_clip
        )
        print(f"Wrote overlay clip: {out_clip}")


if __name__ == "__main__":
    main()
