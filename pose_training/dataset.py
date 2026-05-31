"""QC, filtering, and YOLO pose dataset export for teacher pseudo-labels.

Turns the raw teacher predictions into a clean YOLO pose training set. The
filters are purely automatic (no manual correction): low-confidence instances
are dropped, low-confidence keypoints are demoted to "not labeled" so they do
not contribute to the keypoint loss, and degenerate boxes are removed. A
confidence-distribution figure is rendered so the operator can see what the
thresholds keep and drop.

Note on temporal filtering: frames are sampled ~4 s apart, so genuine motion
dominates between neighbours and frame-to-frame "jump" detection would flag
real movement as error. We therefore rely on the teacher's per-keypoint
confidence (which already encodes its uncertainty on hard joints) plus
geometry, not temporal consistency, at this sampling rate.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .coco import FLIP_IDX, KEYPOINT_NAMES, NUM_KEYPOINTS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "pose_training" / "data" / "pseudo_labels_raw.json"
DEFAULT_DATASET_ROOT = REPO_ROOT / "pose_training" / "data" / "yolo"
DEFAULT_PLOT_DIR = REPO_ROOT / "pose_training" / "data" / "qc"


@dataclass
class FilterConfig:
    """Thresholds for automatic pseudo-label filtering."""

    instance_conf_thresh: float = 0.5
    kpt_conf_thresh: float = 0.3
    min_box_px: float = 20.0
    # Clean-sweep: keep an instance only if the teacher confidently localized at
    # least this many of the 17 keypoints. Partial/occluded people with few
    # reliable joints are weak, noisy training samples, so drop them entirely.
    min_visible_kpts: int = 10
    # Denoise filters. A keypoint that lands well outside the person box is a
    # teacher mislocalization (often snapped to background clutter); demote it.
    # Boxes that are implausibly tiny, near-full-frame, or extremely elongated
    # are dropped as spurious detections.
    kpt_bbox_margin_frac: float = 0.2
    min_box_area_frac: float = 0.004
    max_box_area_frac: float = 0.97
    max_aspect_ratio: float = 4.0


def load_raw(path: Path) -> Dict[str, Any]:
    """Load the raw teacher pseudo-label JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def filter_frame(frame: Dict[str, Any], cfg: FilterConfig) -> Dict[str, Any]:
    """Apply confidence, visibility, and geometry filters to one frame.

    Args:
        frame: One raw pseudo-label frame record.
        cfg: Filter thresholds.

    Returns:
        A copy of the frame whose ``persons`` only contains instances that pass
        every filter, with each keypoint marked visible (v=2) or unlabeled
        (v=0) and unconfident keypoints zeroed.
    """
    width = float(frame["width"])
    height = float(frame["height"])
    kept: List[Dict[str, Any]] = []
    for person in frame["persons"]:
        if person["box_conf"] < cfg.instance_conf_thresh:
            continue
        x1, y1, x2, y2 = person["bbox_xyxy"]
        x1, x2 = max(0.0, min(x1, x2)), min(width, max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(height, max(y1, y2))
        box_w, box_h = x2 - x1, y2 - y1
        if box_w < cfg.min_box_px or box_h < cfg.min_box_px:
            continue
        area_frac = (box_w * box_h) / (width * height)
        if not cfg.min_box_area_frac <= area_frac <= cfg.max_box_area_frac:
            continue
        aspect = max(box_w / box_h, box_h / box_w)
        if aspect > cfg.max_aspect_ratio:
            continue
        # Person box expanded by a margin; keypoints outside it are teacher errors.
        margin_x, margin_y = (
            cfg.kpt_bbox_margin_frac * box_w,
            cfg.kpt_bbox_margin_frac * box_h,
        )
        in_x1, in_x2 = x1 - margin_x, x2 + margin_x
        in_y1, in_y2 = y1 - margin_y, y2 + margin_y
        keypoints: List[List[float]] = []
        visible = 0
        for x, y, conf in person["keypoints"]:
            inside = in_x1 <= x <= in_x2 and in_y1 <= y <= in_y2
            if conf >= cfg.kpt_conf_thresh and inside:
                keypoints.append([float(x), float(y), 2.0])
                visible += 1
            else:
                keypoints.append([0.0, 0.0, 0.0])
        if visible < cfg.min_visible_kpts:
            continue
        kept.append(
            {
                "bbox_xyxy": [x1, y1, x2, y2],
                "box_conf": float(person["box_conf"]),
                "keypoints": keypoints,
            }
        )
    return {**frame, "persons": kept}


def to_yolo_label_lines(frame: Dict[str, Any]) -> List[str]:
    """Render a filtered frame's persons as YOLO pose label lines.

    Format per line: ``class cx cy w h (px py v) x 17``, all normalized to
    image size. Class is always 0 (person).
    """
    width = float(frame["width"])
    height = float(frame["height"])
    lines: List[str] = []
    for person in frame["persons"]:
        x1, y1, x2, y2 = person["bbox_xyxy"]
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        values = [0, _clamp01(cx), _clamp01(cy), _clamp01(bw), _clamp01(bh)]
        fields = [str(values[0])] + [f"{v:.6f}" for v in values[1:]]
        for x, y, v in person["keypoints"]:
            if v > 0:
                fields += [
                    f"{_clamp01(x / width):.6f}",
                    f"{_clamp01(y / height):.6f}",
                    str(int(v)),
                ]
            else:
                fields += ["0.000000", "0.000000", "0"]
        lines.append(" ".join(fields))
    return lines


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def export_dataset(
    payload: Dict[str, Any], dataset_root: Path, cfg: FilterConfig
) -> Dict[str, Dict[str, int]]:
    """Filter every frame and write a YOLO pose dataset on disk.

    Creates ``images/{split}`` and ``labels/{split}`` plus ``data.yaml``. The
    test split is exported too (for held-out evaluation in Phase 5) but is not
    referenced as ``val`` in data.yaml.

    Args:
        payload: Raw teacher pseudo-label payload.
        dataset_root: Output dataset root directory.
        cfg: Filter thresholds.

    Returns:
        Per-split counts of frames and instances written.
    """
    dataset_root = Path(dataset_root)
    summary: Dict[str, Dict[str, int]] = {
        split: {"frames": 0, "instances": 0} for split in ("train", "val", "test")
    }
    for split in ("train", "val", "test"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for frame in payload["frames"]:
        split = frame.get("split", "train")
        if split not in summary:
            continue
        filtered = filter_frame(frame, cfg)
        if not filtered["persons"]:
            continue
        source_image = Path(frame["frame_path"])
        stem = source_image.stem
        dest_image = dataset_root / "images" / split / source_image.name
        shutil.copyfile(source_image, dest_image)
        label_path = dataset_root / "labels" / split / f"{stem}.txt"
        label_path.write_text(
            "\n".join(to_yolo_label_lines(filtered)) + "\n", encoding="utf-8"
        )
        summary[split]["frames"] += 1
        summary[split]["instances"] += len(filtered["persons"])

    _write_data_yaml(dataset_root)
    return summary


def _write_data_yaml(dataset_root: Path) -> None:
    """Write data.yaml describing the COCO-17 pose dataset for Ultralytics."""
    lines = [
        f"path: {dataset_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"kpt_shape: [{NUM_KEYPOINTS}, 3]",
        f"flip_idx: {FLIP_IDX}",
        "",
        "names:",
        "  0: person",
        "",
        "# Keypoint order (COCO-17): " + ", ".join(KEYPOINT_NAMES),
    ]
    (dataset_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_confidences(payload: Dict[str, Any]) -> Dict[str, List[float]]:
    """Gather instance and per-keypoint confidences for plotting/QC."""
    instance_conf: List[float] = []
    head_conf: List[float] = []
    limb_conf: List[float] = []
    for frame in payload["frames"]:
        for person in frame["persons"]:
            instance_conf.append(float(person["box_conf"]))
            for index, (_, _, conf) in enumerate(person["keypoints"]):
                if index < 5:
                    head_conf.append(float(conf))
                else:
                    limb_conf.append(float(conf))
    return {
        "instance_conf": instance_conf,
        "head_kpt_conf": head_conf,
        "limb_kpt_conf": limb_conf,
    }


def plot_confidence_distributions(
    payload: Dict[str, Any], cfg: FilterConfig, out_dir: Path
) -> Path:
    """Render the teacher confidence distributions to PNG and SVG.

    Returns:
        Path to the written PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    confidences = collect_confidences(payload)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(confidences["instance_conf"], bins=40, range=(0, 1), color="#1f77b4")
    axes[0].axvline(
        cfg.instance_conf_thresh, color="#d62728", linestyle="--", linewidth=2
    )
    axes[0].set_title("Teacher instance (box) confidence")
    axes[0].set_xlabel("confidence")
    axes[0].set_ylabel("instances")

    axes[1].hist(
        [confidences["head_kpt_conf"], confidences["limb_kpt_conf"]],
        bins=40,
        range=(0, 1),
        label=["head (kp 0-4)", "limbs (kp 5-16)"],
        color=["#117733", "#882255"],
    )
    axes[1].axvline(cfg.kpt_conf_thresh, color="#d62728", linestyle="--", linewidth=2)
    axes[1].set_title("Teacher per-keypoint confidence")
    axes[1].set_xlabel("confidence")
    axes[1].set_ylabel("keypoints")
    axes[1].legend()

    fig.tight_layout()
    png_path = out_dir / "teacher_confidence_distribution.png"
    svg_path = out_dir / "teacher_confidence_distribution.svg"
    fig.savefig(png_path)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path


def summarize_export(summary: Dict[str, Dict[str, int]]) -> str:
    """Render the export summary as a short human-readable string."""
    lines = ["Exported YOLO pose dataset:"]
    for split in ("train", "val", "test"):
        counts = summary.get(split, {"frames": 0, "instances": 0})
        lines.append(
            f"  {split}: {counts['frames']} frames, {counts['instances']} instances"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter teacher pseudo-labels and export a YOLO pose dataset."
    )
    parser.add_argument("--raw", default=str(DEFAULT_RAW), type=str)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), type=str)
    parser.add_argument("--plot-dir", default=str(DEFAULT_PLOT_DIR), type=str)
    parser.add_argument("--instance-conf-thresh", default=0.5, type=float)
    parser.add_argument("--kpt-conf-thresh", default=0.3, type=float)
    parser.add_argument("--min-box-px", default=20.0, type=float)
    parser.add_argument("--min-visible-kpts", default=10, type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = FilterConfig(
        instance_conf_thresh=args.instance_conf_thresh,
        kpt_conf_thresh=args.kpt_conf_thresh,
        min_box_px=args.min_box_px,
        min_visible_kpts=args.min_visible_kpts,
    )
    payload = load_raw(Path(args.raw))
    png_path = plot_confidence_distributions(payload, cfg, Path(args.plot_dir))
    print(f"Wrote confidence figure: {png_path}")
    summary = export_dataset(payload, Path(args.dataset_root), cfg)
    print(summarize_export(summary))


if __name__ == "__main__":
    main()
