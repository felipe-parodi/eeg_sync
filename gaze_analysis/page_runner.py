"""Run PaGE gaze target estimation on video frames with precomputed head boxes.

Drop-in alternative to :mod:`gaze_analysis.gazelle_runner`: it writes the same
``gaze_heatmap.csv`` columns and the same ``gaze_heatmaps.npz`` (64x64 heatmaps
keyed ``f{frame_idx:06d}_t{track_id}``), so synchrony, plotting and gaze
metrics run unchanged.

PaGE (Ye et al., 2026; MIT license) reports lower gaze-point error than
Gaze-LLE on GazeFollow, VideoAttentionTarget and the ChildPlay children's
benchmark. The released checkpoints are self-contained Hugging Face models
loaded with ``trust_remote_code``; both the weights and the remote model code
are pinned to audited revisions below. They load in the project environment
(Python 3.11, transformers 5.x), so no separate interpreter is needed.

Head boxes are normalized by the size of the extracted frames on disk, not by
the session config's ``image_width``/``image_height``, which can disagree with
the frames (portal configs say 854x480 while portal frames are 960x540).
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gaze_analysis.config import (
    add_time_range_args,
    filter_by_time_range,
    load_session_config,
    resolve_time_range,
)
from gaze_analysis.gaze_schema import GAZE_HEATMAP_COLUMNS
from gaze_analysis.gazelle_runner import _find_frame_paths, _resolve_device
from gaze_analysis.head_bbox import extract_head_bboxes

PAGE_MODELS = {
    "page-vits": "Octopus1/page-vits",
    "page-vitsplus": "Octopus1/page-vitsplus",
    "page-vitb": "Octopus1/page-vitb",
    "page-vithplus": "Octopus1/page-vithplus",
}
DEFAULT_PAGE_MODEL = "page-vitb"
# Audited revisions: weights repo (page-vitb) and the shared remote-code repo.
PAGE_WEIGHTS_REVISIONS = {"page-vitb": "b991d8bcb2890a97db611a2080ca250ecd943952"}
PAGE_CODE_REVISION = "7147c67c440a110f050eb1017de50c7a1925c496"

SCENE_SIZE = (512, 512)
HEAD_SIZE = (256, 256)
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class PageConfig:
    """Configuration for PaGE inference on one camera."""

    camera_dir: str
    session_config_path: str
    camera_id: str = "camera_a"
    device: str = "auto"
    model_name: str = DEFAULT_PAGE_MODEL
    batch_size: int = 16
    half: bool = True
    max_frames: int = 0
    start_s: float | None = None
    end_s: float | None = None
    time_block: str | None = None
    pose_input: str = "pose_3d.csv"
    tracks_input: str = "tracks_2d.csv"
    output_dir: str | None = None
    output_csv: str = "gaze_heatmap.csv"
    output_npz: str = "gaze_heatmaps.npz"


def pixel_head_box(
    norm_box: Sequence[float], width: int, height: int
) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float]]]:
    """Convert a normalized head box to the pixel crop and box PaGE expects.

    Mirrors ``parse_bbox`` in PaGE's ``scripts/inference.py``: round and clip
    to the image, then re-normalize from the clipped pixels.

    Args:
        norm_box: (x1, y1, x2, y2) normalized to [0, 1].
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        ``(pixel_xyxy, normalized_xyxy)``, or None if the box lies outside the
        image or is empty after clipping.
    """
    if norm_box[0] >= 1 or norm_box[1] >= 1 or norm_box[2] <= 0 or norm_box[3] <= 0:
        return None
    x1 = max(0, min(width - 1, int(round(norm_box[0] * width))))
    y1 = max(0, min(height - 1, int(round(norm_box[1] * height))))
    x2 = max(0, min(width, int(round(norm_box[2] * width))))
    y2 = max(0, min(height, int(round(norm_box[3] * height))))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2), (x1 / width, y1 / height, x2 / width, y2 / height)


def heatmap_peak(heatmap: np.ndarray) -> Tuple[float, float, float]:
    """Return the normalized argmax location and value of a 64x64 heatmap.

    Uses the same convention as the Gaze-LLE runner (cell index / 63) so the
    two runners' gaze points are directly comparable.

    Args:
        heatmap: 2-D heatmap.

    Returns:
        ``(gaze_peak_x, gaze_peak_y, peak_value)``.
    """
    h, w = heatmap.shape
    peak_y, peak_x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return (
        float(peak_x) / (w - 1),
        float(peak_y) / (h - 1),
        float(heatmap[peak_y, peak_x]),
    )


def _normalize(rgb: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB -> CHW float32 ImageNet-normalized."""
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - IMAGE_MEAN) / IMAGE_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def prepare_frame(
    frame_path: Path, norm_boxes: Sequence[Sequence[float]]
) -> Tuple[np.ndarray, List[np.ndarray], List[Optional[Tuple[float, ...]]]]:
    """Load one frame and build PaGE's scene tensor and per-head crops.

    Matches ``PaGEImageProcessor`` (PIL bilinear resize, ImageNet stats).

    Args:
        frame_path: Frame image path.
        norm_boxes: Normalized head boxes for the people in this frame.

    Returns:
        ``(scene_chw, head_chw_list, normalized_boxes)``; a box is None when it
        is empty after clipping (its crop is then the whole frame and its row
        is dropped downstream).
    """
    from PIL import Image

    scene = Image.open(frame_path).convert("RGB")
    width, height = scene.size
    scene_arr = _normalize(
        np.asarray(scene.resize((SCENE_SIZE[1], SCENE_SIZE[0]), Image.BILINEAR))
    )
    heads: List[np.ndarray] = []
    boxes: List[Optional[Tuple[float, ...]]] = []
    for norm_box in norm_boxes:
        parsed = pixel_head_box(norm_box, width, height)
        if parsed is None:
            crop = scene
            boxes.append(None)
        else:
            crop = scene.crop(parsed[0])
            boxes.append(parsed[1])
        heads.append(
            _normalize(
                np.asarray(crop.resize((HEAD_SIZE[1], HEAD_SIZE[0]), Image.BILINEAR))
            )
        )
    return scene_arr, heads, boxes


def load_page_model(model_name: str, device: str) -> Any:
    """Load a PaGE checkpoint at its pinned revision.

    Args:
        model_name: Key of :data:`PAGE_MODELS` or a Hugging Face repo id.
        device: Torch device.

    Returns:
        The model in eval mode on ``device``.
    """
    import contextlib
    import io

    from transformers import AutoModel
    from transformers.utils import logging as hf_logging

    repo = PAGE_MODELS.get(model_name, model_name)
    revision = PAGE_WEIGHTS_REVISIONS.get(model_name)
    old = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        # PaGE prints expected backbone key-remap notices while loading.
        with contextlib.redirect_stdout(io.StringIO()):
            model = AutoModel.from_pretrained(
                repo,
                revision=revision,
                code_revision=PAGE_CODE_REVISION,
                trust_remote_code=True,
            )
    finally:
        hf_logging.set_verbosity(old)
    return model.eval().to(device)


def _frame_size(frame_paths: Dict[int, Path]) -> Tuple[int, int]:
    """Return (width, height) of the extracted frames."""
    from PIL import Image

    first = next(iter(frame_paths.values()))
    with Image.open(first) as img:
        return img.size


def run_page_inference(config: PageConfig) -> Dict[str, Any]:
    """Run PaGE gaze estimation on all frames for one camera.

    Args:
        config: PaGE inference configuration.

    Returns:
        Summary dict with row counts and settings.
    """
    import torch

    camera_dir = Path(config.camera_dir)
    out_dir = Path(config.output_dir) if config.output_dir else camera_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    session_config = load_session_config(config.session_config_path)
    session_config.get_camera_mapping(config.camera_id)

    pose_df = pd.read_csv(camera_dir / config.pose_input)
    tracks_df = pd.read_csv(camera_dir / config.tracks_input)
    frame_paths = _find_frame_paths(camera_dir, session_config.frame_rate)
    if not frame_paths:
        raise FileNotFoundError(f"No frames/frame_index.csv under {camera_dir}")
    width, height = _frame_size(frame_paths)

    head_df = extract_head_bboxes(
        pose_df, tracks_df, image_width=width, image_height=height
    )
    head_df = head_df[head_df["frame_idx"].isin(set(frame_paths))].copy()
    t_start, t_end = resolve_time_range(
        session_config, config.start_s, config.end_s, config.time_block
    )
    head_df = filter_by_time_range(head_df, t_start, t_end)
    if config.max_frames > 0:
        keep = sorted(head_df["frame_idx"].unique())[: config.max_frames]
        head_df = head_df[head_df["frame_idx"].isin(keep)].copy()

    device = _resolve_device(config.device)
    use_half = config.half and not str(device).startswith("cpu")
    print(f"Loading PaGE model '{config.model_name}' on {device}...")
    model = load_page_model(config.model_name, device)

    groups = {int(k): g for k, g in head_df.groupby("frame_idx", sort=True)}
    frame_indices = sorted(groups)
    batch_size = max(1, config.batch_size)
    pool = ThreadPoolExecutor(max_workers=8)
    box_cols = ["head_x1", "head_y1", "head_x2", "head_y2"]

    def _prep(batch: List[int]) -> list:
        return list(
            pool.map(
                lambda f: prepare_frame(frame_paths[f], groups[f][box_cols].to_numpy()),
                batch,
            )
        )

    batches = [
        frame_indices[s : s + batch_size]
        for s in range(0, len(frame_indices), batch_size)
    ]
    rows: List[dict] = []
    heatmaps: List[np.ndarray] = []
    keys: List[str] = []
    print(f"Running PaGE on {len(frame_indices)} frames (batch_size={batch_size})...")
    started = time.perf_counter()
    pending = pool.submit(_prep, batches[0]) if batches else None
    for n, batch in enumerate(batches):
        prepared = pending.result()
        if n + 1 < len(batches):
            pending = pool.submit(_prep, batches[n + 1])
        # Rows whose head box is empty after clipping are skipped entirely.
        kept = [
            [i for i, b in enumerate(boxes) if b is not None]
            for _, _, boxes in prepared
        ]
        scenes = torch.from_numpy(np.stack([p[0] for p in prepared]))
        heads = (
            torch.from_numpy(
                np.stack([p[1][i] for p, idx in zip(prepared, kept) for i in idx])
            )
            if any(kept)
            else None
        )
        if heads is None:
            continue
        inputs = {
            "images": scenes.to(device),
            "head_images": [heads.to(device)],
            "bboxes": [[p[2][i] for i in idx] for p, idx in zip(prepared, kept)],
        }
        # Frames with no usable head still need a placeholder list entry.
        if any(len(idx) == 0 for idx in kept):
            keep_frames = [j for j, idx in enumerate(kept) if idx]
            inputs["images"] = inputs["images"][keep_frames]
            inputs["bboxes"] = [inputs["bboxes"][j] for j in keep_frames]
        else:
            keep_frames = list(range(len(batch)))
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_half),
        ):
            out = model(inputs)
        for out_i, j in enumerate(keep_frames):
            fidx = batch[j]
            meta = groups[fidx].iloc[kept[j]]
            hms = out["heatmap"][out_i].float().cpu().numpy()
            inouts = (
                out["inout"][out_i].float().cpu().numpy()
                if out.get("inout") is not None
                else np.zeros(len(meta))
            )
            for k, row in enumerate(meta.itertuples(index=False)):
                gx, gy, peak = heatmap_peak(hms[k])
                rows.append(
                    {
                        "frame_idx": int(row.frame_idx),
                        "timestamp_s": float(row.timestamp_s),
                        "track_id": int(row.track_id),
                        "gaze_peak_x": gx,
                        "gaze_peak_y": gy,
                        "gaze_peak_value": peak,
                        "inout_score": float(inouts[k]),
                        "head_source": row.head_source,
                    }
                )
                heatmaps.append(hms[k].astype(np.float32))
                keys.append(f"f{int(row.frame_idx):06d}_t{int(row.track_id)}")
        if (n + 1) % 20 == 0 or n + 1 == len(batches):
            done = min((n + 1) * batch_size, len(frame_indices))
            print(f"  Processed {done}/{len(frame_indices)} frames", flush=True)
    pool.shutdown()
    elapsed = time.perf_counter() - started

    gaze_df = pd.DataFrame(rows, columns=GAZE_HEATMAP_COLUMNS)
    gaze_csv_path = out_dir / config.output_csv
    gaze_df.to_csv(gaze_csv_path, index=False)
    npz_size_mb = 0.0
    if heatmaps:
        npz_path = out_dir / config.output_npz
        np.savez_compressed(npz_path, heatmaps=np.stack(heatmaps), keys=np.array(keys))
        npz_size_mb = npz_path.stat().st_size / (1024 * 1024)

    summary = {
        "model_name": config.model_name,
        "model_revision": PAGE_WEIGHTS_REVISIONS.get(config.model_name),
        "code_revision": PAGE_CODE_REVISION,
        "device": device,
        "half": use_half,
        "batch_size": batch_size,
        "frame_size": [width, height],
        "total_frames": len(frame_indices),
        "total_gaze_rows": len(gaze_df),
        "inference_seconds": round(elapsed, 1),
        "heatmap_npz_size_mb": round(npz_size_mb, 1),
        "output_csv": str(gaze_csv_path),
    }
    (out_dir / "gaze_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(gaze_df)} gaze rows to {gaze_csv_path} in {elapsed:.0f} s")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for PaGE gaze inference."""
    parser = argparse.ArgumentParser(
        prog="gaze-infer-page",
        description="Run PaGE gaze estimation on video frames.",
    )
    parser.add_argument("--camera-dir", required=True, type=str)
    parser.add_argument("--session-config", required=True, type=str)
    parser.add_argument("--camera-id", default="camera_a", type=str)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--model-name", default=DEFAULT_PAGE_MODEL, type=str)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--no-half", action="store_true", help="Run in float32.")
    parser.add_argument("--max-frames", default=0, type=int)
    parser.add_argument("--pose-input", default="pose_3d.csv", type=str)
    parser.add_argument("--tracks-input", default="tracks_2d.csv", type=str)
    parser.add_argument(
        "--output-dir", default=None, type=str, help="Default: the camera directory."
    )
    parser.add_argument("--output-csv", default="gaze_heatmap.csv", type=str)
    parser.add_argument("--output-npz", default="gaze_heatmaps.npz", type=str)
    add_time_range_args(parser)
    return parser


def main() -> None:
    """CLI entry point for PaGE gaze inference."""
    args = build_arg_parser().parse_args()
    run_page_inference(
        PageConfig(
            camera_dir=args.camera_dir,
            session_config_path=args.session_config,
            camera_id=args.camera_id,
            device=args.device,
            model_name=args.model_name,
            batch_size=args.batch_size,
            half=not args.no_half,
            max_frames=args.max_frames,
            start_s=args.start_s,
            end_s=args.end_s,
            time_block=args.time_block,
            pose_input=args.pose_input,
            tracks_input=args.tracks_input,
            output_dir=args.output_dir,
            output_csv=args.output_csv,
            output_npz=args.output_npz,
        )
    )


if __name__ == "__main__":
    main()
