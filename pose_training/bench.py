"""Benchmark harness for the portal's two-person pose + tracking pipeline.

The harness separates the expensive part (running the pose model) from the cheap
part (tracking / identity association), so tracking variants can be compared in
seconds on every portal session:

1. ``cache``  -- run the pose model once per camera at a low confidence threshold
   and store every detection (box, keypoints, ReID embedding) in an ``.npz``.
2. ``eval``   -- run named association variants from the cache and score them.

Scores:

- ``two_person_rate``: share of frames where both parent and child are output.
- ``role_acc`` / ``recall``: agreement with the colleague's anchor labels
  (``anchors.json`` written by the portal annotation page). These are the only
  human identity labels available. Variants receive the *fit* half of the
  anchors and are scored on the held-out *test* half.
- ``role_flips_per_min``: how often parent and child boxes swap between
  consecutive frames (a proxy for identity switches).

Example:
    uv run python -m pose_training.bench cache
    uv run python -m pose_training.bench eval --variants deployed botsort_fixed
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "video_inference" / "output"
JOB_ROOT = OUTPUT_ROOT / "portal_jobs"
DEFAULT_CACHE_DIR = REPO_ROOT / "pose_training" / "data" / "bench_cache"
DEFAULT_MODEL = REPO_ROOT / "yolo11l-pose-dense.pt"

Box = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Detection cache
# ---------------------------------------------------------------------------


@dataclass
class DetCache:
    """All detections for one camera, stored as flat arrays.

    Detections of frame ``i`` are rows ``offsets[i]:offsets[i + 1]``.

    Attributes:
        frame_idx: Frame index per frame, shape (F,).
        timestamp_s: Source-video timestamp per frame, shape (F,).
        offsets: Row offsets into the detection arrays, shape (F + 1,).
        boxes: Detection boxes in pixels (x1, y1, x2, y2), shape (N, 4).
        conf: Detection confidence, shape (N,).
        keypoints: COCO-17 keypoints (x, y, conf), shape (N, 17, 3).
        embeddings: L2-normalized appearance embeddings, shape (N, D).
        image_size: Frame (width, height) in pixels.
    """

    frame_idx: np.ndarray
    timestamp_s: np.ndarray
    offsets: np.ndarray
    boxes: np.ndarray
    conf: np.ndarray
    keypoints: np.ndarray
    embeddings: np.ndarray
    image_size: tuple[int, int]

    @property
    def n_frames(self) -> int:
        """Number of frames in the cache."""
        return len(self.frame_idx)

    def frame_dets(self, i: int) -> slice:
        """Return the detection row slice for frame position ``i``."""
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))

    def save(self, path: Path) -> None:
        """Save the cache to a compressed ``.npz`` file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            frame_idx=self.frame_idx,
            timestamp_s=self.timestamp_s,
            offsets=self.offsets,
            boxes=self.boxes,
            conf=self.conf,
            keypoints=self.keypoints,
            embeddings=self.embeddings,
            image_size=np.asarray(self.image_size),
        )

    @classmethod
    def load(cls, path: Path) -> DetCache:
        """Load a cache written by :meth:`save`."""
        data = np.load(path)
        width, height = (int(v) for v in data["image_size"])
        return cls(
            frame_idx=data["frame_idx"],
            timestamp_s=data["timestamp_s"],
            offsets=data["offsets"],
            boxes=data["boxes"],
            conf=data["conf"],
            keypoints=data["keypoints"],
            embeddings=data["embeddings"],
            image_size=(width, height),
        )


# ---------------------------------------------------------------------------
# Anchor ground truth
# ---------------------------------------------------------------------------


@dataclass
class AnchorGold:
    """One human-labeled anchor frame with parent and child boxes."""

    timestamp_s: float
    parent_box: Box
    child_box: Box


def load_anchor_gold(job_dir: Path, camera_id: str) -> list[AnchorGold]:
    """Load the colleague's parent/child anchor labels for one camera.

    Args:
        job_dir: Portal job directory containing ``anchors/<camera_id>/``.
        camera_id: ``camera_a`` or ``camera_b``.

    Returns:
        Non-skipped anchors with parent and child boxes, in file order.
    """
    cam_dir = job_dir / "anchors" / camera_id
    labels_path = cam_dir / "anchors.json"
    manifest_path = cam_dir / "manifest.json"
    if not labels_path.exists() or not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_frame = {int(m["frame_idx"]): m for m in manifest}
    gold: list[AnchorGold] = []
    for entry in json.loads(labels_path.read_text(encoding="utf-8")):
        if entry.get("skipped"):
            continue
        meta = by_frame.get(int(entry["frame_idx"]))
        if meta is None:
            continue
        boxes = {int(t["track_id"]): tuple(t["bbox_xyxy"]) for t in meta["tracks"]}
        parent = boxes.get(int(entry["parent_track_id"]))
        child = boxes.get(int(entry["child_track_id"]))
        if parent is None or child is None:
            continue
        gold.append(AnchorGold(float(meta["timestamp_s"]), parent, child))
    return gold


def split_anchors(
    gold: Sequence[AnchorGold],
) -> tuple[list[AnchorGold], list[AnchorGold]]:
    """Split anchors into fit/test halves, alternating in time order.

    Returns:
        ``(fit, test)``: anchors at even and odd time ranks.
    """
    ordered = sorted(gold, key=lambda g: g.timestamp_s)
    return ordered[0::2], ordered[1::2]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _role_boxes(assign: pd.DataFrame) -> dict[int, dict[str, Box]]:
    out: dict[int, dict[str, Box]] = {}
    for row in assign.itertuples(index=False):
        out.setdefault(int(row.frame_idx), {})[str(row.role)] = (
            float(row.x1),
            float(row.y1),
            float(row.x2),
            float(row.y2),
        )
    return out


def two_person_rate(assign: pd.DataFrame, n_frames: int) -> float:
    """Share of frames where both a parent and a child box are output."""
    if n_frames <= 0 or assign.empty:
        return 0.0
    roles = assign.groupby("frame_idx")["role"].nunique()
    return float((roles >= 2).sum() / n_frames)


def anchor_scores(
    assign: pd.DataFrame,
    gold: Sequence[AnchorGold],
    iou_thresh: float = 0.5,
    time_tol_s: float = 0.05,
) -> dict[str, float]:
    """Score role accuracy and recall against anchor labels.

    Each anchor contributes two gold people (parent and child).

    Args:
        assign: Variant output with frame_idx, timestamp_s, role, x1..y2.
        gold: Anchor labels to score against.
        iou_thresh: IoU needed for an output box to match a gold box.
        time_tol_s: Max timestamp difference to match an anchor frame.

    Returns:
        ``n_anchors``, ``role_acc`` (gold people matched by a box with the
        correct role) and ``recall`` (gold people matched by any output box).
    """
    if not gold:
        return {"n_anchors": 0, "role_acc": float("nan"), "recall": float("nan")}
    times = assign.drop_duplicates("frame_idx")[["frame_idx", "timestamp_s"]]
    t_arr = times["timestamp_s"].to_numpy(dtype=float)
    f_arr = times["frame_idx"].to_numpy()
    boxes = _role_boxes(assign)
    correct = covered = 0
    for g in gold:
        frame_boxes: dict[str, Box] = {}
        if len(t_arr):
            k = int(np.argmin(np.abs(t_arr - g.timestamp_s)))
            if abs(t_arr[k] - g.timestamp_s) <= time_tol_s:
                frame_boxes = boxes.get(int(f_arr[k]), {})
        for role, gbox in (("parent", g.parent_box), ("child", g.child_box)):
            if role in frame_boxes and box_iou(frame_boxes[role], gbox) >= iou_thresh:
                correct += 1
            if any(box_iou(b, gbox) >= iou_thresh for b in frame_boxes.values()):
                covered += 1
    n = 2 * len(gold)
    return {"n_anchors": len(gold), "role_acc": correct / n, "recall": covered / n}


def child_taller_rate(assign: pd.DataFrame, ratio: float = 1.15) -> float:
    """Share of two-person frames where the child's box is clearly taller.

    Parents are usually larger than children in this footage, so a high rate
    over a stretch of frames points to swapped roles (it is not zero even
    when correct: a seated parent next to a standing child).
    """
    if assign.empty:
        return float("nan")
    h = assign.assign(h=assign["y2"] - assign["y1"]).pivot_table(
        index="frame_idx", columns="role", values="h", aggfunc="first"
    )
    if not {"parent", "child"} <= set(h.columns):
        return float("nan")
    both = h.dropna(subset=["parent", "child"])
    if both.empty:
        return float("nan")
    return float((both["child"] > ratio * both["parent"]).mean())


def kept_confident_rate(
    assign: pd.DataFrame,
    reference: pd.DataFrame,
    min_conf: float = 0.5,
    iou: float = 0.5,
) -> float:
    """Share of confident reference detections that the variant still outputs.

    Args:
        assign: Variant output (x1..y2 per role).
        reference: Deployed ``tracks_2d.csv`` (bbox_* and track_confidence).
        min_conf: Only reference boxes at or above this confidence count.
        iou: IoU needed for an output box to cover a reference box.

    Returns:
        Fraction of confident reference boxes covered by any output box.
    """
    ref = reference[reference["track_confidence"] >= min_conf]
    if ref.empty:
        return float("nan")
    out_boxes: dict[int, list[Box]] = {}
    for r in assign.itertuples(index=False):
        out_boxes.setdefault(int(r.frame_idx), []).append((r.x1, r.y1, r.x2, r.y2))
    covered = 0
    for r in ref.itertuples(index=False):
        box = (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2)
        if any(box_iou(box, b) >= iou for b in out_boxes.get(int(r.frame_idx), [])):
            covered += 1
    return covered / len(ref)


def role_flips_per_min(assign: pd.DataFrame, margin: float = 0.2) -> float:
    """Rate of parent/child box swaps between consecutive frames.

    A swap is counted when the cross-role IoU sum beats the same-role IoU sum
    by ``margin``. Only consecutive output frames with both roles count.

    Returns:
        Swaps per minute of footage covered by consecutive two-person frames.
    """
    boxes = _role_boxes(assign)
    times = assign.drop_duplicates("frame_idx").set_index("frame_idx")["timestamp_s"]
    frames = sorted(f for f, b in boxes.items() if len(b) >= 2)
    flips = 0
    covered_s = 0.0
    for prev, cur in zip(frames, frames[1:]):
        if cur - prev != 1:
            continue
        p0, c0 = boxes[prev]["parent"], boxes[prev]["child"]
        p1, c1 = boxes[cur]["parent"], boxes[cur]["child"]
        same = box_iou(p1, p0) + box_iou(c1, c0)
        cross = box_iou(p1, c0) + box_iou(c1, p0)
        if cross > same + margin:
            flips += 1
        covered_s += float(times[cur] - times[prev])
    return flips / (covered_s / 60.0) if covered_s > 0 else 0.0


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------


@dataclass
class BenchCamera:
    """One camera of one portal session available for benchmarking."""

    session_id: str
    camera_id: str
    camera_dir: Path
    job_dir: Path

    @property
    def key(self) -> str:
        """Stable identifier, e.g. ``P003_camera_a``."""
        return f"{self.session_id}_{self.camera_id}"


def discover_cameras(job_root: Path = JOB_ROOT) -> list[BenchCamera]:
    """List cameras whose frames and anchor labels are on disk.

    Uses the most recent job per session, since reruns overwrite the session's
    output directory.
    """
    latest: dict[str, Path] = {}
    for job_dir in sorted(job_root.glob("2*")):
        meta_path = job_dir / "submission_metadata.json"
        if not meta_path.exists():
            continue
        session_id = json.loads(meta_path.read_text(encoding="utf-8"))["session_id"]
        latest[session_id] = job_dir  # sorted by timestamp prefix, last wins
    cameras: list[BenchCamera] = []
    for session_id, job_dir in sorted(latest.items()):
        for cam in ("camera_a", "camera_b"):
            camera_dir = OUTPUT_ROOT / session_id / cam
            if (camera_dir / "frames" / "frame_index.csv").exists() and (
                job_dir / "anchors" / cam / "anchors.json"
            ).exists():
                cameras.append(BenchCamera(session_id, cam, camera_dir, job_dir))
    return cameras


# ---------------------------------------------------------------------------
# Cache building
# ---------------------------------------------------------------------------


def batch_reid_features(
    reid: Any, images: Sequence[np.ndarray], boxes: Sequence[np.ndarray]
) -> np.ndarray:
    """Compute L2-normalized ReID embeddings for all boxes of several frames at once.

    Mirrors boxmot's ``get_crops`` preprocessing but runs one forward pass for
    the whole batch instead of one per frame.

    Args:
        reid: boxmot ReID backend (``ReID(...).model``).
        images: BGR frames.
        boxes: Per-frame (n_i, 4) xyxy boxes.

    Returns:
        (sum n_i, D) embeddings in frame order.
    """
    import cv2
    import torch

    crops = []
    for image, frame_boxes in zip(images, boxes):
        h, w = image.shape[:2]
        for box in frame_boxes:
            x1, y1, x2, y2 = np.round(box).astype(int)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crop = image[y1:y2, x1:x2]
            else:
                crop = np.zeros((*reid.input_shape, 3), np.uint8)
            crop = reid.preprocess_fn(crop, reid.input_shape)
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if not crops:
        return np.zeros((0, 512), np.float32)
    dtype = torch.half if reid.half else torch.float
    with torch.no_grad():
        batch = torch.from_numpy(np.stack(crops)).to(reid.device, dtype=dtype)
        batch = batch.permute(0, 3, 1, 2) / 255.0
        batch = (batch - reid.mean_array) / reid.std_array
        feats = reid.inference_postprocess(
            reid.forward(reid.inference_preprocess(batch))
        )
    feats = np.asarray(feats, dtype=np.float32)
    return feats / np.linalg.norm(feats, axis=-1, keepdims=True)


def build_cache(
    camera_dir: Path,
    model_path: Path = DEFAULT_MODEL,
    conf: float = 0.05,
    imgsz: int = 960,
    batch_size: int = 32,
    device: str = "cuda:0",
) -> DetCache:
    """Run the pose model on every extracted frame and cache all detections.

    Args:
        camera_dir: Portal camera output directory with ``frames/``.
        model_path: Ultralytics pose weights.
        conf: Detection threshold (keep low; variants filter later).
        imgsz: Inference size.
        batch_size: Frames per model call.
        device: Torch device.

    Returns:
        The populated detection cache.
    """
    from concurrent.futures import ThreadPoolExecutor

    import cv2
    from boxmot.reid.core.reid import ReID
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    reid = ReID(path=None, device=device, half=True).model
    index = pd.read_csv(camera_dir / "frames" / "frame_index.csv")
    paths = [camera_dir / "frames" / name for name in index["image_name"]]

    offsets = [0]
    boxes, confs, kpts, embs = [], [], [], []
    image_size = (0, 0)
    chunks = [paths[s : s + batch_size] for s in range(0, len(paths), batch_size)]
    pool = ThreadPoolExecutor(max_workers=8)

    def _read(chunk: list[Path]) -> list[np.ndarray]:
        return list(pool.map(lambda p: cv2.imread(str(p)), chunk))

    pending = pool.submit(_read, chunks[0]) if chunks else None
    for n, _chunk in enumerate(chunks):
        images = pending.result()
        if n + 1 < len(chunks):
            pending = pool.submit(_read, chunks[n + 1])
        results = model(
            images,
            conf=conf,
            iou=0.7,
            imgsz=imgsz,
            device=device,
            half=True,
            verbose=False,
            classes=[0],
        )
        chunk_boxes = []
        for image, result in zip(images, results):
            image_size = (image.shape[1], image.shape[0])
            b = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            c = result.boxes.conf.cpu().numpy().astype(np.float32)
            k = np.concatenate(
                [
                    result.keypoints.xy.cpu().numpy(),
                    result.keypoints.conf.cpu().numpy()[..., None],
                ],
                axis=-1,
            ).astype(np.float32)
            chunk_boxes.append(b)
            boxes.append(b)
            confs.append(c)
            kpts.append(k.reshape(-1, 17, 3))
            offsets.append(offsets[-1] + len(b))
        embs.append(batch_reid_features(reid, images, chunk_boxes))
    pool.shutdown()
    dim = max((e.shape[1] for e in embs if e.size), default=512)
    return DetCache(
        frame_idx=index["frame_idx"].to_numpy(),
        timestamp_s=index["timestamp_s"].to_numpy(dtype=float),
        offsets=np.asarray(offsets),
        boxes=np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
        conf=np.concatenate(confs) if confs else np.zeros(0, np.float32),
        keypoints=np.concatenate(kpts) if kpts else np.zeros((0, 17, 3), np.float32),
        embeddings=np.concatenate([e.reshape(-1, dim) for e in embs]),
        image_size=image_size,
    )


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

VariantFn = Callable[[DetCache, Sequence[AnchorGold]], pd.DataFrame]
VARIANTS: dict[str, VariantFn] = {}


def register(name: str) -> Callable[[VariantFn], VariantFn]:
    """Register an association variant under ``name``."""

    def deco(fn: VariantFn) -> VariantFn:
        VARIANTS[name] = fn
        return fn

    return deco


def evaluate_camera(
    camera: BenchCamera,
    variant: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    """Run one variant on one camera and return its scores."""
    gold = load_anchor_gold(camera.job_dir, camera.camera_id)
    fit, test = split_anchors(gold)
    if variant == "deployed":
        assign = deployed_assignment(camera, fit)
        n_frames = len(pd.read_csv(camera.camera_dir / "frames" / "frame_index.csv"))
    else:
        cache = DetCache.load(cache_dir / f"{camera.key}.npz")
        # Look up through the importable module: when run as __main__, variants
        # register on ``pose_training.bench``, not on this module object.
        import pose_training.bench as registry

        assign = registry.VARIANTS[variant](cache, fit)
        n_frames = cache.n_frames
    scores = anchor_scores(assign, test)
    return {
        "camera": camera.key,
        "variant": variant,
        "two_person_rate": two_person_rate(assign, n_frames),
        "role_acc": scores["role_acc"],
        "recall": scores["recall"],
        "n_test_anchors": scores["n_anchors"],
        "role_flips_per_min": role_flips_per_min(assign),
        "child_taller_rate": child_taller_rate(assign),
        "kept_confident": kept_confident_rate(
            assign, pd.read_csv(camera.camera_dir / "tracks_2d.csv")
        ),
    }


def deployed_assignment(camera: BenchCamera, fit: Sequence[AnchorGold]) -> pd.DataFrame:
    """Re-run the deployed role assignment on the deployed raw tracks.

    Uses ``video_analysis.track_filter`` with only the *fit* anchors, so the
    held-out anchors stay unseen, exactly like the variants.
    """
    from pose_training.assoc_variants import anchors_to_filter_json, run_track_filter

    tracks = pd.read_csv(camera.camera_dir / "tracks_2d.csv")
    anchors = anchors_to_filter_json(tracks, fit)
    return run_track_filter(camera, anchors)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_cache = sub.add_parser("cache", help="Build detection caches.")
    p_cache.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p_cache.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p_cache.add_argument("--only", nargs="*", default=None)
    p_eval = sub.add_parser("eval", help="Score association variants.")
    p_eval.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p_eval.add_argument("--variants", nargs="+", required=True)
    p_eval.add_argument("--only", nargs="*", default=None)
    p_eval.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    cameras = discover_cameras()
    if args.only:
        cameras = [c for c in cameras if c.key in set(args.only)]

    if args.cmd == "cache":
        for cam in cameras:
            out = args.cache_dir / f"{cam.key}.npz"
            if out.exists():
                print(f"[bench] cached already: {cam.key}")
                continue
            cache = build_cache(cam.camera_dir, model_path=args.model)
            cache.save(out)
            print(f"[bench] {cam.key}: {cache.n_frames} frames, {len(cache.conf)} dets")
        return

    import pose_training.assoc_variants  # noqa: F401  (registers variants)

    rows = [
        evaluate_camera(c, v, args.cache_dir) for v in args.variants for c in cameras
    ]
    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)
    summary = df.groupby("variant")[
        ["two_person_rate", "role_acc", "recall", "child_taller_rate", "kept_confident"]
    ].mean()
    print(df.to_string(index=False))
    print("\nMEAN over cameras")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
