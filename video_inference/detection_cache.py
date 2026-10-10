"""Per-camera cache of every person detection (box, keypoints, appearance).

Offline identity association (:mod:`video_inference.offline_assoc`) needs all
detections, including faint ones, plus an appearance embedding for each. The
pose model runs once per camera and writes this cache; association can then be
re-run cheaply (for example after the colleague labels anchor frames).
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Keep faint detections: association decides which ones are real people.
DEFAULT_CACHE_CONF = 0.05


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

    def to_frames(self, conf_min: float = 0.0) -> list:
        """Convert to per-frame ``offline_assoc.FrameDetections``.

        Args:
            conf_min: Drop detections below this confidence. Keep the default
                (0) when detection indices must line up with the cache.

        Returns:
            One ``FrameDetections`` per cached frame.
        """
        from video_inference.offline_assoc import FrameDetections

        frames = []
        for i in range(self.n_frames):
            sl = self.frame_dets(i)
            keep = np.arange(sl.start, sl.stop)
            keep = keep[self.conf[keep] >= conf_min]
            frames.append(
                FrameDetections(
                    frame_idx=int(self.frame_idx[i]),
                    timestamp_s=float(self.timestamp_s[i]),
                    boxes=self.boxes[keep],
                    conf=self.conf[keep],
                    keypoints=self.keypoints[keep],
                    embeddings=self.embeddings[keep],
                )
            )
        return frames


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


def build_detection_cache(
    image_paths: Sequence[Path],
    frame_idx: Sequence[int],
    timestamp_s: Sequence[float],
    model_path: str | Path,
    conf: float = DEFAULT_CACHE_CONF,
    imgsz: int = 960,
    batch_size: int = 32,
    device: str = "cuda:0",
    half: bool = True,
    progress: bool = False,
) -> DetCache:
    """Run the pose model and ReID on frames and cache every detection.

    Args:
        image_paths: Frame images in time order.
        frame_idx: Frame index for each image.
        timestamp_s: Source-video timestamp for each image.
        model_path: Ultralytics pose weights.
        conf: Detection threshold (keep low; association filters later).
        imgsz: Inference size.
        batch_size: Frames per model call.
        device: Torch device (``cuda:0``, ``cpu``, ...).
        half: Use half precision (ignored on CPU).
        progress: Print a progress line every ~10% of frames.

    Returns:
        The populated detection cache.
    """
    import cv2
    from boxmot.reid.core.reid import ReID
    from ultralytics import YOLO

    on_cpu = str(device).startswith("cpu")
    use_half = half and not on_cpu
    model = YOLO(str(model_path))
    reid = ReID(path=None, device=device, half=use_half).model
    paths = list(image_paths)

    offsets = [0]
    boxes, confs, kpts, embs = [], [], [], []
    image_size = (0, 0)
    chunks = [paths[s : s + batch_size] for s in range(0, len(paths), batch_size)]
    pool = ThreadPoolExecutor(max_workers=8)

    def _read(chunk: list[Path]) -> list[np.ndarray]:
        return list(pool.map(lambda p: cv2.imread(str(p)), chunk))

    pending = pool.submit(_read, chunks[0]) if chunks else None
    reid_pool = ThreadPoolExecutor(max_workers=1)
    step = max(1, len(chunks) // 10)
    for n in range(len(chunks)):
        images = pending.result()
        if n + 1 < len(chunks):
            pending = pool.submit(_read, chunks[n + 1])
        results = model(
            images,
            conf=conf,
            iou=0.7,
            imgsz=imgsz,
            device=device,
            half=use_half,
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
        # ReID runs in a background thread so its CPU-side cropping overlaps
        # with YOLO on the next batch.
        embs.append(reid_pool.submit(batch_reid_features, reid, images, chunk_boxes))
        if progress and (n + 1) % step == 0:
            done = min(len(paths), (n + 1) * batch_size)
            print(f"[detect] {done}/{len(paths)} frames", flush=True)
    embs = [future.result() for future in embs]
    reid_pool.shutdown()
    pool.shutdown()
    dim = max((e.shape[1] for e in embs if e.size), default=512)
    return DetCache(
        frame_idx=np.asarray(frame_idx),
        timestamp_s=np.asarray(timestamp_s, dtype=float),
        offsets=np.asarray(offsets),
        boxes=np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
        conf=np.concatenate(confs) if confs else np.zeros(0, np.float32),
        keypoints=np.concatenate(kpts) if kpts else np.zeros((0, 17, 3), np.float32),
        embeddings=(
            np.concatenate([e.reshape(-1, dim) for e in embs])
            if embs
            else np.zeros((0, dim), np.float32)
        ),
        image_size=image_size,
    )
