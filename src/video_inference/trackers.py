"""Per-frame multi-object tracker adapters with a uniform interface.

The video inference runners only ever need: "given the N detections in this
frame and the frame image, return a tracker id per detection." Wrapping each
external tracker behind a small adapter keeps the runner code free of
tracker-specific branches and makes it easy to swap motion-only ByteTrack
for appearance-aware BoT-SORT.

Supported backends:
    - ("roboflow", "bytetrack")  -> supervision.ByteTrack (motion only)
    - ("boxmot",  "botsort")     -> boxmot BotSort with OSNet ReID

Both imports are deferred to adapter construction so module import stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol

import numpy as np


class TrackerAdapter(Protocol):
    """Per-frame tracker uniform interface."""

    def assign(
        self,
        detections_xyxy: np.ndarray,
        confidences: np.ndarray,
        image: np.ndarray,
    ) -> List[Optional[int]]:
        """Return one tracker id per input detection (None for unmatched).

        Args:
            detections_xyxy: (N, 4) array of (x1, y1, x2, y2) bboxes.
            confidences: (N,) array of detection confidences.
            image: (H, W, 3) BGR source frame. Required by ReID backends.

        Returns:
            List of length N. Each element is a tracker id (int) or None
            if the underlying tracker did not associate this detection.
        """
        ...


@dataclass
class SupervisionByteTrackAdapter:
    """Wraps supervision's motion-only ByteTrack."""

    lost_track_buffer: int = 60
    frame_rate: float = 30.0
    _tracker: object = field(default=None, init=False, repr=False)
    _sv_module: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import supervision as sv

        self._sv_module = sv
        self._tracker = sv.ByteTrack(
            lost_track_buffer=self.lost_track_buffer,
            frame_rate=float(self.frame_rate),
        )

    def assign(
        self,
        detections_xyxy: np.ndarray,
        confidences: np.ndarray,
        image: np.ndarray,
    ) -> List[Optional[int]]:
        n = len(detections_xyxy)
        if n == 0:
            return []
        sv = self._sv_module
        det = sv.Detections(
            xyxy=np.asarray(detections_xyxy, dtype=float),
            confidence=np.asarray(confidences, dtype=float),
            class_id=np.zeros(n, dtype=int),
        )
        tracked = self._tracker.update_with_detections(det)
        ids = getattr(tracked, "tracker_id", None)
        out: List[Optional[int]] = [None] * n
        if ids is None:
            return out
        for i, tid in enumerate(np.asarray(ids).tolist()):
            if i >= n:
                break
            out[i] = int(tid) if tid is not None else None
        return out


@dataclass
class BoxmotBotSortAdapter:
    """Wraps boxmot's BoT-SORT (motion + appearance ReID)."""

    lost_track_buffer: int = 16  # at 8 fps default == 2 s of memory
    frame_rate: float = 8.0
    device: str = "cpu"
    reid_weights: Optional[Path] = None
    half: bool = False
    _tracker: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Deferred imports: boxmot pulls torch, opencv, etc.
        from boxmot.reid.core.reid import ReID
        from boxmot.trackers.botsort.botsort import BotSort

        # boxmot's BotSort calls ``reid_model.get_features(...)`` directly.
        # ``ReID(...)`` is a high-level wrapper; the actual backend with
        # ``get_features`` is ``ReID(...).model``. See boxmot's own
        # ``create_tracker`` in ``boxmot/trackers/tracker_zoo.py``.
        reid_backend = ReID(
            path=self.reid_weights,
            device=self.device,
            half=self.half,
        ).model
        if hasattr(reid_backend, "warmup"):
            reid_backend.warmup()
        self._tracker = BotSort(
            reid_model=reid_backend,
            with_reid=True,
            track_buffer=int(self.lost_track_buffer),
            frame_rate=int(self.frame_rate),
        )

    def assign(
        self,
        detections_xyxy: np.ndarray,
        confidences: np.ndarray,
        image: np.ndarray,
    ) -> List[Optional[int]]:
        n = len(detections_xyxy)
        out: List[Optional[int]] = [None] * n
        if n == 0:
            return out
        dets = np.zeros((n, 6), dtype=np.float32)
        dets[:, :4] = np.asarray(detections_xyxy, dtype=np.float32)
        dets[:, 4] = np.asarray(confidences, dtype=np.float32)
        # class id stays at 0 (person)
        tracked = self._tracker.update(dets, np.asarray(image))
        if tracked is None or len(tracked) == 0:
            return out
        # boxmot TrackResults exposes id + det_ind; det_ind maps each output
        # row back to the input detection index.
        det_inds = np.asarray(tracked.det_ind, dtype=int).tolist()
        track_ids = np.asarray(tracked.id, dtype=int).tolist()
        for det_i, tid in zip(det_inds, track_ids):
            if 0 <= det_i < n:
                out[det_i] = int(tid)
        return out


def make_tracker(
    backend: str,
    name: str,
    frame_rate: float,
    device: str = "cpu",
) -> TrackerAdapter:
    """Construct the right adapter for a (backend, name) pair.

    Args:
        backend: "roboflow" (supervision) or "boxmot".
        name: "bytetrack" or "botsort".
        frame_rate: Inference fps. Forwarded to the underlying tracker so its
            lost-track buffer is interpreted in real time.
        device: Torch device string ("cpu", "cuda", "cuda:0", "mps").

    Raises:
        ValueError: For any backend/name combination that is not supported.
    """
    backend_lc = (backend or "").lower()
    name_lc = (name or "").lower()
    if backend_lc == "roboflow" and name_lc == "bytetrack":
        return SupervisionByteTrackAdapter(lost_track_buffer=60, frame_rate=frame_rate)
    if backend_lc == "boxmot" and name_lc == "botsort":
        return BoxmotBotSortAdapter(
            lost_track_buffer=max(8, int(round(frame_rate * 2.0))),
            frame_rate=frame_rate,
            device=device,
        )
    raise ValueError(
        f"Unsupported tracker backend={backend!r} name={name!r}. "
        "Supported: ('roboflow','bytetrack'), ('boxmot','botsort')."
    )
