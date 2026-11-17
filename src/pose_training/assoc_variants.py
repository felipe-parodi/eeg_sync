"""Tracking / identity-association variants scored by ``pose_training.bench``.

Every variant maps a :class:`~pose_training.bench.DetCache` plus the *fit*
anchors to an assignment table with one row per output person per frame:
``frame_idx, timestamp_s, role, x1, y1, x2, y2, conf, det_row``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from pose_training.bench import (
    AnchorGold,
    BenchCamera,
    DetCache,
    box_iou,
    register,
)
from video_analysis.track_filter import AnchorAnnotation, _roles_from_user_anchors
from video_inference.ultralytics_runner import _assign_output_track_slot, _greedy_nms

ROLE_NAMES = {0: "parent", 1: "child"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def anchors_to_filter_json(
    tracks: pd.DataFrame, fit: Sequence[AnchorGold], min_iou: float = 0.5
) -> list[AnchorAnnotation]:
    """Translate box-level anchors into track-id anchors for ``tracks``.

    Args:
        tracks: Table with frame_idx, timestamp_s, track_id and bbox columns.
        fit: Anchors to translate.
        min_iou: IoU needed to match an anchor box to a track box.

    Returns:
        Anchor annotations usable by ``video_analysis.track_filter``.
    """
    out: list[AnchorAnnotation] = []
    times = tracks.drop_duplicates("frame_idx")[["frame_idx", "timestamp_s"]]
    t_arr = times["timestamp_s"].to_numpy(dtype=float)
    f_arr = times["frame_idx"].to_numpy()
    for g in fit:
        if not len(t_arr):
            break
        k = int(np.argmin(np.abs(t_arr - g.timestamp_s)))
        if abs(t_arr[k] - g.timestamp_s) > 0.05:
            continue
        frame = tracks[tracks["frame_idx"] == f_arr[k]]
        ids = {}
        for role, gbox in (("parent", g.parent_box), ("child", g.child_box)):
            best, best_iou = None, min_iou
            for row in frame.itertuples(index=False):
                iou = box_iou(
                    (row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2), gbox
                )
                if iou >= best_iou:
                    best, best_iou = int(row.track_id), iou
            ids[role] = best
        if (
            ids["parent"] is None
            or ids["child"] is None
            or ids["parent"] == ids["child"]
        ):
            continue
        out.append(
            AnchorAnnotation(
                frame_idx=int(f_arr[k]),
                parent_track_id=ids["parent"],
                child_track_id=ids["child"],
            )
        )
    return out


def nearest_anchor_roles(
    tracks: pd.DataFrame, anchors: Sequence[AnchorAnnotation]
) -> pd.DataFrame:
    """Deployed role logic: each row takes the role of the nearest anchor's id."""
    roles = _roles_from_user_anchors(tracks, list(anchors))
    kept = tracks[roles.notna()].copy()
    kept["role"] = roles[roles.notna()].astype(int).map(ROLE_NAMES).to_numpy()
    return kept.rename(
        columns={
            "bbox_x1": "x1",
            "bbox_y1": "y1",
            "bbox_x2": "x2",
            "bbox_y2": "y2",
            "track_confidence": "conf",
        }
    )


def run_track_filter(
    camera: BenchCamera, anchors: Sequence[AnchorAnnotation]
) -> pd.DataFrame:
    """Apply the deployed nearest-anchor role assignment to deployed tracks."""
    tracks = pd.read_csv(camera.camera_dir / "tracks_2d.csv")
    return nearest_anchor_roles(tracks, anchors)


def _tracks_frame(rows: list[dict]) -> pd.DataFrame:
    cols = [
        "frame_idx",
        "timestamp_s",
        "track_id",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "conf",
        "det_row",
    ]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# BoT-SORT variants (online tracker from cache + slot recycling + anchors)
# ---------------------------------------------------------------------------


class _NoCMC:
    def apply(self, img: np.ndarray, dets: np.ndarray | None = None) -> np.ndarray:
        return np.eye(2, 3)


def botsort_tracks(
    cache: DetCache,
    conf_min: float,
    greedy_nms_iou: float | None,
    top_k_before: int | None,
    top_k_after: int | None,
    track_buffer: int,
    new_track_thresh: float,
    track_low_thresh: float = 0.1,
    recycle_slots: bool = True,
) -> pd.DataFrame:
    """Run boxmot BoT-SORT over cached detections (with cached embeddings).

    Args:
        cache: Detection cache.
        conf_min: Drop detections below this confidence before tracking.
        greedy_nms_iou: Extra box NMS (deployed: 0.45); ``None`` disables it.
        top_k_before: Keep top-K by confidence before tracking (deployed: 2).
        top_k_after: Keep top-K tracked people by confidence after tracking.
        track_buffer: Raw boxmot ``track_buffer`` (boxmot rescales by fps/30).
        new_track_thresh: Minimum confidence to start a new track.
        track_low_thresh: boxmot second-stage association threshold.
        recycle_slots: Fold tracker ids into 2 output slots like the runner.

    Returns:
        Tracks table with ``track_id`` (slot or raw tracker id).
    """
    from boxmot.trackers.botsort.botsort import BotSort

    tracker = BotSort(
        reid_model=None,
        with_reid=True,
        track_buffer=track_buffer,
        frame_rate=6,
        new_track_thresh=new_track_thresh,
        track_low_thresh=track_low_thresh,
        cmc_method="ecc",
    )
    tracker.cmc = _NoCMC()
    width, height = cache.image_size
    dummy = np.zeros((height, width, 3), np.uint8)
    s2o: dict[int, int] = {}
    o2s: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    rows: list[dict] = []
    for i in range(cache.n_frames):
        sl = cache.frame_dets(i)
        idx = np.arange(sl.start, sl.stop)
        idx = idx[cache.conf[idx] >= conf_min]
        idx = idx[np.argsort(-cache.conf[idx])]
        if greedy_nms_iou is not None and len(idx):
            idx = np.asarray(_greedy_nms(idx, cache.boxes, cache.conf, greedy_nms_iou))
        if top_k_before is not None:
            idx = idx[:top_k_before]
        dets = np.zeros((len(idx), 6), np.float32)
        if len(idx):
            dets[:, :4] = cache.boxes[idx]
            dets[:, 4] = cache.conf[idx]
        embs = (
            cache.embeddings[idx]
            if len(idx)
            else np.zeros((0, cache.embeddings.shape[1]))
        )
        out = tracker.update(dets, dummy, embs)
        tracked = []
        if out is not None and len(out):
            for tid, di in zip(
                np.asarray(out.id).tolist(), np.asarray(out.det_ind).tolist()
            ):
                if 0 <= int(di) < len(idx):
                    tracked.append(
                        (float(dets[int(di), 4]), int(tid), int(idx[int(di)]))
                    )
        tracked.sort(reverse=True)
        if top_k_after is not None:
            tracked = tracked[:top_k_after]
        active: set[int] = set()
        frame_idx = int(cache.frame_idx[i])
        for conf, tid, row in tracked:
            out_id = (
                _assign_output_track_slot(
                    tid, frame_idx, 2, s2o, o2s, last_seen, active
                )
                if recycle_slots
                else tid
            )
            b = cache.boxes[row]
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_s": float(cache.timestamp_s[i]),
                    "track_id": out_id,
                    "bbox_x1": float(b[0]),
                    "bbox_y1": float(b[1]),
                    "bbox_x2": float(b[2]),
                    "bbox_y2": float(b[3]),
                    "conf": conf,
                    "det_row": row,
                }
            )
    return _tracks_frame(rows)


@register("botsort_deployed_cfg")
def botsort_deployed_cfg(cache: DetCache, fit: Sequence[AnchorGold]) -> pd.DataFrame:
    """Deployed configuration replayed from the cache (sanity check)."""
    tracks = botsort_tracks(
        cache,
        conf_min=0.3,
        greedy_nms_iou=0.45,
        top_k_before=2,
        top_k_after=None,
        track_buffer=12,
        new_track_thresh=0.6,
    )
    return nearest_anchor_roles(tracks, anchors_to_filter_json(tracks, fit))


@register("botsort_fixed")
def botsort_fixed(cache: DetCache, fit: Sequence[AnchorGold]) -> pd.DataFrame:
    """Corrected tracker config: 2 s buffer, low new-track threshold, no extra NMS."""
    tracks = botsort_tracks(
        cache,
        conf_min=0.1,
        greedy_nms_iou=None,
        top_k_before=None,
        top_k_after=2,
        track_buffer=300,  # boxmot: int(6 / 30 * 300) = 60 frames = 10 s
        new_track_thresh=0.35,
    )
    return nearest_anchor_roles(tracks, anchors_to_filter_json(tracks, fit))


# ---------------------------------------------------------------------------
# Offline two-identity association
# ---------------------------------------------------------------------------


def cache_to_frames(cache: DetCache, conf_min: float = 0.0) -> list:
    """Convert a detection cache into ``offline_assoc.FrameDetections``."""
    return cache.to_frames(conf_min)


def offline_variant(
    cache: DetCache, fit: Sequence[AnchorGold], **overrides
) -> pd.DataFrame:
    """Run offline association with ``AssocConfig`` overrides."""
    from video_inference.offline_assoc import (
        AnchorBoxes,
        AssocConfig,
        associate_two_people,
    )

    cfg = AssocConfig(**overrides)
    anchors = [AnchorBoxes(g.timestamp_s, g.parent_box, g.child_box) for g in fit]
    return associate_two_people(cache_to_frames(cache, cfg.conf_min), anchors, cfg)


register("offline")(lambda cache, fit: offline_variant(cache, fit))
register("offline_t20")(lambda cache, fit: offline_variant(cache, fit, track_conf=0.2))
register("offline_t40")(lambda cache, fit: offline_variant(cache, fit, track_conf=0.4))
register("offline_c10")(lambda cache, fit: offline_variant(cache, fit, conf_min=0.1))
register("offline_nofill")(lambda cache, fit: offline_variant(cache, fit, fill_iou=2.0))
register("offline_noanchor")(lambda cache, fit: offline_variant(cache, []))
register("offline_cp5")(lambda c, f: offline_variant(c, f, change_threshold=5.0))
register("offline_cpoff")(lambda c, f: offline_variant(c, f, change_threshold=1e9))
register("offline_local")(lambda c, f: offline_variant(c, f, anchor_force_s=2.0))
register("offline_local_cpoff")(
    lambda c, f: offline_variant(c, f, anchor_force_s=2.0, change_threshold=1e9)
)
register("offline_local_cp5")(
    lambda c, f: offline_variant(c, f, anchor_force_s=2.0, change_threshold=5.0)
)
for _w in (10.0, 30.0, 60.0):
    register(f"offline_local{int(_w)}_cpoff")(
        lambda c, f, w=_w: offline_variant(c, f, anchor_force_s=w, change_threshold=1e9)
    )
for _tc in (0.25, 0.4):
    register(f"offline_t{int(_tc * 100)}")(
        lambda c, f, tc=_tc: offline_variant(c, f, track_conf=tc)
    )
register("offline_c10_def")(lambda c, f: offline_variant(c, f, conf_min=0.1))
register("offline_c15_def")(lambda c, f: offline_variant(c, f, conf_min=0.15))
register("offline_cd0")(lambda c, f: offline_variant(c, f, cooccur_discount=0.0))
register("offline_cd08")(lambda c, f: offline_variant(c, f, cooccur_discount=0.8))
register("offline_cd1_r3")(lambda c, f: offline_variant(c, f, chain_rounds=3))
register("offline_jump2")(lambda c, f: offline_variant(c, f, jump_max=2.0))
register("offline_jump2p5")(
    lambda c, f: offline_variant(c, f, jump_max=2.0, jump_penalty=5.0)
)
register("offline_nojump")(lambda c, f: offline_variant(c, f, jump_penalty=0.0))
register("offline_norescue")(lambda c, f: offline_variant(c, f, rescue_conf=2.0))
register("offline_rescue35")(lambda c, f: offline_variant(c, f, rescue_conf=0.35))
