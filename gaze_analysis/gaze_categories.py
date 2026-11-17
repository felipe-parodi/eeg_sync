"""Frame-level parent-child gaze categories.

Each person's gaze point (from the gaze model) is located relative to the other
person's head and body boxes, in pixels:

- ``mutual_gaze``: each gaze point lands on the other's face (head box
  enlarged 1.5x).
- ``parent_watching`` / ``child_watching``: one gaze point lands on the other
  person (face or body) and the other's gaze does not land on them.
- ``both_watching``: each looks at the other, but not face to face.
- ``joint_attention``: both look at something in view, neither at the other,
  and their gaze points are within 1.5 mean head-heights of each other.
- ``independent``: both look at something in view, at different things.
- ``unknown``: a gaze is out of frame (model in/out score < 0.5) or a box is
  missing, and no watching was detected.

Single-frame changes of category are absorbed into the surrounding category.
:func:`chance_shares` gives each category's rate when the child's gaze is
shifted in time, which is the level expected from where people sit alone.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

CATEGORIES = [
    "mutual_gaze",
    "joint_attention",
    "parent_watching",
    "child_watching",
    "both_watching",
    "independent",
    "unknown",
]
INOUT_THRESHOLD = 0.5
FACE_SCALE = 1.5
JOINT_HEAD_HEIGHTS = 1.5
CHANCE_SHIFT_S = 60.0
# Okabe-Ito colorblind-safe palette.
CATEGORY_COLORS = {
    "mutual_gaze": "#009E73",
    "joint_attention": "#0072B2",
    "parent_watching": "#E69F00",
    "child_watching": "#CC79A7",
    "both_watching": "#D55E00",
    "independent": "#999999",
    "unknown": "#DDDDDD",
}

Box = Optional[Tuple[float, float, float, float]]


def _boxes(series: pd.Series) -> np.ndarray:
    """Stack box tuples into an (N, 4) float array, NaN for missing boxes."""
    out = np.full((len(series), 4), np.nan)
    for i, box in enumerate(series):
        if box is not None and not (isinstance(box, float) and np.isnan(box)):
            out[i] = box
    return out


def _inside(
    x: np.ndarray, y: np.ndarray, box: np.ndarray, scale: float = 1.0
) -> np.ndarray:
    cx = (box[:, 0] + box[:, 2]) / 2
    cy = (box[:, 1] + box[:, 3]) / 2
    hw = (box[:, 2] - box[:, 0]) / 2 * scale
    hh = (box[:, 3] - box[:, 1]) / 2 * scale
    with np.errstate(invalid="ignore"):
        hit = (np.abs(x - cx) <= hw) & (np.abs(y - cy) <= hh)
    return hit & ~np.isnan(box).any(axis=1)


def classify_frames(frames: pd.DataFrame) -> pd.Series:
    """Assign one gaze category per frame.

    Args:
        frames: Table from :func:`build_frame_table` (pixel gaze points
            ``p_gx/p_gy/c_gx/c_gy``, in/out scores ``p_in/c_in``, and box
            tuples ``p_head/c_head/p_body/c_body``; ``None`` when missing).

    Returns:
        Category per row (index aligned with ``frames``).
    """
    p_head, c_head = _boxes(frames["p_head"]), _boxes(frames["c_head"])
    p_body, c_body = _boxes(frames["p_body"]), _boxes(frames["c_body"])
    p_in = frames["p_in"].to_numpy(float) >= INOUT_THRESHOLD
    c_in = frames["c_in"].to_numpy(float) >= INOUT_THRESHOLD
    px, py = frames["p_gx"].to_numpy(float), frames["p_gy"].to_numpy(float)
    cx, cy = frames["c_gx"].to_numpy(float), frames["c_gy"].to_numpy(float)

    p_face = p_in & _inside(px, py, c_head, FACE_SCALE)
    c_face = c_in & _inside(cx, cy, p_head, FACE_SCALE)
    p_on = p_face | (p_in & _inside(px, py, c_body))
    c_on = c_face | (c_in & _inside(cx, cy, p_body))

    head_h = np.nanmean(
        np.stack([p_head[:, 3] - p_head[:, 1], c_head[:, 3] - c_head[:, 1]]), axis=0
    )
    with np.errstate(invalid="ignore"):
        close = np.hypot(px - cx, py - cy) <= JOINT_HEAD_HEIGHTS * head_h
    known = p_in & c_in & ~np.isnan(p_body).any(axis=1) & ~np.isnan(c_body).any(axis=1)

    cats = np.select(
        [
            p_face & c_face,
            p_on & c_on,
            p_on,
            c_on,
            known & close,
            known,
        ],
        [
            "mutual_gaze",
            "both_watching",
            "parent_watching",
            "child_watching",
            "joint_attention",
            "independent",
        ],
        default="unknown",
    )
    return pd.Series(cats, index=frames.index, name="gaze_category")


def absorb_blips(categories: pd.Series, frame_idx: pd.Series) -> pd.Series:
    """Relabel one-frame category changes that sit between two equal frames.

    Only consecutive frames count as neighbors, so gaps are left alone.

    Args:
        categories: Category per frame, in time order.
        frame_idx: Time-grid frame index per row.

    Returns:
        Smoothed categories.
    """
    cats = categories.to_numpy(object).copy()
    f = frame_idx.to_numpy()
    for i in range(1, len(cats) - 1):
        if (
            cats[i - 1] == cats[i + 1] != cats[i]
            and f[i] - f[i - 1] == 1
            and f[i + 1] - f[i] == 1
        ):
            cats[i] = cats[i - 1]
    return pd.Series(cats, index=categories.index, name=categories.name)


def category_shares(categories: pd.Series) -> dict[str, float]:
    """Percent of frames in each category (all categories, summing to 100)."""
    counts = categories.value_counts()
    total = max(1, len(categories))
    return {f"{c}_pct": float(counts.get(c, 0) / total * 100) for c in CATEGORIES}


def chance_shares(frames: pd.DataFrame, shift_rows: int) -> dict[str, float]:
    """Category shares with the child's gaze shifted in time (circularly).

    Positions and boxes stay real; only the pairing between the parent's and
    the child's gaze in time is broken.

    Args:
        frames: Frame table in time order.
        shift_rows: Rows to shift the child's gaze by.

    Returns:
        Category shares, as :func:`category_shares`.
    """
    if len(frames) < 2:
        return category_shares(pd.Series([], dtype=object))
    shift = shift_rows % len(frames) or len(frames) // 2
    shifted = frames.copy()
    for col in ("c_gx", "c_gy", "c_in"):
        shifted[col] = np.roll(frames[col].to_numpy(), -shift)
    return category_shares(classify_frames(shifted))


def build_frame_table(
    gaze_df: pd.DataFrame,
    head_boxes_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    image_size: Tuple[int, int],
    parent_id: int = 0,
    child_id: int = 1,
) -> pd.DataFrame:
    """Join both people's gaze, head boxes and body boxes, one row per frame.

    Args:
        gaze_df: ``gaze_heatmap.csv`` rows (gaze point as image fractions).
        head_boxes_df: ``extract_head_bboxes`` output (normalized boxes).
        tracks_df: Tracks with pixel ``bbox_*`` columns.
        image_size: (width, height) of the frames in pixels.
        parent_id: Parent track ID.
        child_id: Child track ID.

    Returns:
        Frames where both people have a gaze estimate, in time order, with
        pixel gaze points and box tuples (``None`` when missing).
    """
    width, height = image_size
    scale = np.array([width, height, width, height], dtype=float)

    def per_track(track_id: int, prefix: str) -> pd.DataFrame:
        g = gaze_df[gaze_df["track_id"] == track_id]
        out = pd.DataFrame(
            {
                "frame_idx": g["frame_idx"].astype(int).to_numpy(),
                "timestamp_s": g["timestamp_s"].to_numpy(float),
                f"{prefix}_gx": g["gaze_peak_x"].to_numpy(float) * width,
                f"{prefix}_gy": g["gaze_peak_y"].to_numpy(float) * height,
                f"{prefix}_in": (
                    g["inout_score"].to_numpy(float)
                    if "inout_score" in g
                    else np.ones(len(g))
                ),
            }
        )
        heads = head_boxes_df[head_boxes_df["track_id"] == track_id]
        head_map = {
            int(r.frame_idx): tuple(
                np.array([r.head_x1, r.head_y1, r.head_x2, r.head_y2]) * scale
            )
            for r in heads.itertuples(index=False)
        }
        bodies = tracks_df[tracks_df["track_id"] == track_id]
        body_map = {
            int(r.frame_idx): (
                float(r.bbox_x1),
                float(r.bbox_y1),
                float(r.bbox_x2),
                float(r.bbox_y2),
            )
            for r in bodies.itertuples(index=False)
        }
        out[f"{prefix}_head"] = [head_map.get(f) for f in out["frame_idx"]]
        out[f"{prefix}_body"] = [body_map.get(f) for f in out["frame_idx"]]
        return out

    parent = per_track(parent_id, "p")
    child = per_track(child_id, "c").drop(columns="timestamp_s")
    frames = parent.merge(child, on="frame_idx").sort_values("frame_idx")
    return frames.reset_index(drop=True)


def gaze_categories(frames: pd.DataFrame, smooth: bool = True) -> pd.DataFrame:
    """Classify a frame table and absorb one-frame blips.

    Args:
        frames: Table from :func:`build_frame_table`.
        smooth: Absorb single-frame category changes.

    Returns:
        ``frame_idx``, ``timestamp_s``, ``gaze_category`` per frame.
    """
    cats = classify_frames(frames)
    if smooth:
        cats = absorb_blips(cats, frames["frame_idx"])
    return pd.DataFrame(
        {
            "frame_idx": frames["frame_idx"].to_numpy(),
            "timestamp_s": frames["timestamp_s"].to_numpy(),
            "gaze_category": cats.to_numpy(),
        }
    )


# Shorter shifts leave the child's gaze too close in time to the parent's for
# the chance level to mean "unrelated"; such blocks get no chance level.
MIN_CHANCE_SHIFT_S = 20.0


def chance_shift_rows(frames: pd.DataFrame, shift_s: float = CHANCE_SHIFT_S) -> int:
    """Rows corresponding to ``shift_s`` (half the block if it is shorter).

    Returns 0 (no chance level) when that shift would be under
    ``MIN_CHANCE_SHIFT_S``, i.e. for blocks with under 40 s of data.
    """
    if len(frames) < 2:
        return 0
    span = float(frames["timestamp_s"].iloc[-1] - frames["timestamp_s"].iloc[0])
    shift = min(shift_s, span / 2)
    if shift < MIN_CHANCE_SHIFT_S:
        return 0
    per_row = span / (len(frames) - 1) if span > 0 else 1.0
    return max(1, int(round(shift / per_row)))


def heatmap_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap (0..1) of two gaze heatmaps after removing their background.

    Each map has its median (diffuse background) subtracted, is clipped at
    zero and normalized to sum to 1; the score is the summed elementwise
    minimum (1 = identical gaze distributions, 0 = disjoint).
    """
    out = []
    for hm in (a, b):
        h = np.clip(np.asarray(hm, float) - np.median(hm), 0, None)
        s = h.sum()
        if s <= 0:
            return 0.0
        out.append(h / s)
    return float(np.minimum(out[0], out[1]).sum())
