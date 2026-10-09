"""Offline two-identity association for parent/child pose tracking.

The portal processes finished recordings with exactly two people of interest
(windows with extra people are excluded before inference). This module uses
that by solving identity offline instead of frame by frame:

1. Remove duplicate detections within a frame using keypoint similarity, so two
   people whose boxes overlap (a child on a parent's lap) are both kept.
2. Link confident detections into conservative tracklets (box overlap plus
   appearance), ending a tracklet whenever the match is ambiguous. Faint
   detections form their own tracklets, kept only if they persist. Tracklets
   are split where appearance and box size jump (an identity hop).
3. Label tracklets parent or child. Tracklets touched by a human anchor box take
   its label; a tracklet whose anchors disagree is split where its appearance
   changes most. Other tracklets are labeled by appearance similarity to the
   anchored ones plus spatial continuity, and a role never holds two tracklets
   at once. Without anchors, the two strongest co-occurring tracklets seed the
   roles and the taller person is the parent.
4. Fill short gaps in each role with faint detections that sit on the
   interpolated path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# COCO-17 per-keypoint OKS sigmas.
COCO_SIGMAS = np.array(
    [
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
    ]
)
ROLES = ("parent", "child")
OUTPUT_COLUMNS = [
    "frame_idx",
    "timestamp_s",
    "role",
    "x1",
    "y1",
    "x2",
    "y2",
    "conf",
    "det_index",
    "filled",
]


@dataclass
class AssocConfig:
    """Tunable parameters for :func:`associate_two_people`.

    Attributes:
        conf_min: Detections below this confidence are ignored entirely.
        track_conf: Detections at or above this confidence build the main
            tracklets; fainter ones only form persistent tracklets or fill gaps.
        dedup_oks: Same-frame detections with keypoint similarity above this
            are treated as duplicates of one person.
        dedup_iou: Same-frame boxes with IoU above this are duplicates even if
            their keypoints disagree.
        link_iou: Minimum box IoU to continue a tracklet.
        link_app_weight: Weight of appearance distance in the linking cost.
        ambiguity_margin: A link is refused (the tracklet ends) when a rival
            detection or tracklet has a cost within this margin.
        max_gap_s: Longest gap a tracklet may bridge.
        min_faint_s: Minimum duration of a tracklet made of faint detections.
        overlap_tol: Frames a tracklet may overlap a role before it is
            considered a conflict (overlapping frames are not placed).
        max_overlap_frac: Larger overlaps are also tolerated up to this
            fraction of the tracklet's length.
        app_scale: Nats of parent-vs-child evidence per unit difference in
            appearance similarity to the two role prototypes.
        size_weight: Weight of the per-camera box-size evidence.
        evidence_weight: Scale of the summed parent-vs-child evidence in a
            tracklet's reward (per detection, clipped to +-3 nats).
        continuity_bonus: Reward for chaining two tracklets that continue
            spatially within ``continuity_gap_s``.
        jump_window_s: Chained tracklets closer in time than this must not
            jump further than ``jump_max`` body heights.
        jump_max: Maximum plausible displacement (in body heights) within
            ``jump_window_s``.
        jump_penalty: Cost of an implausible jump between chained tracklets.
        chain_rounds: Rounds of solving both role chains and refitting the
            appearance and size models.
        cooccur_discount: How much of a tracklet's role evidence is ignored
            for the frames it shares with the other role's chain (1 = all).
        rescue_conf: Tracklets left out of both chains with at least this
            mean confidence still fill a role that is free for most of their
            frames.
        change_window: Detections compared on each side of a candidate
            identity change inside a tracklet (18 = 3 s at 6 fps).
        change_threshold: Change score above which a tracklet is split at a
            suspected identity hop (about 7 is a sensible value). ``None``
            disables splitting; on the benchmark, anchor-local labels
            (``anchor_force_s``) handled hops better.
        anchor_force_s: If set, an anchor fixes its tracklet's role only
            within this many seconds of the anchor; the rest of the tracklet
            is labeled by evidence. ``None`` fixes the whole tracklet.
        continuity_gap_s: Max gap for the continuity bonus.
        anchor_iou: IoU needed to match an anchor box to a detection.
        static_min_s: Minimum duration for the static false-positive check.
        static_max_conf: Tracklets at or above this mean confidence are
            never treated as static false positives.
        static_max_motion: Center motion (std / box height) below which a
            long faint tracklet counts as static clutter.
        fill_max_gap_s: Longest role gap filled from faint detections.
        fill_iou: IoU between a faint detection and the interpolated box
            needed to fill a gap frame.
    """

    conf_min: float = 0.05
    track_conf: float = 0.3
    dedup_oks: float = 0.5
    dedup_iou: float = 0.8
    link_iou: float = 0.3
    link_app_weight: float = 0.5
    ambiguity_margin: float = 0.1
    max_gap_s: float = 0.75
    min_faint_s: float = 1.0
    overlap_tol: int = 2
    max_overlap_frac: float = 0.3
    app_scale: float = 10.0
    size_weight: float = 1.0
    evidence_weight: float = 0.5
    continuity_bonus: float = 3.0
    jump_window_s: float = 1.0
    jump_max: float = 1.0
    jump_penalty: float = 20.0
    chain_rounds: int = 2
    cooccur_discount: float = 0.8
    rescue_conf: float = 0.5
    change_window: int = 18
    change_threshold: float | None = None
    anchor_force_s: float | None = 10.0
    continuity_gap_s: float = 1.5
    anchor_iou: float = 0.5
    static_min_s: float = 10.0
    static_max_conf: float = 0.35
    static_max_motion: float = 0.03
    fill_max_gap_s: float = 3.0
    fill_iou: float = 0.4


@dataclass
class FrameDetections:
    """All candidate detections in one frame."""

    frame_idx: int
    timestamp_s: float
    boxes: np.ndarray  # (n, 4) xyxy
    conf: np.ndarray  # (n,)
    keypoints: np.ndarray  # (n, 17, 3)
    embeddings: np.ndarray  # (n, d), L2-normalized


@dataclass
class AnchorBoxes:
    """Human label for one frame: the parent's and the child's boxes."""

    timestamp_s: float
    parent_box: tuple[float, float, float, float]
    child_box: tuple[float, float, float, float]


@dataclass
class Tracklet:
    """A short run of detections believed to be one person."""

    frames: list[int] = field(default_factory=list)  # positions in the frame list
    dets: list[int] = field(default_factory=list)  # detection index in that frame
    boxes: list[np.ndarray] = field(default_factory=list)
    confs: list[float] = field(default_factory=list)
    embs: list[np.ndarray] = field(default_factory=list)
    last_t: float = 0.0

    def add(
        self,
        pos: int,
        det: int,
        box: np.ndarray,
        conf: float,
        emb: np.ndarray,
        t: float,
    ) -> None:
        """Append one detection."""
        self.frames.append(pos)
        self.dets.append(det)
        self.boxes.append(box)
        self.confs.append(conf)
        self.embs.append(emb)
        self.last_t = t

    @property
    def weight(self) -> float:
        """Total detection confidence (length x confidence)."""
        return float(sum(self.confs))

    def mean_conf(self) -> float:
        """Mean detection confidence."""
        return float(np.mean(self.confs))

    def mean_emb(self) -> np.ndarray:
        """Confidence-weighted, L2-normalized mean embedding."""
        e = np.average(
            np.asarray(self.embs), axis=0, weights=np.asarray(self.confs) + 1e-6
        )
        return e / (np.linalg.norm(e) + 1e-12)

    def running_emb(self, n: int = 5) -> np.ndarray:
        """Mean embedding of the last ``n`` detections."""
        e = np.mean(np.asarray(self.embs[-n:]), axis=0)
        return e / (np.linalg.norm(e) + 1e-12)

    def median_height(self) -> float:
        """Median box height in pixels."""
        return float(np.median([b[3] - b[1] for b in self.boxes]))

    def split(self, at_pos: int) -> tuple[Tracklet, Tracklet]:
        """Split into frames before ``at_pos`` and frames from ``at_pos`` on."""
        a, b = Tracklet(), Tracklet()
        for i, pos in enumerate(self.frames):
            target = a if pos < at_pos else b
            target.frames.append(pos)
            target.dets.append(self.dets[i])
            target.boxes.append(self.boxes[i])
            target.confs.append(self.confs[i])
            target.embs.append(self.embs[i])
        return a, b


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def keypoint_similarity(
    a: np.ndarray, b: np.ndarray, scale: float, vis_thresh: float = 0.3
) -> float:
    """OKS-style similarity of two COCO-17 skeletons.

    Args:
        a: (17, 3) keypoints (x, y, conf).
        b: (17, 3) keypoints (x, y, conf).
        scale: Person scale in pixels (square root of box area).
        vis_thresh: Keypoints below this confidence in either skeleton are ignored.

    Returns:
        Similarity in [0, 1], or ``nan`` when fewer than 3 keypoints are shared.
    """
    vis = (a[:, 2] > vis_thresh) & (b[:, 2] > vis_thresh)
    if vis.sum() < 3 or scale <= 0:
        return float("nan")
    d2 = np.sum((a[vis, :2] - b[vis, :2]) ** 2, axis=1)
    k2 = (2 * COCO_SIGMAS[vis]) ** 2
    return float(np.mean(np.exp(-d2 / (2 * scale**2 * k2))))


def dedup_frame(frame: FrameDetections, cfg: AssocConfig) -> np.ndarray:
    """Return indices of non-duplicate detections in one frame, by confidence.

    Unlike box NMS, two strongly overlapping boxes are both kept when their
    skeletons differ (child on lap, hug), and dropped when the skeletons match.
    """
    order = np.argsort(-frame.conf)
    order = order[frame.conf[order] >= cfg.conf_min]
    iou = box_iou_matrix(frame.boxes, frame.boxes)
    keep: list[int] = []
    for j in order:
        duplicate = False
        for i in keep:
            if iou[i, j] > cfg.dedup_iou:
                duplicate = True
                break
            bi = frame.boxes[i]
            scale = float(np.sqrt(max((bi[2] - bi[0]) * (bi[3] - bi[1]), 1.0)))
            oks = keypoint_similarity(frame.keypoints[i], frame.keypoints[j], scale)
            if not np.isnan(oks) and oks > cfg.dedup_oks:
                duplicate = True
                break
        if not duplicate:
            keep.append(int(j))
    return np.asarray(keep, dtype=int)


def build_tracklets(
    frames: Sequence[FrameDetections],
    cfg: AssocConfig,
    candidates: Sequence[np.ndarray] | None = None,
) -> list[Tracklet]:
    """Link per-frame detections into conservative tracklets.

    Matching between active tracklets and the current frame's detections uses
    the Hungarian algorithm on ``(1 - IoU) + w * (1 - appearance similarity)``,
    gated by ``link_iou``. A link is refused when a rival detection or rival
    tracklet is nearly as good (``ambiguity_margin``): the tracklet then ends
    and the detection starts a new one, so tracklets do not hop between people.
    A tracklet also ends after ``max_gap_s`` without a match.

    Args:
        frames: Per-frame detections.
        cfg: Association parameters.
        candidates: Optional per-frame detection indices to use; defaults to
            :func:`dedup_frame` of each frame.

    Returns:
        All tracklets.
    """
    finished: list[Tracklet] = []
    active: list[Tracklet] = []
    margin = cfg.ambiguity_margin
    for pos, frame in enumerate(frames):
        t = frame.timestamp_s
        still = []
        for tr in active:
            (still if t - tr.last_t <= cfg.max_gap_s + 1e-6 else finished).append(tr)
        active = still
        cand = (
            dedup_frame(frame, cfg)
            if candidates is None
            else np.asarray(candidates[pos])
        )
        matched_dets: set[int] = set()
        ended: set[int] = set()
        if active and len(cand):
            last_boxes = np.asarray([tr.boxes[-1] for tr in active])
            iou = box_iou_matrix(last_boxes, frame.boxes[cand])
            run = np.asarray([tr.running_emb() for tr in active])
            app = run @ frame.embeddings[cand].T
            cost = (1 - iou) + cfg.link_app_weight * (1 - app)
            valid = iou >= cfg.link_iou
            cost = np.where(valid, cost, 1e6)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if not valid[r, c]:
                    continue
                rival_det = np.delete(cost[r], c).min(initial=1e6) < cost[r, c] + margin
                rival_trk = (
                    np.delete(cost[:, c], r).min(initial=1e6) < cost[r, c] + margin
                )
                if rival_det or rival_trk:
                    ended.add(r)
                    continue
                d = int(cand[c])
                active[r].add(
                    pos, d, frame.boxes[d], float(frame.conf[d]), frame.embeddings[d], t
                )
                matched_dets.add(d)
        if ended:
            finished.extend(tr for i, tr in enumerate(active) if i in ended)
            active = [tr for i, tr in enumerate(active) if i not in ended]
        for d in cand:
            if int(d) in matched_dets:
                continue
            tr = Tracklet()
            tr.add(
                pos,
                int(d),
                frame.boxes[d],
                float(frame.conf[d]),
                frame.embeddings[d],
                t,
            )
            active.append(tr)
    return finished + active


def _find_det(
    frame: FrameDetections, box: Sequence[float], min_iou: float
) -> int | None:
    if len(frame.boxes) == 0:
        return None
    iou = box_iou_matrix(np.asarray([box], dtype=float), frame.boxes)[0]
    best = int(np.argmax(iou))
    return best if iou[best] >= min_iou else None


def _change_scores(tr: Tracklet, window: int) -> np.ndarray:
    """Identity-change score before each detection of a tracklet.

    Compares the ``window`` detections before and after each index: a drop in
    mean appearance similarity and a jump in median log box height both raise
    the score. Indices too close to either end score 0.
    """
    n = len(tr.frames)
    scores = np.zeros(n)
    if n < 2 * window + 1:
        return scores
    log_h = np.log(np.maximum([b[3] - b[1] for b in tr.boxes], 1.0))
    embs = np.asarray(tr.embs)
    csum = np.vstack([np.zeros(embs.shape[1]), np.cumsum(embs, axis=0)])
    for i in range(window, n - window + 1):
        before = csum[i] - csum[i - window]
        after = csum[i + window] - csum[i]
        cos = float(
            before @ after / (np.linalg.norm(before) * np.linalg.norm(after) + 1e-12)
        )
        dh = abs(
            float(np.median(log_h[i - window : i]) - np.median(log_h[i : i + window]))
        )
        scores[i] = (0.95 - cos) / 0.05 + dh / 0.15
    return scores


def _split_at_identity_changes(
    tracklets: list[Tracklet], cfg: AssocConfig
) -> list[Tracklet]:
    """Split tracklets where appearance and size change abruptly (identity hops).

    Two people crossing can make a tracker swap them; each half of a swapped
    tracklet is then labeled on its own evidence.
    """
    out: list[Tracklet] = []
    for tr in tracklets:
        scores = _change_scores(tr, cfg.change_window)
        if cfg.change_threshold is None:
            out.append(tr)
            continue
        cuts: list[int] = []
        order = np.argsort(-scores)
        for i in order:
            if scores[i] < cfg.change_threshold:
                break
            if all(abs(int(i) - c) >= cfg.change_window for c in cuts):
                cuts.append(int(i))
        if not cuts:
            out.append(tr)
            continue
        rest = tr
        for i in sorted(cuts, reverse=True):
            rest, tail = rest.split(tr.frames[i])
            out.append(tail)
        out.append(rest)
    return [t for t in out if t.frames]


def _localize_anchor_labels(
    tracklets: list[Tracklet],
    votes: dict[int, list[tuple[int, str]]],
    times: np.ndarray,
    window_s: float,
) -> tuple[list[Tracklet], dict[int, str]]:
    """Keep anchor labels only near the anchors; split off the rest of the tracklet.

    An anchor proves who a tracklet is at the anchor frame, not minutes later:
    an undetected identity hop would otherwise carry the label to the wrong
    person. The parts further than ``window_s`` from every anchor become
    ordinary tracklets labeled by evidence.
    """
    out: list[Tracklet] = []
    labels: dict[int, str] = {}
    for k, tr in enumerate(tracklets):
        if k not in votes:
            out.append(tr)
            continue
        positions = [p for p, _ in votes[k]]
        t_lo = times[min(positions)] - window_s
        t_hi = times[max(positions)] + window_s
        lo = int(np.searchsorted(times, t_lo, side="left"))
        hi = int(np.searchsorted(times, t_hi, side="right"))
        before, rest = tr.split(lo)
        middle, after = rest.split(hi)
        for piece, forced in ((before, False), (middle, True), (after, False)):
            if not piece.frames:
                continue
            if forced:
                labels[len(out)] = votes[k][0][1]
            out.append(piece)
    return out, labels


def _anchor_votes(
    frames: Sequence[FrameDetections],
    tracklets: list[Tracklet],
    anchors: Sequence[AnchorBoxes],
    cfg: AssocConfig,
) -> dict[int, list[tuple[int, str]]]:
    """Map tracklet index -> list of (frame position, role) anchor votes."""
    times = np.asarray([f.timestamp_s for f in frames])
    owner = {
        (pos, det): k
        for k, tr in enumerate(tracklets)
        for pos, det in zip(tr.frames, tr.dets)
    }
    votes: dict[int, list[tuple[int, str]]] = {}
    for anchor in anchors:
        if not len(times):
            break
        pos = int(np.argmin(np.abs(times - anchor.timestamp_s)))
        if abs(times[pos] - anchor.timestamp_s) > 0.05:
            continue
        for role, box in (("parent", anchor.parent_box), ("child", anchor.child_box)):
            det = _find_det(frames[pos], box, cfg.anchor_iou)
            if det is None or (pos, det) not in owner:
                continue
            votes.setdefault(owner[(pos, det)], []).append((pos, role))
    return votes


def _best_split(tr: Tracklet, lo_pos: int, hi_pos: int) -> int:
    """Frame position in (lo_pos, hi_pos] where appearance changes the most."""
    idx = [i for i, p in enumerate(tr.frames) if lo_pos < p <= hi_pos]
    if not idx:
        return (lo_pos + hi_pos + 1) // 2
    embs = np.asarray(tr.embs)
    best_i, best_sim = idx[0], np.inf
    for i in idx:
        before = embs[max(0, i - 3) : i].mean(axis=0)
        after = embs[i : i + 3].mean(axis=0)
        sim = float(
            before @ after / (np.linalg.norm(before) * np.linalg.norm(after) + 1e-12)
        )
        if sim < best_sim:
            best_i, best_sim = i, sim
    return tr.frames[best_i]


def _split_conflicting(
    tracklets: list[Tracklet], votes: dict[int, list[tuple[int, str]]]
) -> tuple[list[Tracklet], dict[int, str]]:
    """Split tracklets whose anchors disagree; return per-tracklet anchor labels."""
    out: list[Tracklet] = []
    labels: dict[int, str] = {}
    for k, tr in enumerate(tracklets):
        pieces = [(tr, sorted(votes.get(k, [])))]
        done: list[tuple[Tracklet, list[tuple[int, str]]]] = []
        while pieces:
            piece, pv = pieces.pop()
            change = next(
                (i for i in range(1, len(pv)) if pv[i][1] != pv[i - 1][1]), None
            )
            if change is None:
                done.append((piece, pv))
                continue
            at = _best_split(piece, pv[change - 1][0], pv[change][0])
            left, right = piece.split(at)
            pieces.append((left, [v for v in pv if v[0] < at]))
            pieces.append((right, [v for v in pv if v[0] >= at]))
        for piece, pv in done:
            if not piece.frames:
                continue
            if pv:
                labels[len(out)] = pv[0][1]
            out.append(piece)
    return out, labels


def _seed_without_anchors(tracklets: list[Tracklet]) -> dict[int, str]:
    """Pick the two strongest co-occurring tracklets; the taller one is the parent."""
    order = sorted(range(len(tracklets)), key=lambda k: -tracklets[k].weight)
    if not order:
        return {}
    a = order[0]
    frames_a = set(tracklets[a].frames)
    b = next((k for k in order[1:] if frames_a & set(tracklets[k].frames)), None)
    if b is None:
        return {a: "parent"}
    if tracklets[a].median_height() >= tracklets[b].median_height():
        return {a: "parent", b: "child"}
    return {a: "child", b: "parent"}


def _is_static_false_positive(
    tr: Tracklet, times: np.ndarray, cfg: AssocConfig
) -> bool:
    """Long, faint, motionless tracklets are scene clutter (coat, toy, lens blob).

    A seated person still moves a little and is detected with higher
    confidence, so both low confidence and near-zero motion are required.
    """
    duration = times[tr.frames[-1]] - times[tr.frames[0]]
    if duration < cfg.static_min_s or tr.mean_conf() >= cfg.static_max_conf:
        return False
    boxes = np.asarray(tr.boxes)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    height = float(np.median(boxes[:, 3] - boxes[:, 1]))
    motion = float(np.max(np.std(centers, axis=0))) / max(height, 1.0)
    return motion < cfg.static_max_motion


SizeModel = tuple[float, float, float, float]  # slope, parent icpt, child icpt, sigma


def _fit_size_model(
    tracklets: list[Tracklet], assigned: dict[int, str]
) -> SizeModel | None:
    """Fit log(box height) = intercept[role] + slope * foot_y on labeled tracklets.

    The shared slope absorbs camera perspective (people further from the camera
    have their feet higher in the image and look smaller); the role intercepts
    capture that the parent is bigger than the child at the same place.

    Returns:
        ``(slope, parent_intercept, child_intercept, sigma)`` or ``None`` when
        either role has fewer than 20 labeled detections.
    """
    xs, ys, is_parent = [], [], []
    for k, role in assigned.items():
        for b in tracklets[k].boxes:
            h = float(b[3] - b[1])
            if h <= 1:
                continue
            xs.append(float(b[3]))
            ys.append(np.log(h))
            is_parent.append(role == "parent")
    is_parent_arr = np.asarray(is_parent)
    if is_parent_arr.sum() < 20 or (~is_parent_arr).sum() < 20:
        return None
    design = np.column_stack([np.asarray(xs), is_parent_arr, ~is_parent_arr]).astype(
        float
    )
    coef, *_ = np.linalg.lstsq(design, np.asarray(ys), rcond=None)
    resid = np.asarray(ys) - design @ coef
    sigma = max(float(np.std(resid)), 0.1)
    return float(coef[0]), float(coef[1]), float(coef[2]), sigma


def _size_llr(tr: Tracklet, model: SizeModel, cap: float = 3.0) -> np.ndarray:
    """Per-detection log-likelihood ratio parent vs child from box size."""
    slope, a_p, a_c, sigma = model
    boxes = np.asarray(tr.boxes)
    h = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    base = np.log(h) - slope * boxes[:, 3]
    llr = (-0.5 * ((base - a_p) / sigma) ** 2) + 0.5 * ((base - a_c) / sigma) ** 2
    return np.clip(llr, -cap, cap)


def _role_prototypes(
    tracklets: list[Tracklet], assigned: dict[int, str]
) -> dict[str, np.ndarray]:
    """Weighted mean appearance embedding of the tracklets holding each role."""
    protos: dict[str, np.ndarray] = {}
    for role in ROLES:
        members = [k for k, r in assigned.items() if r == role]
        if members:
            w = np.asarray([tracklets[k].weight for k in members])
            e = np.average(
                [tracklets[k].mean_emb() for k in members], axis=0, weights=w
            )
            protos[role] = e / (np.linalg.norm(e) + 1e-12)
    return protos


class _MaxFenwick:
    """Prefix-maximum Fenwick tree holding ``(value, payload)`` pairs."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.val = np.full(n + 1, -np.inf)
        self.arg = np.full(n + 1, -1, dtype=int)

    def update(self, i: int, value: float, payload: int) -> None:
        """Offer ``value`` at position ``i``."""
        i += 1
        while i <= self.n:
            if value > self.val[i]:
                self.val[i], self.arg[i] = value, payload
            i += i & -i

    def query(self, i: int) -> tuple[float, int]:
        """Max over positions ``0..i`` (inclusive); ``(-inf, -1)`` if empty."""
        best, arg = -np.inf, -1
        i += 1
        while i > 0:
            if self.val[i] > best:
                best, arg = self.val[i], int(self.arg[i])
            i -= i & -i
        return best, arg


def _best_chain(
    tracklets: list[Tracklet],
    node_weight: np.ndarray,
    times: np.ndarray,
    cfg: AssocConfig,
) -> list[int]:
    """Highest-scoring sequence of time-disjoint tracklets for one role.

    Dynamic programming over tracklets sorted by start. A chain may skip any
    tracklet and may contain long gaps (the person left the view). Consecutive
    members within ``continuity_gap_s`` that continue spatially earn
    ``continuity_bonus``; members within ``jump_window_s`` whose positions are
    implausibly far apart pay ``jump_penalty``.

    Args:
        tracklets: Candidate tracklets.
        node_weight: Reward for including each tracklet (``-inf`` forbids it).
        times: Timestamp per frame position.
        cfg: Association parameters.

    Returns:
        Indices of the chosen tracklets, in time order.
    """
    n = len(tracklets)
    if n == 0:
        return []
    starts = np.asarray([tr.frames[0] for tr in tracklets])
    ends = np.asarray([tr.frames[-1] for tr in tracklets])
    order = np.argsort(starts, kind="stable")
    near_s = max(cfg.continuity_gap_s, cfg.jump_window_s)
    tree = _MaxFenwick(len(times))
    dp = np.full(n, -np.inf)
    prev = np.full(n, -1, dtype=int)
    by_end: list[list[int]] = [[] for _ in range(len(times))]
    for j in order:
        if not np.isfinite(node_weight[j]):
            continue
        sj = int(starts[j])
        t_start = times[sj]
        # Far predecessors (ended more than near_s earlier): transition 0.
        far_cut = int(np.searchsorted(times, t_start - near_s, side="left")) - 1
        far_cut = min(far_cut, sj - 1)
        best, arg = tree.query(far_cut) if far_cut >= 0 else (-np.inf, -1)
        if best < 0:
            best, arg = 0.0, -1
        # Near predecessors: continuity bonus or jump penalty.
        first_box = np.asarray(tracklets[j].boxes[0])
        for pos in range(max(far_cut + 1, 0), sj):
            for i in by_end[pos]:
                gap = t_start - times[ends[i]]
                last_box = np.asarray(tracklets[i].boxes[-1])
                trans = 0.0
                if gap <= cfg.continuity_gap_s and (
                    box_iou_matrix(last_box[None], first_box[None])[0, 0] > 0.1
                ):
                    trans = cfg.continuity_bonus
                elif gap <= cfg.jump_window_s:
                    c_i = (last_box[:2] + last_box[2:]) / 2
                    c_j = (first_box[:2] + first_box[2:]) / 2
                    height = max(
                        last_box[3] - last_box[1], first_box[3] - first_box[1], 1.0
                    )
                    if np.linalg.norm(c_i - c_j) > cfg.jump_max * height:
                        trans = -cfg.jump_penalty
                if dp[i] + trans > best:
                    best, arg = dp[i] + trans, i
        dp[j] = node_weight[j] + best
        prev[j] = arg
        by_end[int(ends[j])].append(int(j))
        tree.update(int(ends[j]), dp[j], int(j))
    j = int(np.argmax(dp))
    if not np.isfinite(dp[j]) or dp[j] <= 0:
        return []
    chain = []
    while j >= 0:
        chain.append(j)
        j = int(prev[j])
    return chain[::-1]


def _coverage_reward(tr: Tracklet) -> float:
    """Reward for covering a role with this tracklet: up to 1 per detection by confidence."""
    return float(np.sum(np.clip((np.asarray(tr.confs) - 0.1) / 0.2, -0.5, 1.0)))


def associate_two_people(
    frames: Sequence[FrameDetections],
    anchors: Sequence[AnchorBoxes],
    cfg: AssocConfig | None = None,
) -> pd.DataFrame:
    """Assign parent/child identities to detections across a whole recording.

    After building tracklets and applying anchor labels, each role is solved
    as a global best chain of time-disjoint tracklets (:func:`_best_chain`).
    A tracklet's reward combines detection confidence with parent-vs-child
    evidence from appearance (relative to the role prototypes) and box size
    (a per-camera perspective model). Parent and child chains are solved in
    turn, each excluding the other's tracklets, and the evidence models are
    refit from the chosen chains between rounds.

    Args:
        frames: Per-frame candidate detections, in time order.
        anchors: Human parent/child boxes on some frames (may be empty).
        cfg: Association parameters.

    Returns:
        One row per assigned detection with columns ``frame_idx``,
        ``timestamp_s``, ``role``, ``x1``, ``y1``, ``x2``, ``y2``, ``conf``,
        ``det_index`` (index into that frame's detection arrays) and
        ``filled`` (True when the row came from gap filling).
    """
    cfg = cfg or AssocConfig()
    times = np.asarray([f.timestamp_s for f in frames])
    kept = [dedup_frame(f, cfg) for f in frames]
    strong = [k[f.conf[k] >= cfg.track_conf] for f, k in zip(frames, kept)]
    faint = [k[f.conf[k] < cfg.track_conf] for f, k in zip(frames, kept)]
    tracklets = build_tracklets(frames, cfg, strong)
    tracklets += [
        tr
        for tr in build_tracklets(frames, cfg, faint)
        if times[tr.frames[-1]] - times[tr.frames[0]] >= cfg.min_faint_s
    ]

    tracklets = _split_at_identity_changes(tracklets, cfg)
    votes = _anchor_votes(frames, tracklets, anchors, cfg)
    tracklets, forced = _split_conflicting(tracklets, votes)
    if cfg.anchor_force_s is not None and forced:
        votes = _anchor_votes(frames, tracklets, anchors, cfg)
        tracklets, forced = _localize_anchor_labels(
            tracklets, votes, times, cfg.anchor_force_s
        )
    labels = dict(forced) if forced else _seed_without_anchors(tracklets)

    static = np.asarray([_is_static_false_positive(tr, times, cfg) for tr in tracklets])
    coverage = np.asarray([_coverage_reward(tr) for tr in tracklets])
    chains: dict[str, list[int]] = {
        r: [k for k, v in forced.items() if v == r] for r in ROLES
    }
    for _ in range(cfg.chain_rounds):
        protos = _role_prototypes(tracklets, labels)
        size_model = _fit_size_model(tracklets, labels)
        evidence = np.zeros(len(tracklets))  # summed parent-vs-child log-odds
        for k, tr in enumerate(tracklets):
            per_det = np.zeros(len(tr.frames))
            if len(protos) == 2:
                emb = tr.mean_emb()
                per_det += cfg.app_scale * float(
                    emb @ protos["parent"] - emb @ protos["child"]
                )
            if size_model is not None:
                per_det += cfg.size_weight * _size_llr(tr, size_model)
            evidence[k] = cfg.evidence_weight * float(np.sum(np.clip(per_det, -3, 3)))
        for role in ROLES:
            other = "child" if role == "parent" else "parent"
            sign = 1.0 if role == "parent" else -1.0
            # A tracklet alongside the other role's chain is a second person in
            # those frames, so evidence matters less there: it decides which
            # role, not whether a confident detection survives.
            other_frames = {pos for k in chains[other] for pos in tracklets[k].frames}
            overlap = np.asarray(
                [
                    sum(pos in other_frames for pos in tr.frames) / len(tr.frames)
                    for tr in tracklets
                ]
            )
            discount = 1.0 - cfg.cooccur_discount * overlap
            weight = coverage + sign * evidence * discount
            weight[static] = -np.inf
            weight[chains[other]] = -np.inf
            for k, r in forced.items():
                weight[k] = 1e6 if r == role else -np.inf
            chains[role] = _best_chain(tracklets, weight, times, cfg)
        labels = {k: r for r in ROLES for k in chains[r]}

    occupied: dict[str, dict[int, tuple[int, int]]] = {r: {} for r in ROLES}
    for role in ROLES:
        for k in chains[role]:
            for i, pos in enumerate(tracklets[k].frames):
                occupied[role].setdefault(pos, (k, i))

    # Rescue: a confident tracklet left out of both chains still fills a role
    # that is free for most of its frames (evidence picks when both are free).
    in_chain = {k for r in ROLES for k in chains[r]}
    rescue = sorted(
        (
            k
            for k, tr in enumerate(tracklets)
            if k not in in_chain and not static[k] and tr.mean_conf() >= cfg.rescue_conf
        ),
        key=lambda k: -tracklets[k].weight,
    )
    for k in rescue:
        tr = tracklets[k]
        free = {
            r: sum(pos not in occupied[r] for pos in tr.frames) / len(tr.frames)
            for r in ROLES
        }
        options = [r for r in ROLES if free[r] >= 1 - cfg.max_overlap_frac]
        if not options:
            continue
        if len(options) == 2:
            role = "parent" if evidence[k] >= 0 else "child"
        else:
            role = options[0]
        for i, pos in enumerate(tr.frames):
            occupied[role].setdefault(pos, (k, i))

    rows: list[dict] = []
    used: dict[int, set[int]] = {}
    for role in ROLES:
        for pos, (k, i) in occupied[role].items():
            tr = tracklets[k]
            det = tr.dets[i]
            if det in used.setdefault(pos, set()):
                continue  # one detection cannot hold both roles
            used[pos].add(det)
            rows.append(_row(frames[pos], role, tr.boxes[i], tr.confs[i], det, False))

    rows += _fill_gaps(frames, kept, occupied, tracklets, used, cfg)
    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return out.sort_values(["frame_idx", "role"]).reset_index(drop=True)


def _row(
    frame: FrameDetections,
    role: str,
    box: np.ndarray,
    conf: float,
    det: int,
    filled: bool,
) -> dict:
    return {
        "frame_idx": frame.frame_idx,
        "timestamp_s": frame.timestamp_s,
        "role": role,
        "x1": float(box[0]),
        "y1": float(box[1]),
        "x2": float(box[2]),
        "y2": float(box[3]),
        "conf": float(conf),
        "det_index": int(det),
        "filled": filled,
    }


def _fill_gaps(
    frames: Sequence[FrameDetections],
    kept: Sequence[np.ndarray],
    occupied: dict[str, dict[int, tuple[int, int]]],
    tracklets: list[Tracklet],
    used: dict[int, set[int]],
    cfg: AssocConfig,
) -> list[dict]:
    """Fill short gaps in each role with faint detections on the interpolated path."""
    rows: list[dict] = []
    for role in ROLES:
        positions = sorted(occupied[role])
        for p0, p1 in zip(positions, positions[1:]):
            if p1 - p0 < 2:
                continue
            t0, t1 = frames[p0].timestamp_s, frames[p1].timestamp_s
            if t1 - t0 > cfg.fill_max_gap_s:
                continue
            k0, i0 = occupied[role][p0]
            k1, i1 = occupied[role][p1]
            b0, b1 = np.asarray(tracklets[k0].boxes[i0]), np.asarray(
                tracklets[k1].boxes[i1]
            )
            for pos in range(p0 + 1, p1):
                frame = frames[pos]
                cand = [d for d in kept[pos] if d not in used.setdefault(pos, set())]
                if not cand:
                    continue
                alpha = (frame.timestamp_s - t0) / max(t1 - t0, 1e-9)
                interp = (1 - alpha) * b0 + alpha * b1
                iou = box_iou_matrix(interp[None], frame.boxes[cand])[0]
                j = int(np.argmax(iou))
                if iou[j] < cfg.fill_iou:
                    continue
                det = int(cand[j])
                used[pos].add(det)
                rows.append(
                    _row(frame, role, frame.boxes[det], frame.conf[det], det, True)
                )
    return rows
