"""Score gaze predictions against the colleague's portal gaze labels.

The labels (``portal.gaze_labels``) say, per frame and person, whether they
look at the other person's face, body, an object, or elsewhere, and
optionally where. A model's gaze point is mapped onto the same categories with
the partner's box: the top 30% of the box counts as the face.

CLI (any number of gaze tables, e.g. two models):
    python -m gaze_analysis.label_eval --camera-dir data/output/1012/camera_a \\
        --gaze gaze_heatmap.csv gaze_heatmap_page.csv
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PERSON_TRACK = {"parent": 0, "child": 1}
FACE_FRACTION = 0.30
INOUT_THRESHOLD = 0.5
# Collapse the label set onto what a gaze point can tell apart.
LABEL_TO_EVAL = {
    "partner_face": "partner_face",
    "partner_body": "partner_body",
    "object": "not_partner",
    "elsewhere": "not_partner",
}


def predicted_target(
    point: Sequence[float],
    partner_box: Sequence[float],
    image_width: float,
    image_height: float,
    inout: float = 1.0,
) -> str:
    """Map a predicted gaze point onto a label category.

    Args:
        point: Gaze point (x, y) as fractions of the image size.
        partner_box: The other person's box (x1, y1, x2, y2) in pixels.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        inout: Model's probability that the target is inside the image.

    Returns:
        ``partner_face``, ``partner_body`` or ``not_partner``.
    """
    if inout < INOUT_THRESHOLD:
        return "not_partner"
    x, y = point[0] * image_width, point[1] * image_height
    x1, y1, x2, y2 = partner_box
    if not (x1 <= x <= x2 and y1 <= y <= y2):
        return "not_partner"
    face_bottom = y1 + FACE_FRACTION * (y2 - y1)
    return "partner_face" if y <= face_bottom else "partner_body"


def score_gaze_predictions(
    labels: pd.DataFrame,
    gaze: pd.DataFrame,
    tracks: pd.DataFrame,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Compare one model's gaze table with the labels.

    Args:
        labels: Rows with ``frame_idx``, ``person``, ``target``, ``point``.
        gaze: ``gaze_heatmap.csv`` rows (``gaze_peak_x/y`` in 0..1,
            ``inout_score``).
        tracks: Track boxes on the same frame grid (``tracks_2d_smooth.csv``).
        image_size: (width, height) in pixels.

    Returns:
        Counts, category accuracy, looking-at-partner accuracy, and the
        median/mean distance between labeled and predicted points (fraction
        of image size, as in the GazeFollow benchmark).
    """
    width, height = image_size
    gaze_by_key = {
        (int(r.frame_idx), int(r.track_id)): r for r in gaze.itertuples(index=False)
    }
    box_by_key = {
        (int(r.frame_idx), int(r.track_id)): (
            r.bbox_x1,
            r.bbox_y1,
            r.bbox_x2,
            r.bbox_y2,
        )
        for r in tracks.itertuples(index=False)
    }
    rows = []
    for label in labels.itertuples(index=False):
        truth = LABEL_TO_EVAL.get(label.target)
        if truth is None or label.person not in PERSON_TRACK:
            continue
        own = PERSON_TRACK[label.person]
        partner = 1 - own
        pred_row = gaze_by_key.get((int(label.frame_idx), own))
        partner_box = box_by_key.get((int(label.frame_idx), partner))
        if pred_row is None or partner_box is None:
            continue
        point = (float(pred_row.gaze_peak_x), float(pred_row.gaze_peak_y))
        inout = float(getattr(pred_row, "inout_score", 1.0))
        pred = predicted_target(point, partner_box, width, height, inout)
        error = np.nan
        if isinstance(label.point, (list, tuple)) and len(label.point) == 2:
            error = float(
                np.hypot(point[0] - label.point[0], point[1] - label.point[1])
            )
        rows.append(
            {
                "person": label.person,
                "truth": truth,
                "pred": pred,
                "error": error,
            }
        )
    table = pd.DataFrame(rows, columns=["person", "truth", "pred", "error"])
    if table.empty:
        return {"n_labels": 0}
    at_partner_true = table["truth"] != "not_partner"
    at_partner_pred = table["pred"] != "not_partner"
    errors = table["error"].dropna()
    scores: dict[str, Any] = {
        "n_labels": int(len(table)),
        "target_accuracy": float((table["truth"] == table["pred"]).mean()),
        "looking_at_partner_accuracy": float(
            (at_partner_true == at_partner_pred).mean()
        ),
        "n_points": int(len(errors)),
        "point_error_median": float(errors.median()) if len(errors) else None,
        "point_error_mean": float(errors.mean()) if len(errors) else None,
        "confusion": pd.crosstab(table["truth"], table["pred"]).to_dict(),
    }
    for person, group in table.groupby("person"):
        scores[f"{person}_target_accuracy"] = float(
            (group["truth"] == group["pred"]).mean()
        )
    return scores


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    from gaze_analysis.config import frame_image_size
    from portal.gaze_labels import load_gaze_labels

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera-dir", required=True, type=Path)
    parser.add_argument("--gaze", nargs="+", default=["gaze_heatmap.csv"])
    parser.add_argument("--tracks", default="tracks_2d_smooth.csv")
    args = parser.parse_args(argv)

    labels = pd.DataFrame(load_gaze_labels(args.camera_dir))
    if labels.empty:
        raise SystemExit(f"No gaze labels saved yet in {args.camera_dir}")
    tracks = pd.read_csv(args.camera_dir / args.tracks)
    image_size = frame_image_size(args.camera_dir, (854, 480))
    results = {}
    for name in args.gaze:
        gaze = pd.read_csv(args.camera_dir / name)
        results[name] = score_gaze_predictions(labels, gaze, tracks, image_size)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
