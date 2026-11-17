"""Side-by-side gallery of frames where two association variants disagree.

For visual precision checks: renders the baseline (left) and candidate (right)
assignments on the same frame, grouped by what changed (person added, person
removed, roles changed). Writes JPEG tiles plus a self-contained local HTML
page. The page holds participant frames, so keep it local; do not upload it.

Example:
    uv run python -m pose_training.bench_gallery --baseline deployed \
        --candidate offline --out pose_training/data/gallery_offline
"""

from __future__ import annotations

import argparse
import base64
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pose_training.bench import (
    DEFAULT_CACHE_DIR,
    BenchCamera,
    DetCache,
    box_iou,
    deployed_assignment,
    discover_cameras,
    load_anchor_gold,
    split_anchors,
)

COLORS = {"parent": (0, 200, 0), "child": (0, 140, 255)}
SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def run_variant(camera: BenchCamera, name: str, cache_dir: Path) -> pd.DataFrame:
    """Run a named variant on one camera with the fit anchors."""
    import pose_training.assoc_variants  # noqa: F401
    import pose_training.bench as registry

    fit, _ = split_anchors(load_anchor_gold(camera.job_dir, camera.camera_id))
    if name == "deployed":
        return deployed_assignment(camera, fit)
    cache = DetCache.load(cache_dir / f"{camera.key}.npz")
    return registry.VARIANTS[name](cache, fit)


def classify_diff(base: dict[str, tuple], cand: dict[str, tuple]) -> str | None:
    """Label how two per-frame role->box maps differ, or None if they agree."""
    if len(cand) > len(base):
        return "added"
    if len(cand) < len(base):
        return "removed"
    for role, box in cand.items():
        if role not in base or box_iou(box, base[role]) < 0.5:
            return "changed"
    return None


def _boxes(df: pd.DataFrame) -> dict[int, dict[str, tuple]]:
    out: dict[int, dict[str, tuple]] = {}
    for r in df.itertuples(index=False):
        out.setdefault(int(r.frame_idx), {})[r.role] = (
            r.x1,
            r.y1,
            r.x2,
            r.y2,
            float(r.conf),
        )
    return out


def draw(image: np.ndarray, roles: dict[str, tuple], title: str) -> np.ndarray:
    """Draw role boxes (with confidence) on a copy of ``image``."""
    img = image.copy()
    for role, (x1, y1, x2, y2, conf) in roles.items():
        color = COLORS[role]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            img,
            f"{role[0].upper()} {conf:.2f}",
            (int(x1) + 2, max(14, int(y1) + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    cv2.putText(img, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", default="deployed")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-camera", type=int, default=6)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cameras = [
        c
        for c in discover_cameras()
        if (args.cache_dir / f"{c.key}.npz").exists()
        and (not args.only or c.key in args.only)
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[tuple[str, np.ndarray]]] = {
        "added": [],
        "removed": [],
        "changed": [],
    }
    counts: dict[str, dict[str, int]] = {}
    for cam in cameras:
        base = _boxes(run_variant(cam, args.baseline, args.cache_dir))
        cand = _boxes(run_variant(cam, args.candidate, args.cache_dir))
        index = pd.read_csv(cam.camera_dir / "frames" / "frame_index.csv")
        names = dict(zip(index["frame_idx"], index["image_name"]))
        diffs: dict[str, list[int]] = {"added": [], "removed": [], "changed": []}
        for f in sorted(set(base) | set(cand)):
            kind = classify_diff(
                {k: v[:4] for k, v in base.get(f, {}).items()},
                {k: v[:4] for k, v in cand.get(f, {}).items()},
            )
            if kind:
                diffs[kind].append(f)
        counts[cam.key] = {k: len(v) for k, v in diffs.items()}
        for kind, frames in diffs.items():
            for f in rng.sample(frames, min(args.per_camera, len(frames))):
                image = cv2.imread(str(cam.camera_dir / "frames" / names[f]))
                tile = np.hstack(
                    [
                        draw(image, base.get(f, {}), f"{args.baseline}"),
                        draw(image, cand.get(f, {}), f"{args.candidate}"),
                    ]
                )
                label = f"{cam.key} frame {f}"
                cv2.imwrite(str(args.out / f"{kind}_{cam.key}_{f}.jpg"), tile)
                groups[kind].append((label, tile))
        print(f"[gallery] {cam.key}: {counts[cam.key]}", flush=True)

    parts = [
        "<html><head><meta charset=utf-8><title>Association diff gallery</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee}"
        "figure{margin:6px 0}img{width:100%;max-width:1400px}</style></head><body>",
        f"<h1>{args.baseline} (left) vs {args.candidate} (right)</h1>",
        "<p>Green = parent, orange = child, number = detection confidence.</p>",
        "<pre>" + pd.DataFrame(counts).T.to_string() + "</pre>",
    ]
    for kind, tiles in groups.items():
        parts.append(f"<h2>{kind} ({len(tiles)} shown)</h2>")
        for label, tile in tiles:
            ok, buf = cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, 80])
            parts.append(
                f"<figure><img src='data:image/jpeg;base64,{base64.b64encode(buf).decode()}'>"
                f"<figcaption>{label}</figcaption></figure>"
            )
    parts.append("</body></html>")
    (args.out / "gallery.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"[gallery] wrote {args.out / 'gallery.html'}")


if __name__ == "__main__":
    main()
