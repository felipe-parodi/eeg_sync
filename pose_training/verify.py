"""Blind A/B human-verify harness for pose models.

Because every automatic metric we have is graded against teacher labels (circular),
the only non-circular quality signal is a human looking at predictions on the real
footage. This builds a *blind* comparison: for a stratified sample of frames it
renders model A vs model B side by side with the left/right order randomized, so
the reviewer cannot tell which model is which. The reviewer marks L / R / T (tie)
in a CSV; ``summarize`` maps the blind sides back to model names and tallies.

Usage:
    # 1. Build the set (renders pairs + a verdicts.csv to fill in)
    python -m pose_training.verify build --weights-a <fused3.pt> --weights-b <new.pt> --n 150
    # 2. Open pose_training/data/verify/pairs/*.png, fill the 'verdict' column
    #    of verdicts.csv with L (left better), R (right better), or T (tie).
    # 3. Tally
    python -m pose_training.verify summarize
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "pose_training" / "data" / "frame_manifest.csv"
DEFAULT_OUT = REPO_ROOT / "pose_training" / "data" / "verify"


def stratified_sample(
    rows: List[Dict[str, str]], n: int, seed: int
) -> List[Dict[str, str]]:
    """Sample ``n`` frames spread evenly across (participant, angle) sessions."""
    groups: Dict[Any, List[Dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["participant"], row["angle"]), []).append(row)
    rng = random.Random(seed)
    for frames in groups.values():
        rng.shuffle(frames)
    order = sorted(groups)
    picked: List[Dict[str, str]] = []
    cursors = {key: 0 for key in order}
    while len(picked) < n and any(cursors[k] < len(groups[k]) for k in order):
        for key in order:
            if cursors[key] < len(groups[key]):
                picked.append(groups[key][cursors[key]])
                cursors[key] += 1
                if len(picked) >= n:
                    break
    return picked


def tally(records: List[Dict[str, Any]], name_a: str, name_b: str) -> Dict[str, int]:
    """Map blind L/R verdicts back to model names and count wins/ties."""
    result = {name_a: 0, name_b: 0, "tie": 0, "unjudged": 0}
    for record in records:
        verdict = str(record.get("verdict", "")).strip().upper()
        left_is_a = str(record.get("left_is_a")).lower() in ("true", "1", "yes")
        if verdict == "L":
            result[name_a if left_is_a else name_b] += 1
        elif verdict == "R":
            result[name_b if left_is_a else name_a] += 1
        elif verdict == "T":
            result["tie"] += 1
        else:
            result["unjudged"] += 1
    return result


def _predict(model: Any, frame_path: str, imgsz: int, device: str, max_persons: int):
    from .evaluate import extract_persons
    from .pseudo_label import _result_to_arrays

    result = model(frame_path, imgsz=imgsz, conf=0.3, device=device, verbose=False)[0]
    boxes, confs, kxy, kconf = _result_to_arrays(result)
    return extract_persons(boxes, confs, kxy, kconf, max_persons)


def build_verify_set(
    weights_a: str,
    weights_b: str,
    rows: List[Dict[str, str]],
    out_dir: Path,
    n: int,
    seed: int,
    imgsz: int = 960,
    device: str = "0",
    max_persons: int = 4,
) -> Path:
    """Render blind side-by-side A/B pairs and a verdicts CSV. Returns CSV path."""
    import cv2
    import numpy as np
    from ultralytics import YOLO

    from .evaluate import _draw_pose

    model_a, model_b = YOLO(weights_a), YOLO(weights_b)
    picked = stratified_sample(rows, n, seed)
    rng = random.Random(seed)
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for index, row in enumerate(picked):
        path = row["frame_path"]
        image = cv2.imread(path)
        if image is None:
            continue
        a = _draw_pose(image, _predict(model_a, path, imgsz, device, max_persons))
        b = _draw_pose(image, _predict(model_b, path, imgsz, device, max_persons))
        left_is_a = rng.random() < 0.5
        left, right = (a, b) if left_is_a else (b, a)
        cv2.putText(
            left, "LEFT", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        cv2.putText(
            right, "RIGHT", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        gap = np.full((left.shape[0], 6, 3), 255, dtype=np.uint8)
        name = f"{index:03d}_{Path(path).stem}.png"
        cv2.imwrite(str(pairs_dir / name), np.hstack([left, gap, right]))
        records.append(
            {
                "id": index,
                "image": name,
                "frame_path": path,
                "left_is_a": left_is_a,
                "verdict": "",
            }
        )
    csv_path = out_dir / "verdicts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "image", "frame_path", "left_is_a", "verdict"]
        )
        writer.writeheader()
        writer.writerows(records)
    return csv_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blind A/B human-verify harness.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Render blind A/B pairs + verdicts.csv.")
    build.add_argument("--weights-a", required=True, type=str)
    build.add_argument("--weights-b", required=True, type=str)
    build.add_argument("--name-a", default="model_a", type=str)
    build.add_argument("--name-b", default="model_b", type=str)
    build.add_argument("--manifest", default=str(DEFAULT_MANIFEST), type=str)
    build.add_argument("--out-dir", default=str(DEFAULT_OUT), type=str)
    build.add_argument("--n", default=150, type=int)
    build.add_argument("--seed", default=0, type=int)
    build.add_argument("--device", default="0", type=str)
    summ = sub.add_parser("summarize", help="Tally a filled-in verdicts.csv.")
    summ.add_argument("--csv", default=str(DEFAULT_OUT / "verdicts.csv"), type=str)
    summ.add_argument("--name-a", default="model_a", type=str)
    summ.add_argument("--name-b", default="model_b", type=str)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "build":
        with open(args.manifest, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        csv_path = build_verify_set(
            args.weights_a,
            args.weights_b,
            rows,
            Path(args.out_dir),
            args.n,
            args.seed,
            device=args.device,
        )
        print(f"Wrote {csv_path}. Open {Path(args.out_dir) / 'pairs'} and fill the")
        print("'verdict' column with L (left better), R (right better), or T (tie).")
        print(
            f"Then: python -m pose_training.verify summarize --csv {csv_path} "
            f"--name-a {args.name_a} --name-b {args.name_b}"
        )
    else:
        with open(args.csv, encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        result = tally(records, args.name_a, args.name_b)
        print("Verify tally:")
        for key, value in result.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
