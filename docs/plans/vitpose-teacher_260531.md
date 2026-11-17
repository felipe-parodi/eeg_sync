# ViTPose Top-Down Teacher — Plan

Plan date: 2026-05-31. Goal: replace the `yolo11x-pose` teacher with a stronger
top-down teacher (ViTPose, HuggingFace `transformers`) to produce better COCO-17
pseudo-labels, then distill into `yolo11l-pose`. Motivated by the v1/v2 finding
that a same-family teacher (x-pose ≈ stock-L) caps the student at ~stock-L.

## Why top-down can beat the current teacher

`yolo11x-pose` is bottom-up: it localizes keypoints on the full frame, so small,
occluded, or overhead people get few pixels. A top-down model crops each person
to a normalized box and predicts keypoints on the zoomed crop, which is markedly
more accurate for exactly our hard cases (kids, occlusion, overhead/wide GoPro).
ViTPose is among the strongest COCO top-down models and is available in
`transformers`.

Honest caveat: ViTPose is still COCO-trained, so it improves *localization and
robustness* but does not erase COCO's *category* bias on children's body
proportions. Expect clear gains on small/occluded/overhead localization, partial
gains on kid-specific proportion errors. This is why Phase B (verify the teacher
is actually better) gates the whole plan.

## Architecture (drop-in teacher swap)

The rest of the pipeline stays identical. Only the label *source* changes.

```
frames (manifest)
   -> person boxes        [detector: reuse YOLO person detections]
   -> ViTPose on each crop [top-down keypoints, COCO-17 + scores]
   -> pseudo_labels_raw.json  [SAME schema as pseudo_label.py]
   -> dataset.py  (filter + export)        [unchanged]
   -> train.py    (frozen recipe)          [unchanged]
   -> evaluate.py (vs stock-L on held-out) [unchanged]
```

New module `pose_training/vitpose_label.py` mirrors `pseudo_label.py`'s I/O: it
reads `frame_manifest.csv` and writes the same `pseudo_labels_raw.json` (frames
-> persons -> bbox_xyxy/box_conf/keypoints[17x[x,y,conf]]), so `dataset.py`,
`train.py`, and `evaluate.py` need no changes.

## Dependencies

- `transformers` (>= 4.47, which has `VitPoseForPoseEstimation` +
  `VitPoseImageProcessor`) and likely `timm`. Install with
  `uv pip install transformers timm` (NOT `uv add`, which re-syncs the project
  and fails while the portal holds `video-processing-portal.exe`).
- torch/cuda already present. Run the huge variant on the A6000 (device 1,
  48 GB) to keep the 4090 free.

## Model choice

- Start: `usyd-community/vitpose-base-simple` (fast, COCO-17) to validate the
  wiring and the teacher-quality comparison.
- Scale: `usyd-community/vitpose-plus-huge` (or `-large`) for the final label
  pass if Phase B confirms it helps. ViTPose-plus is multi-dataset (MoE); select
  the COCO expert/dataset index in the processor.

## Boxes for the top-down crop

Top-down needs person boxes. Reuse YOLO person detection (the `yolo11x-pose` or a
plain detector already finds boxes well — box mAP50 ≈ 0.976). Two options:
1. Reuse the boxes already in `pseudo_labels_raw.json` from the x-pose run.
2. Run a YOLO detector pass fresh inside `vitpose_label.py`.
Option 2 keeps the module self-contained; box quality is not the bottleneck.

## Phases

### Phase A — Wiring
- Install deps; load `VitPoseImageProcessor` + `VitPoseForPoseEstimation`.
- `vitpose_label.py`: per frame, get YOLO boxes -> processor crops -> ViTPose ->
  post-process to image coords -> write the shared JSON schema. Keep all people
  (max_persons ~6). Optional manual flip-TTA (same approach as `pseudo_label`).
- Unit-test the box->crop->keypoint plumbing with a mocked model (mirror the
  `pseudo_label` tests).

### Phase B — Verify the teacher is actually better (gate)
- On a sample of HARD held-out frames (kids/overhead/occlusion), render
  `yolo11x-pose` vs ViTPose keypoints side by side. Confirm ViTPose is visibly
  better on those cases before spending a full label + train cycle. If it is not
  better, stop — the premise fails.

### Phase C — Full label + distill
- Run ViTPose over all sampled frames -> raw labels.
- `dataset.py` filter + export (same denoise filters).
- `train.py` frozen recipe -> `student_l_vitpose`.
- `evaluate.py` vs stock-L on held-out + overlay clips for human review.

## Open questions (decide before Phase C)

- Checkpoint: base-simple for the test, then huge/large for the real pass?
- Flip-TTA on ViTPose: yes/no (accuracy vs 2x compute).
- Keep the x-pose boxes or run a fresh detector pass?

## Risks

- ViTPose still COCO-domain: helps localization, not kid-proportion bias (Phase B
  measures this).
- `transformers` ViTPose API churn across versions; pin a known-good version.
- Top-down compute: per-person crops, ~3x frames; batch and use the A6000.
