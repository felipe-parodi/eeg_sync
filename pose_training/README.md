# pose_training

Pseudo-label the project's compressed session videos with a strong teacher
(`yolo11x-pose`) and fine-tune a deployable student (`yolo11l-pose`) on the
result, to improve pose quality on this domain (parent-child interaction,
overhead/wide GoPro angles, occlusion). The keypoint layout stays COCO-17, so
every downstream stage (gaze head bbox, overlays, metrics) works unchanged.

Plan and rationale: [`docs/plans/pseudo-label-pose-train_260531.md`](../docs/plans/pseudo-label-pose-train_260531.md).

Labeling is fully automatic (no manual keypoint correction). The teacher is the
quality ceiling, so the pipeline invests in a strong teacher pass (higher
inference resolution + manual horizontal-flip TTA) and strict automatic
filtering rather than hand-fixing labels.

## Pipeline

Run from the repo root with the CUDA env (`uv run ...`). All generated data
lands under the gitignored `pose_training/data/` and `pose_training/runs/`.

```bash
# 0. Pre-flight: CUDA, GPUs, ffmpeg, weights, flip_idx
uv run python -m pose_training.preflight

# 1. Sample + dedup frames; participant-disjoint train/val/test split
uv run python -m pose_training.sampling --sample-fps 0.25 \
    --max-frames-per-video 400 --dedup-threshold 4

# 2. Teacher pseudo-labels (yolo11x-pose, flip-TTA, imgsz 960, all people)
uv run python -m pose_training.pseudo_label --imgsz 960 --device 0 --max-persons 6

# 3. QC confidence plot + filtering + YOLO dataset export
uv run python -m pose_training.dataset

# 4. Fine-tune the student from COCO weights
uv run python -m pose_training.train --epochs 100 --imgsz 960 --batch 16 --device 0

# 5. Evaluate vs the stock portal model (metrics + confidence plot + overlay clip)
uv run python -m pose_training.evaluate \
    --new-weights pose_training/runs/student_l_v1/weights/best.pt \
    --overlay-video compressed_video/1014_FULL_SHORT_compressed.mp4
```

## Modules

| Module | Purpose |
|---|---|
| `coco.py` | COCO-17 keypoint names, `flip_idx`, skeleton, OKS sigmas (single source of truth). |
| `preflight.py` | Environment checks before any heavy step. |
| `sampling.py` | Even frame sampling, dHash dedup, participant-level split, manifest. |
| `pseudo_label.py` | Teacher inference with manual horizontal-flip TTA; keeps all people. |
| `dataset.py` | Confidence-distribution QC plot, automatic filtering, YOLO pose export. |
| `train.py` | Student fine-tuning wrapper over Ultralytics. |
| `evaluate.py` | Held-out metrics, old-vs-new confidence plot, side-by-side overlay clips. |

## Notes

- `max_persons=6` at label time intentionally labels every visible person (not
  just parent/child) so the detector learns to find everyone; parent/child
  assignment stays a downstream step (`video-filter-tracks`).
- Temporal-jump filtering is not used: frames are ~4 s apart, so real motion
  would be flagged as error. Filtering relies on per-keypoint confidence +
  geometry.
- Test-split metrics measure agreement with the teacher on unseen participants,
  not ground-truth correctness. Human side-by-side overlay review is the
  acceptance gate.
