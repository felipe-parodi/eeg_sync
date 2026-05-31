# Pseudo-Label and Train a Domain-Adapted YOLO11 Pose Model

Plan date: 2026-05-31. Convention: numbered phases for the `/phase` skill.

## Goal

Improve portal pose quality on parent-child interaction footage by
pseudo-labeling the 35 compressed session videos with a strong teacher
(`yolo11x-pose`) and fine-tuning a deployable student (`yolo11l-pose`) on the
result. Keep the COCO-17 keypoint layout so every downstream stage (gaze head
bbox, overlays, metrics) works unchanged. Labeling is fully automatic; no manual
keypoint correction in this iteration.

## Decisions (locked with the user)

- Teacher: `yolo11x-pose`. Student: `yolo11l-pose`. Skeleton: COCO-17.
- Labeling: auto only. No human keypoint correction this pass.
- Acceptance: human side-by-side overlay review on held-out participants, plus a
  confidence-distribution plot and OKS agreement on a small hand-checked set.
- Reported failure modes to target: small children / seated poses, overhead
  (TALL) and wide-angle distortion, parent-child occlusion and frame-edge
  truncation, and frame-to-frame jitter / dropped low-confidence joints.

## Honest scope and the auto-only ceiling

Self-training cannot exceed the teacher on systematic errors. `yolo11x-pose` is
trained on the same COCO distribution as the current L model, so it carries the
same weaknesses on young children, overhead views, and heavy occlusion. Expect:

- Large gains: jitter, confidence calibration, dropped joints, detection
  stability. The student learns a smoothed, more consistent version of the
  teacher.
- Modest gains: kids, overhead, occlusion. These need either a genuinely
  stronger teacher signal or human correction.

To raise the ceiling without manual work, the teacher runs with test-time
augmentation and the pseudo-labels pass strict automatic filtering (Phase 3).
If Phase 5 still shows weak kids/overhead cases, the documented next step is a
small active-learning correction pass; this plan does not assume it.

## Environment and compute (pre-flight)

- GPUs present: RTX A6000 (48 GB) and RTX 4090 (24 GB). Training target: 4090,
  with A6000 as fallback or for the X teacher pass.
- The default `python` on PATH is `torch 2.3.1+cpu`. Training and GPU teacher
  inference need a CUDA-enabled environment. Create a separate `uv` env with a
  CUDA build of torch; do not disturb the existing CPU env the portal uses.
- Weights to fetch once: `yolo11x-pose.pt`, `yolo11l-pose.pt` (already present),
  `yolo11m-pose.pt`. `*.pt` is gitignored; never commit weights.

## Data inventory

- `compressed_video/`: 35 clips, 18 participants (1010-1020, P001-P009) x 2
  angles (TALL, SHORT), 960x540 at 10 fps, 23-106 min each.
- Labels are generated on this compressed domain on purpose: the portal runs on
  similarly compressed footage, so train and deploy distributions match.
- Volume is large and highly redundant (hundreds of thousands of near-duplicate
  frames), so Phase 1 samples and deduplicates rather than labeling every frame.

## Code and data layout

- New package `pose_training/` with small, testable modules plus thin CLIs,
  mirroring the existing `video_inference` / `video_analysis` style (Google
  docstrings, type hints, `build_arg_parser` + `main`).
- Generated data under a gitignored path (e.g. `pose_training/data/`): sampled
  frames, label txts, `data.yaml`, split manifests, run outputs. Confirm the
  ignore rule before writing.
- Reuse `video_inference/frames.py` for extraction and the COCO-17 conventions
  already encoded in `visualize_pose_tracks.py` and `head_bbox.py`.

## Phase 0: Setup and pre-flight

- Create the CUDA `uv` env; verify `torch.cuda.is_available()` and both GPUs.
- Download teacher/student weights; confirm `yolo11x-pose` loads and runs on one
  test frame.
- Add `pose_training/data/` to `.gitignore` if not already covered.
- Test: a pre-flight script that asserts CUDA, GPU names, weight files, and
  ffmpeg/ffprobe presence, and prints a one-line ready/not-ready summary.

## Phase 1: Frame sampling and deduplication

- Sample candidate frames per video at a low rate (start 1 fps; tunable) using
  the existing ffmpeg extraction path.
- Deduplicate near-identical frames (perceptual hash or small-embedding cosine
  distance) so static stretches do not dominate.
- Stratify the kept set across participant x angle so overhead (TALL),
  occlusion-prone, and seated frames are represented, not just easy standing
  shots. Target first training set: ~3,000-6,000 frames total (tunable).
- Hold out 3 participants entirely for the Phase 5 test set; they never enter
  training. Split the remaining 15 into train/val by participant (no frame from
  one participant in both splits) to prevent leakage.
- Output: `frame_manifest.csv` (frame path, participant, angle, split, source
  video, source timestamp).
- Test: manifest has no participant in two splits; per-stratum counts are
  non-zero; sampled frame count matches the manifest.

## Phase 2: Teacher pseudo-labeling

- Run `yolo11x-pose` on the sampled frames with horizontal-flip TTA and, if
  affordable, multi-scale, at a higher `imgsz` than the portal (e.g. 960-1280)
  to get the best keypoints the teacher can produce. This is still fully
  automatic.
- Keep up to 2 person instances per frame (parent and child); export each
  instance's bbox, 17 keypoints, per-keypoint confidence, and instance
  confidence to a raw JSON/parquet, preserving participant/angle/frame keys.
- Reuse the multi-person handling already in `ultralytics_runner.py` rather than
  re-implementing NMS and person selection.
- Test: output row count matches input frames; per-instance arrays are length-17
  with finite coordinates; confidence in [0, 1].

## Phase 3: Pseudo-label QC, filtering, and dataset export

- Confidence distribution analysis (the plot the user asked for): histograms of
  per-keypoint and per-instance confidence across the whole set, split by angle
  and by keypoint group (head vs limbs), before and after filtering. Save PNG +
  SVG at 300 dpi, color-blind palette, per the figure rules.
- Automatic filters (no manual work):
  - Instance confidence below a threshold (start ~0.5) drops the instance.
  - Per-keypoint: low-confidence joints marked not-visible (v=0) rather than
    fed as confident truth.
  - Temporal consistency: within a participant/angle track, flag keypoints that
    jump implausibly between adjacent sampled frames and demote them.
  - Geometry sanity: drop instances with degenerate or out-of-frame boxes and
    anatomically impossible limb lengths.
- Export YOLO pose format: per-image `.txt` (`class cx cy w h` + 17*(x,y,v),
  normalized), images dir, and `data.yaml` with `kpt_shape: [17, 3]` and the
  correct COCO `flip_idx`. The flip_idx must swap left/right joints; a wrong
  flip_idx silently corrupts every left/right keypoint under horizontal-flip
  augmentation, so unit-test it explicitly.
- Output: `data.yaml`, `images/{train,val}`, `labels/{train,val}`, plus a
  retained test set of held-out participants.
- Test: YOLO label parsing round-trips; normalized coords in [0, 1]; flip_idx is
  a valid COCO permutation; counts match the post-filter manifest.

## Phase 4: Train the student

- Fine-tune `yolo11l-pose` from the COCO-pretrained checkpoint (not from
  scratch) on the 4090. Start ~100 epochs, `imgsz` matched toward deployment
  (e.g. 960), standard pose augmentation, early-stopping on val pose mAP.
- Log to the run dir; keep `best.pt`. Do not commit weights.
- Ask the user before any change to portal memory/cleanup behavior in training
  scripts, per the memory-management rule.
- Test: a tiny smoke run (few images, 1-2 epochs) produces a loadable `best.pt`
  and a results CSV; assert the run completes and metrics are finite.

## Phase 5: Evaluation and acceptance gate

- Metrics: val/test pose mAP and OKS; confidence distribution of the new L model
  vs the current L model on the held-out participants; OKS agreement against a
  small (~150-frame) hand-checked gold set. Note plainly that agreement-with-
  teacher measures consistency, not correctness; the gold set and human review
  are what establish correctness.
- Human review (the real gate): render old-L vs new-L overlay clips on held-out
  participants, both angles, including seated/overhead/occlusion segments. The
  colleague judges. Per the validation rules, do not assert the output is
  correct from metrics alone.
- Acceptance: proceed to Phase 6 only if the colleague confirms the new model is
  visibly better on the targeted failure modes and metrics do not regress.

## Phase 6: Integration (only after acceptance)

- Place the new weights where the portal can load them; point
  `PORTAL_DEFAULT_MODEL_PATH` (`portal/jobs.py`) at the new file, keep the stock
  `yolo11l-pose.pt` as a documented fallback.
- Update `tests/test_portal_jobs.py` expectations and the model references in
  `README.md`, `docs/PIPELINE.md`, `docs/COLLEAGUE_QUICKSTART.md`,
  `video_inference/README.md`.
- Conventional commits per phase; feature branch; no force push.

## Risks and limitations

- Auto-only ceiling: limited gains on kids/overhead/occlusion (see above).
- Pseudo-label echo: the student can amplify confident teacher mistakes;
  Phase 3 filtering and the held-out human review are the guardrails.
- Eval-vs-teacher is not ground truth: correctness rests on the gold set and
  human overlays, not on OKS agreement.
- flip_idx correctness is load-bearing; it is unit-tested in Phase 3.
- Two-person completeness: frames where the teacher finds only one person are
  kept as single-instance labels, not discarded, to avoid teaching the student
  that a second person is absent.

## Open tunables (defaults chosen; adjust after Phase 3)

- Sampling rate (1 fps), target frame count (3-6k), dedup threshold.
- Instance/keypoint confidence thresholds (0.5 / per-keypoint demotion).
- Teacher TTA/multi-scale and `imgsz`; student `imgsz` and epoch count.
- Number of held-out test participants (3).
