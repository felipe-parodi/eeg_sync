# Pose Quality Overhaul Plan

## Goal

Raise the share of frames where both the parent and the child are detected,
keep their identities correct, and improve keypoints, for the portal's
2-person pose pipeline. Scope is the pose estimator and tracker. Downstream
metrics are out of scope.

## Evidence from the 2026-10-09 review

- Uploads are 960x540, 10 fps, about 50-85 kbps. This is the same footage the
  `yolo11l-pose-dense` student was trained on. Extraction resamples it to
  6 fps, which makes frame timing uneven.
- Across 37 camera views, the median share of frames with both people
  detected is about 0.55, and several views are below 0.2.
- I sampled 960 frames where only one person was kept and reran detection at
  conf 0.1. 50% have no second detection, which is mostly true absence. 31%
  have a second detection only at 0.1-0.3. 12% have a second detection at
  conf >= 0.3 that the tracker discarded. 7% have one removed by the extra
  greedy NMS at IoU 0.45, which hits lap-sitting and hugging.
- BoT-SORT setup problems:
  - The lost-track buffer is scaled for frame rate twice and ends up at
    2 frames.
  - `new_track_thresh` is 0.6.
  - ECC camera-motion compensation runs on a static camera.
  - Top-2 selection by confidence happens before tracking.
- The colleague's `anchors.json` files hold human parent/child labels on about
  300 frames across 20 sessions. These are the only human identity ground
  truth.

## Phase 0: Benchmark harness

- `pose_training/bench.py` does three things:
  - caches per-frame detections at conf 0.05 (boxes, keypoints, ReID
    embeddings) for every portal session with frames on disk;
  - runs a named tracking/association variant from the cache;
  - scores the variant.
- Metrics:
  - **two_person_rate**: share of frames with both roles present.
  - **anchor_role_acc**: share of human anchor boxes matched (IoU >= 0.5) by
    the variant's box with the correct role.
  - **anchor_recall**: share of anchor boxes covered by any output box.
  - **role_flips_per_min**: how often the parent and child boxes swap
    positions between consecutive frames.
  - **static_fp_rate**: share of output boxes that do not move all session.
- Tests use synthetic detections for the scoring functions.

## Phase 1: Config fixes in the current runner

- Fix the BoT-SORT buffer double-scaling, set `new_track_thresh` to about 0.35,
  turn off CMC, and feed detections with conf >= 0.1 to the tracker.
- Keep the top 2 after tracking instead of before.
- Replace the greedy box NMS with keypoint-aware (OKS) NMS.
- Evaluate on the benchmark.

## Phase 2: Offline two-identity association

- New module `video_inference/offline_assoc.py`:
  1. Build high-precision tracklets.
  2. Link the tracklets into exactly two identities. Linking uses appearance
     (ReID), size, motion and a learned adult/child score, with the anchors
     as hard constraints.
  3. Fill frames with low-confidence detections that fit a predicted track.
- Static false-positive suppression for each camera.
- Evaluate with leave-anchors-out: hold out half the anchors as test labels.

## Phase 3: Model experiments, ranked by the benchmark

- Open-vocabulary or appearance adult/child classifier (OWLv2 / CLIP on
  crops), scored against the anchor role labels.
- Test-time options on the failure frames: imgsz 1280, flip TTA, SAHI-style
  tiling.
- SAM 2 video propagation started from the anchor boxes on hard sessions.
- Top-down keypoint refinement (ViTPose / RTMPose-x on tracked boxes), judged
  by blind A/B with `pose_training/verify.py`.

## Phase 4: Integration

- Wire the winners into `video_infer run` and the portal defaults.
- Choose an extraction fps that divides 10 fps evenly (5 or 10).
- Update tests and documentation.
- Run the full benchmark on all sessions and produce a before/after gallery
  for human visual sign-off.

## Validation rule

Programmatic metrics rank the candidates. Final acceptance needs human visual
review of the overlays and the A/B galleries.

## Results (2026-10-09)

Benchmark: 31 camera views from 20 portal sessions. Role accuracy is scored on
held-out halves of the colleague's anchor labels.

| Pipeline | Frames with both people | Role accuracy | Child-taller frames | Confident deployed boxes kept |
|---|---|---|---|---|
| Deployed (BoT-SORT + nearest-anchor roles) | 0.520 | 0.784 | 0.156 | 1.000 |
| BoT-SORT with corrected settings | 0.601 | 0.855 | 0.135 | - |
| Offline association, no anchors | 0.626 | 0.810 | 0.171 | - |
| **Offline association (shipped)** | **0.726** | **0.992** | **0.100** | **0.999** |

Phase 3 results:

- Larger inference size (1280) recovered a confident second person in only 3.6%
  of the remaining single-person frames (0.7% at 960). Flip TTA had no effect.
  Detection recall is near its ceiling for 960x540 footage. Not shipped.
- ViTPose-plus-Huge on tracked boxes reduced keypoint jitter by a median ~38%
  (8 of 8 person-tracks). It can lock onto the other person in overlapping
  boxes, so it is gated by keypoint similarity to the YOLO skeleton. Shipped
  as `video_inference/topdown_refine.py`.

Known failures:

- Confident model false positives on clutter (a coat on a chair in P003
  camera_a, a curtain fold at the lens in 1019 camera_a). Motion and keypoint
  quality do not separate them from still, reading parents. Fix at the model
  (hard negatives in the next distillation round) or with a "not a person"
  anchor button.
- One box that covers an adult and a child lying together gets a single role.
- Overlap (parent bending over child) still produces mixed skeletons.

Not changed: extraction at 6 fps from 10 fps uploads gives uneven frame
timing (0.1/0.2 s steps). 5 or 10 fps would make it uniform.
