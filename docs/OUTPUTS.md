# Outputs

What lives where after a pipeline run, with column dictionaries for every CSV. Use this to understand what to load into pandas/R/Excel and what each column means.

## Directory layout per session

```
video_inference/output/<session_id>/
├── session_summary.json                  # written by video-infer run
├── session_metrics.csv                   # ← `video-session-report`: one row per camera × block
├── report.html                           # ← `video-session-report`: one-page summary (start here)
└── <camera>/                             # camera_a, camera_b
    ├── manifest.json                     # provenance + assumptions
    ├── frames/                           # extracted JPGs
    │   └── frame_000001.jpg, ...
    ├── intermediate/
    │   ├── inference_raw.json            # backend's raw output (pre-tracking)
    │   └── detections.npz                # offline tracker: every detection + ReID embedding
    │
    ├── tracks_2d.csv                     # ← raw output of `video-infer`
    ├── pose_3d.csv
    │
    ├── tracks_2d_filtered.csv            # ← after `video-filter-tracks` or `offline_assoc`
    ├── pose_3d_filtered.csv
    ├── two_person_summary.json           # ← `offline_assoc`: frames with parent / child / both
    ├── pose_3d_refined.csv               # ← `topdown_refine`: ViTPose-refined keypoints
    ├── keypoint_refinement_summary.json
    │
    ├── track_corrections.json            # ← after `video-annotate-tracks`
    ├── tracks_2d_corrected.csv           # ← after `video-correct-tracks`
    ├── pose_3d_corrected.csv
    │
    ├── tracks_2d_interpolated.csv        # ← after `video-interpolate`
    ├── pose_3d_interpolated.csv
    │
    ├── tracks_2d_smooth.csv              # ← after `video-smooth`
    ├── pose_3d_smooth.csv
    ├── smoothing_summary.json
    │
    ├── gaze_heatmap.csv                  # ← after `gaze-infer`
    ├── gaze_heatmaps.npz
    │
    ├── synchrony_metrics.csv             # ← after `gaze-synchrony`
    ├── synchrony_summary.json
    ├── synchrony_dashboard.png           # ← after `gaze-plot`
    │
    ├── pose_metrics_summary.json         # ← `video-pose-metrics`: per block, incl. coverage
    ├── pose_metrics_proximity.csv
    ├── pose_metrics_xcorr.csv
    ├── pose_proximity.png / pose_synchrony.png (+ .svg)
    ├── gaze_metrics_summary.json         # ← `video-gaze-metrics`: per block, with chance levels
    ├── gaze_metrics_categories.csv
    ├── gaze_metrics_convergence.csv
    ├── gaze_categories.png / gaze_convergence.png (+ .svg)
    ├── gaze_labels.json                  # ← optional portal gaze labels (human coding)
    ├── overlay_<block>.mp4, overlay_<block>_end.mp4   # ← portal overlay clips
    │
    ├── block_checks/                     # ← after `video-gaze-snapshots`
    │   └── <block>_sample_*.png
    │
    └── <user-chosen>.mp4                 # ← after `video-visualize` (path is whatever you pass to --output-video)
    └── <user-chosen>/                    # ← after `video-visualize-3d` (path is whatever you pass to --output-dir)
```

Files only exist after the stage that produces them. Earlier-stage files are never overwritten — each cleanup stage writes a new suffix.

> **Always pass `--pose-input` and `--tracks-input` explicitly** to downstream CLIs. Defaults are inconsistent across modules (`video-smooth` expects `_interpolated.csv`, `video-gaze-metrics` expects `_filtered_5hz.csv`) and won't match the standard colleague workflow. Pick the suffix you actually have on disk and pass it.

---

## `manifest.json`

| Key | Type | Notes |
|---|---|---|
| `schema_version` | string | e.g. `"1.0"`. |
| `session_id` | string | Matches the session subfolder name. |
| `source_videos` | object | `{camera_a: <path>, camera_b: <path>}`. |
| `assumptions.max_persons` | int (≥1) | What `--max-persons` was set to. |
| `assumptions.enforce_exact_person_count` | bool | |
| `outputs.tracks_2d` | string | Relative path. |
| `outputs.pose_3d` | string | Relative path. |

Validated by `video_inference.schema.validate_session_output`.

---

## `tracks_2d.csv`

Per-frame bounding boxes. One row per (frame, person).

| Column | Type | Description |
|---|---|---|
| `frame_idx` | int | Sequential frame index, starts at 0. |
| `timestamp_s` | float | Seconds from video start (= `frame_idx / fps`). |
| `track_id` | int | `0..max_persons-1`. |
| `track_label` | string | Human-readable: `parent`/`child` in 2-person mode, or `person_XX` otherwise. Stable per `track_id` within a run. |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | float | Pixel coordinates in the **compressed** video frame. |
| `track_confidence` | float | `0..1`. Detection score. |

Constraints (enforced by `schema.validate_session_output`):
- No duplicate `(frame_idx, track_id)`.
- Each frame has 1..`max_persons` rows (or exactly `max_persons` when `--enforce-exact-person-count`).
- `track_id` ↔ `track_label` mapping is consistent across the file.

---

## `pose_3d.csv`

Per-keypoint pose data. **Despite the name, columns hold whatever the backend produced** — 2D keypoints from the Ultralytics or RTMLib `Body` backends use `z_m=0.0`; only `sam3d` and `rtmlib --rtmlib-3d` produce real depth.

| Column | Type | Description |
|---|---|---|
| `frame_idx` | int | Same as `tracks_2d.csv`. |
| `timestamp_s` | float | |
| `track_id` | int | Joins to `tracks_2d.csv`. |
| `track_label` | string | |
| `keypoint_name` | string | `kp_000`..`kp_016` — COCO-17 ordering: `0=nose, 1=L-eye, 2=R-eye, 3=L-ear, 4=R-ear, 5=L-shoulder, 6=R-shoulder, 7=L-elbow, 8=R-elbow, 9=L-wrist, 10=R-wrist, 11=L-hip, 12=R-hip, 13=L-knee, 14=R-knee, 15=L-ankle, 16=R-ankle`. |
| `x_m`, `y_m`, `z_m` | float | Coordinates. For 2D backends these are pixels in the compressed-video frame and `z_m=0.0` everywhere. For 3D backends, metres in camera space. |
| `keypoint_confidence` | float | `0..1`. |

Constraints:
- No duplicate `(frame_idx, track_id, keypoint_name)`.
- All coordinates finite.

Smoothing-stage files (`pose_3d_smooth.csv`) share the same schema; only values change.

---

## `gaze_heatmap.csv`

Written by `gaze-infer` (PaGE ViT-B; `gaze-infer-gazelle` writes the same
format with Gaze-LLE). One row per (frame, person). `gaze_summary.json` records
the model, its pinned revision, and the frame size used.

| Column | Type | Description |
|---|---|---|
| `frame_idx` | int | |
| `timestamp_s` | float | |
| `track_id` | int | |
| `gaze_peak_x` | float | Gaze target x as a **fraction of image width** (`0..1`; argmax of the 64×64 heatmap). |
| `gaze_peak_y` | float | Gaze target y as a fraction of image height. |
| `gaze_peak_value` | float | Peak heatmap intensity (`0..1`). |
| `inout_score` | float | Probability the gaze target is inside the image (`0..1`; <0.5 = likely looking out of view). |
| `head_source` | string | `keypoints` (head bbox derived from face keypoints) or `bbox_fallback` (top 30% of body bbox when face keypoints are low-confidence). |

Companion file `gaze_heatmaps.npz` holds the full 64×64 heatmaps. Load with `np.load(...)["heatmaps"]` — shape `(N, 64, 64)` aligned to the CSV rows.

---

## `synchrony_metrics.csv`

Written by `gaze-synchrony`. One row per frame (parent and child collapsed into pairwise metrics).

| Column | Type | Description |
|---|---|---|
| `frame_idx` | int | |
| `timestamp_s` | float | |
| `torso_distance_px` | float | Euclidean distance between parent and child torso centroids (mean of shoulders + hips), in compressed-frame pixels. |
| `gaze_category` | string | One of the categories below. |
| `gaze_convergence_score` | float | Overlap of the parent's and child's gaze heatmaps after removing their background (`0` = disjoint, `1` = identical). |

Gaze categories (`gaze_analysis/gaze_categories.py`). Each person's gaze point
is placed relative to the other person's boxes, in pixels:

| Category | Rule |
|---|---|
| `mutual_gaze` | Each gaze point lands on the other's face (head box enlarged 1.5×). |
| `parent_watching` / `child_watching` | One gaze point lands on the other (face or body); the other's does not land back. |
| `both_watching` | Each looks at the other, not face to face. |
| `joint_attention` | Both look at something in view, neither at the other, gaze points within 1.5 mean head-heights. |
| `independent` | Both look at different things in view. |
| `unknown` | A gaze is out of view (`inout_score` < 0.5) or a box is missing. |

Single-frame category changes are absorbed into the surrounding category.

Plus a windowed cross-correlation table under `synchrony_summary.json`:

```json
{
  "movement_xcorr": [
    {"window_start_s": 0.0, "window_end_s": 5.0, "peak_xcorr": 0.42, "peak_lag_s": 0.2},
    ...
  ]
}
```

---

## Per-block metrics

`video-pose-metrics` writes `pose_metrics_summary.json`: one object per block on
that camera's own clock.

| Key | Description |
|---|---|
| `block`, `block_start_s`, `block_end_s`, `block_frames` | The block and its length in frames. |
| `frames_with_parent`, `frames_with_child`, `frames_with_both` | Frames in which each person (or both) was tracked. |
| `coverage_both` | `frames_with_both / block_frames`. All dyad metrics use only those frames. |
| `low_coverage` | `true` when `coverage_both` < 0.5: interpret the block's metrics with care. |
| `proximity_mean/std/median/n_frames` | Torso distance (pixels). |
| `xcorr_mean/std`, `lag_mean_s/std_s`, `xcorr_n_windows` | Movement synchrony in 5 s windows. Windows sit inside unbroken stretches where both people are tracked and never cross a gap; stretches shorter than 5 s give no window. |

Frame-level tables: `pose_metrics_proximity.csv` and `pose_metrics_xcorr.csv`.

`video-gaze-metrics` writes `gaze_metrics_summary.json`: per block, `n_frames`
and `<category>_pct` for every category (summing to 100). It also writes
`<category>_chance_pct`, the same share with the child's gaze shifted 60 s in
time. That shift keeps seating positions but breaks the moment-to-moment
pairing, so a category means something only when it clearly exceeds its
chance level. `convergence_mean/std/n_frames` and `convergence_chance` give
the gaze-map overlap. Frame-level tables: `gaze_metrics_categories.csv` and
`gaze_metrics_convergence.csv`.

## `session_metrics.csv` and `all_sessions_metrics.csv`

`video-session-report` joins the per-camera summaries into one row per camera ×
block:
- `session_id`, `camera_id`, `block`, and the coverage columns above.
- Pose metrics, under their own names.
- Gaze metrics, prefixed `gaze_`.
- Whole-camera tracking stats from `two_person_summary.json`, prefixed `tracking_`.

`video_inference/output/all_sessions_metrics.csv` stacks every session's table.
The portal home page offers it as "Download all sessions table". `report.html`
shows the same numbers with coverage warnings, chance levels, figures, and
links to the overlay clips.

## `gaze_labels.json` (optional, portal gaze labeling page)

A human coder marks, for a sample of frames, what the parent and the child are
looking at:
- `target` is one of `partner_face`, `partner_body`, `object`, `elsewhere`, or
  `cant_tell`.
- `point` optionally records where they look, as `[x, y]` image fractions.

`gaze-label-eval --camera-dir <cam> --gaze gaze_heatmap.csv` scores any gaze
table against these labels.

---

## `track_corrections.json` (from `video-annotate-tracks`)

```json
{
  "<frame_idx>": {"<old_track_id>": <new_track_id>, ...},
  ...
}
```

`apply_corrections` does the swap atomically per frame, so `{"0": 1, "1": 0}` actually swaps the two IDs rather than mapping both to 1.

---

## Sanity-check files

| File | Produced by | Use |
|---|---|---|
| `overlay.mp4` / `overlay_<block>.mp4` | `video-visualize` (portal: first and last 30 s of each block) | Watch the pose tracking play back over the original frames. The fastest way to spot a swap or a missed detection. |
| `viz_3d/*.png` (+ optional `skeleton.mp4`) | `video-visualize-3d` | Confirm 3D pose looks plausible (limb lengths constant, no jitter to infinity). |
| `block_checks/<block>_sample_*.png` | `video-gaze-snapshots` | Confirm gaze heatmaps land in plausible places (e.g. on a toy during joint attention). |
| `synchrony_dashboard.png` | `gaze-plot` | 4-panel summary across the whole session, shaded by block. |

---

## EEG-side outputs (sync workflow only)

`sync_results.json` — produced by `eeg-sync`. Documented in [`../README.md`](../README.md#sync_resultsjson-schema).
