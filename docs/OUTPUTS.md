# Outputs

What lives where after a pipeline run, with column dictionaries for every CSV. Use this to understand what to load into pandas/R/Excel and what each column means.

## Directory layout per session

```
video_inference/output/<session_id>/
├── session_summary.json                  # written by video-infer run
└── <camera>/                             # camera_a, camera_b
    ├── manifest.json                     # provenance + assumptions
    ├── frames/                           # extracted JPGs
    │   └── frame_000001.jpg, ...
    ├── intermediate/
    │   └── inference_raw.json            # backend's raw output (pre-tracking)
    │
    ├── tracks_2d.csv                     # ← raw output of `video-infer`
    ├── pose_3d.csv
    │
    ├── tracks_2d_filtered.csv            # ← after `video-filter-tracks`
    ├── pose_3d_filtered.csv
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
    ├── pose_metrics_<block>.csv          # ← after `video-pose-metrics`
    ├── pose_metrics_<block>.png
    ├── gaze_metrics_<block>.csv          # ← after `video-gaze-metrics`
    ├── gaze_metrics_<block>.png
    │
    ├── block_checks/                     # ← after `video-gaze-snapshots`
    │   └── <block>_sample_*.png
    │
    └── viz/  or  overlay.mp4             # ← after `video-visualize`
    └── viz_3d/                           # ← after `video-visualize-3d`
```

Files only exist after the stage that produces them. Earlier-stage files are never overwritten — each cleanup stage writes a new suffix.

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

Written by `gaze-infer`. One row per (frame, person).

| Column | Type | Description |
|---|---|---|
| `frame_idx` | int | |
| `timestamp_s` | float | |
| `track_id` | int | |
| `gaze_peak_x` | float | x-pixel of the gaze heatmap argmax, in the compressed frame. |
| `gaze_peak_y` | float | y-pixel of the gaze heatmap argmax. |
| `gaze_peak_value` | float | Peak heatmap intensity (`0..1`). |
| `inout_score` | float | Gazelle's in-frame / out-of-frame score (`0..1`; <0.5 means gaze likely out-of-frame). |
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
| `gaze_category` | string | `mutual` (parent ↔ child), `joint` (both looking at a third location), `parent_watching` (parent → child), `child_watching` (child → parent), or `independent`. |
| `gaze_convergence_score` | float | Cosine similarity of parent and child gaze heatmaps (`0..1`). |

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

## Per-block metrics CSVs

`video-pose-metrics` writes `pose_metrics_<block>.csv`:

| Column | Description |
|---|---|
| `frame_idx`, `timestamp_s` | |
| `torso_distance_px` | (same as synchrony_metrics) |
| `movement_xcorr_peak`, `movement_xcorr_lag_s` | Windowed cross-correlation peak in the surrounding window. |

`video-gaze-metrics` writes `gaze_metrics_<block>.csv`:

| Column | Description |
|---|---|
| `frame_idx`, `timestamp_s` | |
| `gaze_category` | |
| `gaze_convergence_score` | |

Both also write a paired `.png`: per-block panels comparing parent-child metrics across the named `session_blocks` (colored by `session_blocks[i].color`).

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
| `overlay.mp4` | `video-visualize` | Watch the pose tracking play back over the original frames. The fastest way to spot a swap or a missed detection. |
| `viz_3d/*.png` (+ optional `skeleton.mp4`) | `video-visualize-3d` | Confirm 3D pose looks plausible (limb lengths constant, no jitter to infinity). |
| `block_checks/<block>_sample_*.png` | `video-gaze-snapshots` | Confirm gaze heatmaps land in plausible places (e.g. on a toy during joint attention). |
| `synchrony_dashboard.png` | `gaze-plot` | 4-panel summary across the whole session, shaded by block. |

---

## EEG-side outputs (sync workflow only)

`sync_results.json` — produced by `eeg-sync`. Documented in [`../README.md`](../README.md#sync_resultsjson-schema).
