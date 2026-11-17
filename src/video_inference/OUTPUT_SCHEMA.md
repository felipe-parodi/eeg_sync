Video Inference Output Schema (Phase 2 Draft)
=============================================

This document defines the minimal output contract for `video_inference/`.

Goals
-----
- Keep outputs deterministic and machine-parseable.
- Support configurable multi-person tracking via `assumptions.max_persons`.
- Preserve stable identity mapping across frames.

Required files per session
--------------------------
- `manifest.json`
- `tracks_2d.csv`
- `pose_3d.csv`

`manifest.json` required keys
-----------------------------
- `schema_version`
- `session_id`
- `source_videos`
- `assumptions.max_persons` (integer >= 1)
- `assumptions.enforce_exact_person_count` (boolean)
- `outputs.tracks_2d`
- `outputs.pose_3d`

`tracks_2d.csv` required columns
--------------------------------
- `frame_idx`
- `timestamp_s`
- `track_id` (`0..max_persons-1`)
- `track_label` (stable per `track_id`)
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`
- `track_confidence` (`0..1`)

Rules:
- if `enforce_exact_person_count=true`, each frame must have exactly `max_persons` rows
- otherwise each frame must have `1..max_persons` rows
- no duplicate `(frame_idx, track_id)`
- `track_id` -> `track_label` mapping must remain consistent

`pose_3d.csv` required columns
------------------------------
- `frame_idx`
- `timestamp_s`
- `track_id`
- `track_label`
- `keypoint_name`
- `x_m`, `y_m`, `z_m`
- `keypoint_confidence` (`0..1`)

Rules:
- no duplicate `(frame_idx, track_id, keypoint_name)`
- finite 3D coordinates
- `track_id` values must stay within `0..max_persons-1`

Synthetic fixture
-----------------
- A fully synthetic example session is stored under:
  `tests/fixtures/video_inference/mock_session/`
- Use this fixture for tests and local development instead of real participant data.
