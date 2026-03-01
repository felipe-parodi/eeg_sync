Video Inference Output Schema (Phase 2 Draft)
=============================================

This document defines the minimal output contract for `video_inference/`.

Goals
-----
- Keep outputs deterministic and machine-parseable.
- Enforce exactly two tracked identities for this project.
- Preserve a stable parent/child mapping across frames.

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
- `assumptions.max_persons` (must be `2`)
- `outputs.tracks_2d`
- `outputs.pose_3d`

`tracks_2d.csv` required columns
--------------------------------
- `frame_idx`
- `timestamp_s`
- `track_id` (`0` and `1`)
- `track_label` (`parent` and `child`)
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`
- `track_confidence` (`0..1`)

Rules:
- each frame must have exactly 2 rows (one per track)
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
- same `track_id` set as `tracks_2d.csv`

Synthetic fixture
-----------------
- A fully synthetic example session is stored under:
  `tests/fixtures/video_inference/mock_session/`
- Use this fixture for tests and local development instead of real participant data.
