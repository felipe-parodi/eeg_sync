# Analysis-Window-Only Inference Plan

## Goal

Run video inference only on post-exclusion analysis windows while preserving
original source-video timestamps in all downstream outputs.

## Constraints

- Keep default `video-infer run` behavior unchanged unless a new flag is used.
- Keep `yolo11m-pose.pt` for the portal.
- Preserve original source-video timestamps in `tracks_2d.csv`, `pose_3d.csv`,
  overlays, metrics, and gaze inputs.
- Exclude extra-person windows before inference.
- Avoid clip concatenation that loses the source-video timeline.
- Keep outputs compatible with existing schema and downstream CLIs.

## Phase 1: Segment-Aware Extraction

- Add parsing for `name,start,end;name,start,end` analysis windows.
- Add segmented frame extraction that writes only requested windows.
- Preserve original timestamps in `frame_index.csv`.
- Make the inference pipeline use `frame_index.csv` timestamps instead of
  assuming `timestamp_s = frame_idx / fps`.
- Update track filtering to use `timestamp_s` windows so segmented frame indices
  remain valid.
- Wire the portal's post-exclusion windows into `video-infer run`.

## Phase 2: Reporting And QA

- Record analysis-window-only mode in portal metadata and session summaries.
- Add clearer job status around segmented extraction.
- Add a small synthetic integration test with multiple windows and gaps.

## Test Strategy

- Unit-test analysis-window parsing.
- Unit-test dry-run ffmpeg commands for segmented extraction.
- Unit-test timestamp preservation in generated frame indices.
- Unit-test pipeline CSV timestamp mapping.
- Unit-test portal command construction includes `--analysis-windows`.
- Unit-test track filtering with sparse segmented `frame_idx` and original
  `timestamp_s`.
