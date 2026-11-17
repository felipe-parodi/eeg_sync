# Portal Optional-Camera And Compressor Plan

## Goal

Make portal v1 easier for non-technical collaborators while keeping processing
local and reliable.

## Phase 1: Portal Iteration

- Allow either GoPro video to be omitted, while requiring at least one uploaded
  video.
- Add clear warning labels that one or two videos are allowed.
- Add separate Video A and Video B analysis-block text boxes.
- Apply extra-person exclusions independently to each camera's block list.
- Pass camera-specific analysis windows into `video-infer run`.
- Add hover info icons for form fields.
- Add a compact output guide for proximity, movement synchrony, and gaze
  estimation figures/tables.

## Phase 2: Standalone Compression Script

- Add a single-file standard-library Python compressor for collaborators.
- Support one file, multiple files, or a directory of videos.
- Prefer hardware H.264 encoders when available, with CPU fallback.
- Use portal-compatible defaults: 8 FPS, max width 854 px, no audio, aggressive
  compression.
- Print clear progress and write a JSON summary.

## Test Strategy

- Unit-test optional-video metadata validation.
- Unit-test per-camera processing command construction.
- Unit-test pipeline support for video A only, video B only, and per-camera
  analysis-window flags.
- Unit-test portal HTML contains optional-camera warnings, info icons, and output
  guide sections.
- Unit-test standalone compressor discovery and ffmpeg command construction.
