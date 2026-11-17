# Portal Roadmap

This document captures likely next steps for the web portal after the first
upload-to-results loop is working reliably.

## V1 Validation Priorities

Before adding new features, test the current portal with a short compressed
session. The first checks should confirm that:

- The shared-password login works from a collaborator laptop through the public
  tunnel.
- Two uploaded videos are written to the local workstation.
- Oversized inputs trigger compression before inference.
- Analysis blocks and exclusion windows produce the expected split windows.
- The job status page updates through processing.
- Results are packaged into a downloadable zip.
- Email notification works, or an `email_preview.txt` is written when SMTP is
  not configured.

Only after this path works should we test multi-gigabyte GoPro uploads.

## Upload Speed Roadmap

The main speed limit for remote collaborators is uploading raw GoPro files before
the workstation can compress them. The preferred v1 workflow for full-size
recordings is:

1. give collaborators a one-click local compressor,
2. upload the compressed videos through the portal,
3. keep server-side NVENC compression as a validation/fallback step for files
   that are still too large.

The portal now uploads both cameras in parallel and writes chunks directly into
their final video files, so it no longer does a separate chunk-stitch copy.
The portal also runs only one GPU processing job at a time to avoid local
workstation memory exhaustion.

The portal now runs inference only on requested analysis blocks. Segment-aware
frame extraction preserves original source-video timestamps in `frame_index.csv`
and downstream CSV outputs, so filtering and metrics stay aligned to the times
entered by the researcher.

The next processing speedups are to preflight video duration before upload,
surface upload throughput in the browser, and avoid re-running downstream stages
when a failed job is retried after a fix.

## Parent/Child Review From Sample Frames

A strong next portal feature is a lightweight identity-review step before final
metrics. The portal could sample about 10 frames across each user-entered
analysis block, render detector boxes on those frames, and ask the researcher to
click which box is the parent and which box is the child.

This review could support the pipeline in two ways:

- **Verification:** compare the selected boxes against the inferred
  `track_id=0` and `track_id=1` assignments after `video-filter-tracks`.
- **Constraint:** use the selected identities as anchor points for downstream ID
  correction, especially when parent/child size assumptions are unreliable.

The first implementation should probably be verification-only. It would flag
sessions for review when sampled frames disagree with the automatic assignment.
Using the clicks as hard constraints should come later, because incorrect
clicks could otherwise corrupt an entire session.

## 3D Tracking Roadmap

There are two reasonable paths for 3D parent-child tracking.

### Camera Calibration

If both GoPros view the same interaction space, a calibrated multi-camera setup
could triangulate 3D pose from corrected 2D tracks. This is the more
geometrically grounded approach, but it requires a repeatable calibration
workflow:

- collect calibration footage for each camera pair,
- estimate intrinsics and extrinsics,
- synchronize camera timestamps,
- validate reprojection error,
- triangulate only after parent/child IDs are corrected.

This path is best if the physical camera setup is stable across sessions.

### 3D Lifter Model

An alternative is to run a 2D-to-3D lifter model on the two corrected
parent/child tracks. This avoids camera calibration and may be easier to deploy
for v2. The tradeoff is that lifted 3D coordinates are model-dependent and may
not represent metric depth accurately.

This path is best if we mainly need relative body posture or movement features,
not precise physical distances in the room.

## Recommended Order

1. Stabilize portal v1 on short compressed videos.
2. Test one full-size two-camera upload over the Penn VPN.
3. Add job cancellation, retry, and clearer failure messages.
4. Add sampled-frame parent/child verification.
5. Decide whether 3D should be calibration-based, lifter-based, or both.
