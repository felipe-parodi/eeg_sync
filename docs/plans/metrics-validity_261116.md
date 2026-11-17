# Metrics Validity Plan

## Goal

Make per-block pose metrics trustworthy enough to set a lab-wide inclusion rule
before more sessions are processed. Runs in parallel with
`portal-revamp_261116.md`; the two meet at the column contract below.

Branch: `feat/metrics-validity` (from `feat/pose-quality-overhaul`).

## Evidence (2026-11-16, sessions 1011v3-1020v3 in `video_inference/output/`)

- Low coverage is almost always the child missing; parent is tracked in ~100% of
  frames in nearly every block.
- 1019v3 camera_a: an out-of-focus object covers ~half the frame. The model
  detects it (and once the doorway) as a person; by a box-shape heuristic the
  "child" in grocery_part1 is the object in ~73% of its rows and the "parent" in
  grocery_part2 in ~88%. Keypoint confidence on the object is 0.66-0.79, the same
  range as real people, so keypoint confidence cannot gate quality.
- Movement synchrony is at chance: 1016v3 camera_a (100% coverage) grocery 0.512
  vs 0.498 with child shifted 60 s; storybook 0.478 vs 0.469. `xcorr_mean` sits
  at 0.45-0.59 in every block of every session.
- `compute_movement_xcorr` windows are counted in frames over the merged series,
  so a "5 s" window spans gaps: 19-25% of windows exceed 10 s in 1017v3/1019v3
  (max 313 s). Velocity central differences also cross gaps.
- `tracking_*` columns in `session_metrics.csv` are whole-video values repeated on
  every block row.

## Column Contract (shared with the portal revamp)

Added to `session_metrics.csv` (one row per camera x block):

| column | meaning |
|---|---|
| `valid_seconds_both` | seconds with both people tracked (`frames_with_both / frame_rate`) |
| `n_segments` | contiguous both-tracked runs (gap > 1 s splits) |
| `longest_segment_s` | longest run, seconds |
| `suspect_person_share` | share of both-tracked frames where either box fails the false-person check |
| `xcorr_chance_mean` | same xcorr with child shifted 60 s within the block |
| `xcorr_minus_chance` | `xcorr_mean - xcorr_chance_mean` |
| `quality_verdict` | `usable` / `check` / `insufficient` |
| `quality_reasons` | short plain-language reasons, `;`-separated |

New `task_metrics.csv`: one row per camera x task (`<task>_partN` rejoined).

## Phase 0: Frame timing and gap filling (added 2026-11-16, done before Phase 3)

Found by an exploratory bug hunt and checked on 1016v3/1019v3: uploads are 10 fps
and `fps=6` keeps frames at 100/200/200 ms steps while stamping them 167 ms apart,
so displacement repeats every 3 frames (0.72/0.99/0.98 of the mean) in both
people. The user chose 5 fps.

- `video_inference/frames.py`: probe the source rate; keep every k-th source frame
  (k = round(source fps / 5)), counted from the start of the video so every block
  lands on one shared grid; stamp each frame with its true source time (ffmpeg
  `showinfo`); write `frames/extraction_info.json` (source fps, k, achieved rate).
- Portal compressors (`portal/jobs.py`, `scripts/compress_videos_for_portal.py`)
  output 10 fps (was 8, which can only give an even 4 fps).
- `PORTAL_DEFAULT_FRAME_RATE = 5.0`; after phase 1 the portal reads the achieved
  rate per camera, uses it downstream, and stops with a plain message if the
  cameras disagree.
- `video_analysis/interpolate.py`: fill only at extracted frame times
  (`frame_index.csv`), only gaps no longer than `max_gap_seconds` between two
  observations, and never across an extraction window. (Bug hunt findings 2-3:
  `limit=` partly filled every gap; off-grid blocks were dropped.)
- Coverage counts frames from `frame_index.csv` when available.
- Sessions must be re-run from frame extraction afterwards.
- Note: `offline_assoc` frame-count thresholds were tuned at 6 fps (e.g. 18
  frames = 3 s); check them at 5 fps.

## Phase 1: Gap-aware, time-based movement synchrony

- In `gaze_analysis/synchrony.py`, split the merged parent/child series into
  contiguous runs (gap > 1.5 median frame intervals).
- Compute velocity within runs only.
- Define windows in seconds inside each run (`window_s`, 50% overlap); drop runs
  shorter than one window.
- Add `segment_id` to `pose_metrics_xcorr.csv`.
- Tests first: a synthetic series with a 60 s gap never yields a window spanning
  it; every window span <= `window_s` + one frame; a run shorter than `window_s`
  yields no window.

## Phase 2: Synchrony chance baseline (done; result: real minus chance = 0.008 on average, sd 0.033, n = 34 blocks)

- Circularly shift the child series by 60 s within the block (same convention as
  gaze chance), recompute, and write `xcorr_chance_mean` and
  `xcorr_minus_chance`.
- Tests: identical signals score well above chance; independent noise gives
  `xcorr_minus_chance` near 0; when less than 2 x shift of usable data exists,
  shift by half of it (same rule as the gaze chance level).
- Re-check 1016v3 and 1018v3 after phases 1-2. If synchrony is still at chance,
  report that to the user before going further (possible next steps: lag-0
  correlation, longer windows, or keypoint-level motion energy instead of torso
  speed).

## Phase 2b: Alternative synchrony measures (done 2026-11-16; adopted whole-body lag-0 20 s)

Result on 1011v3-1020v3 (38 blocks; real minus 60 s-shift chance): current
torso peak 5 s +0.011 (effect 0.27); torso lag-0 20 s +0.127 (1.44); whole-body
motion energy lag-0 20 s +0.172 (1.68). Controls: regressing out background
motion +0.16 to +0.18; excluding frames with parent/child keypoints within
10 px +0.13 to +0.16; cameras static. Box-overlap exclusion is untestable in
storybook (boxes overlap in 93-97% of frames). Implemented as
`compute_movement_sync` (`sync_*` columns) with the chance level pooled over
several shifts (single-shift chance varied up to 0.17 per block); xcorr kept
as superseded.

## Phase 2b (original plan text)

After Phase 0 re-extraction, compare on 1011-1020: lag-0 correlation, longer
windows (10-30 s), and whole-body motion energy (all keypoints) against their
chance baselines. Keep the measure that clearly separates real from shifted
data; otherwise drop movement synchrony from the portal and report.

## Phase 3: False-person check

- Flag a track row as suspect when its box is static over a long stretch (box
  center and size nearly constant over >= 10 s), or matches a recurring static
  region in the detection cache.
- Write a per-frame `suspect` flag and the block-level `suspect_person_share`.
- Validate on 1019v3 (expect high share in grocery) and 1016v3/1018v3 (expect
  ~0). Render flagged frames to PNG for **user visual confirmation**; do not
  claim correctness from numbers alone.

## Phase 4: Valid time and verdict (done 2026-11-16: defaults 60 s / 50% / 3 windows; `suspect_person_share` waits for Phase 3; tracking_* columns left as they are)

- Add `valid_seconds_both`, `n_segments`, `longest_segment_s`.
- Replace the single 50% flag with `quality_verdict` + `quality_reasons`, driven
  by valid seconds and suspect share. Coverage stays as a reported column but
  no longer decides the verdict alone.
- Thresholds live as named constants in `pose_metrics.py`; the starting values
  are an **open question for the lab** (see below).
- Stop repeating whole-video `tracking_*` values per block; move them to
  `session_summary.json` / report header.

## Phase 5: Task-level table

- `video_analysis/session_report.py`: write `task_metrics.csv` by rejoining
  `<task>_partN`.
  - Proximity: pool per-frame distances across parts, recompute mean/median/sd
    (never average part medians).
  - Synchrony: pool windows across parts (windows never cross part boundaries,
    by construction).
  - Gaze shares: pool frames.
  - Sum valid seconds; verdict from pooled values.
- Add `proximity_median_bodylen`: distance divided by the parent's median
  shoulder-to-hip length in that block, so values compare across cameras and
  sessions.
- Append task rows to the study-wide table as `all_sessions_task_metrics.csv`.
- Tests: two parts with known frames give the pooled mean/median; part medians
  are not averaged.

## Phase 5b: Report and study table hygiene (bug hunt)

- Refuse or archive an existing `output/<session_id>` folder at submit, so a
  re-run never mixes old and new rows (finding 5; portal side in revamp Phase 3).
- `all_sessions_metrics.csv`: add a `metrics_version` column and include only
  current-version sessions (finding 6).
- Gaze and synchrony chance: report missing below a minimum usable span instead
  of shifting by seconds; smooth real and chance categories the same way
  (finding 11).
- Low-confidence infill in `temporal_smooth.py` must not cross gaps (finding 12,
  confirm first).

## Phase 5c: Anchor coverage (bug hunt finding 7)

- Guarantee at least 2 anchor frames per block or part (storybook blocks got 0-1).
- Blind-check parent/child labels in the affected overlay clips.

## Phase 6: Docs and re-run

- Update `docs/OUTPUTS.md`, report wording, and `COLLEAGUE_QUICKSTART.md`.
- Add a metrics-only re-run script (no re-inference) and regenerate 1011v3-1020v3.
- Then draft the reply to the colleague (user chose: reply after fixes land).

## Open Questions (for the user / lab)

- Starting thresholds for `usable` / `check` / `insufficient` (e.g. minimum valid
  seconds per task). Should be fixed before analysis, not tuned on results.
- Camera policy when both exist: best camera per task, or a fixed primary camera.
- Whether 1019 has a camera B recording (unknown; ask the colleague).
- Missing frames are likely not random (child leaves view when far away), so
  frame-weighting cannot remove that bias; how to report it.

## Test Strategy

pytest, TDD, synthetic fixtures only. Integration check by re-running metrics on
the local v3 sessions and diffing old vs new `session_metrics.csv`.
