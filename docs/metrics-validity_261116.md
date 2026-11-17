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

## Phase 2: Synchrony chance baseline

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

## Phase 3: False-person check

- Flag a track row as suspect when its box is static over a long stretch (box
  center and size nearly constant over >= 10 s), or matches a recurring static
  region in the detection cache.
- Write a per-frame `suspect` flag and the block-level `suspect_person_share`.
- Validate on 1019v3 (expect high share in grocery) and 1016v3/1018v3 (expect
  ~0). Render flagged frames to PNG for **user visual confirmation**; do not
  claim correctness from numbers alone.

## Phase 4: Valid time and verdict

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
