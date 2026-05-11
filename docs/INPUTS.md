# Inputs

Reference for what the pipeline expects on disk before any CLI runs. Read this if you're setting up a new session or troubleshooting "command says file not found."

## Folder layout

```
eeg_sync/
├── session_config.json              # one per session (or pass --session-config)
├── video_inference/
│   ├── data/                        # raw videos go here (gitignored)
│   │   ├── camera_a_raw.mov
│   │   └── camera_b_raw.mov
│   ├── compressed/                  # video-compress-rapid writes here
│   │   ├── camera_a_raw_rapid.mp4
│   │   └── camera_b_raw_rapid.mp4
│   └── output/<session_id>/         # pipeline writes here, one subdir per camera
│       ├── camera_a/
│       └── camera_b/
└── templates/                       # copy-and-edit starting points
```

`video_inference/data/`, `video_inference/compressed/`, and `video_inference/output/` are gitignored. Anything in them is local to your machine.

---

## Videos

| Property | Expected |
|---|---|
| Format | Any container ffmpeg + OpenCV can read (`.mov`, `.mp4`, `.avi`, etc.). GoPro MOV is the tested case. |
| Compression | **Must be compressed before inference.** Target under 50 MB. Use `video-compress-rapid`. |
| Resolution after compression | 854×480 by default (`video-compress-rapid --max-width 854`). Match `image_width` / `image_height` in `session_config.json` if you change it. |
| Cameras per session | 1 or 2. `--video-a` is required; `--video-b` is optional. |
| People in analyzed segments | **Exactly 2** (parent + child). Anything else needs `video-annotate-tracks` to relabel or `--max-persons > 2` followed by `video-filter-tracks` to prune. |

Raw videos belong in `video_inference/data/`. Never commit them — that folder is gitignored.

---

## session_config.json

Single source of truth for per-session analysis windows and per-camera person mappings. Copy from [`templates/session_config.example.json`](../templates/session_config.example.json).

Fields:

| Field | Type | Required by | Notes |
|---|---|---|---|
| `session_id` | string | metrics CLIs (only when not set elsewhere) | Used for plot titles and output paths. |
| `output_dir` | string | (reference only — pipeline takes `--output-dir`) | Where camera output dirs live. |
| `image_width`, `image_height` | int | `gaze-infer`, `video-gaze-metrics`, `video-pose-metrics` | Pixel dimensions of the **compressed** video (not the raw GoPro). |
| `frame_rate` | float | metrics CLIs | The fps of the CSVs you're feeding them (matches `--frame-rate` from `video-infer run` or downstream resampling). |
| `camera_mappings[].camera_id` | string | gaze + metrics CLIs | `camera_a` or `camera_b`. |
| `camera_mappings[].parent_track_id` | int | gaze + metrics CLIs **unless** you ran `video-filter-tracks` (which produces parent=0/child=1 automatically). | The raw track_id that corresponds to the parent in `tracks_2d.csv`. |
| `camera_mappings[].child_track_id` | int | same | The raw track_id for the child. |
| `session_blocks[].name` | string | gaze + metrics CLIs | Short identifier (no spaces preferred). |
| `session_blocks[].start_s`, `end_s` | number | gaze + metrics CLIs | Seconds from the **start of the video** (not the start of the task). |
| `session_blocks[].color` | string | plotting only | Hex or matplotlib color name. Used for shading in dashboards. |

### Parent/child track IDs — three workflows

You have to tell the gaze and metrics CLIs which detected track is the parent and which is the child. Pick one:

1. **(Recommended) Run `video-filter-tracks` after inference.** It writes `tracks_2d_filtered.csv` / `pose_3d_filtered.csv` with `track_id=0` = parent and `track_id=1` = child (assigned by mean bbox area per block). Then in `session_config.json`, leave `parent_track_id=0` and `child_track_id=1` for every camera, and point the metrics CLIs at the `_filtered.csv` outputs.
2. **Hand-look at the overlay video.** Run `video-visualize`, watch which track ID consistently belongs to which person, write those IDs into `camera_mappings`.
3. **Annotate by hand** with `video-annotate-tracks` to enforce track_id=0 for the parent and track_id=1 for the child across the whole session.

Workflow 1 is the default for the colleague path. Workflows 2 and 3 exist for when the auto-assignment gets it wrong.

---

## Segment timings (collaborator input)

When a collaborator records new sessions, ask them for the start/end times of the analyzable segments per session, in `MM:SS`. Template: [`templates/segment_timings.example.csv`](../templates/segment_timings.example.csv):

```csv
session_id,block_name,start_time,end_time,notes
P001c,grocery_free_play,13:26,23:40,
P001c,storybook_reading,29:22,37:06,
```

This is a **human-to-human format**, not consumed by any CLI directly. Convert it to seconds and paste into `session_config.json` → `session_blocks`.

---

## Exclusion windows (collaborator input)

Times when someone other than parent + child was in frame. Template: [`templates/exclusion_windows.example.csv`](../templates/exclusion_windows.example.csv):

```csv
session_id,start_time,end_time,reason
P001c,8:14,8:42,experimenter entered to swap toys
```

Three options for handling:

1. **Trim the block.** Adjust `session_blocks[i].start_s` / `end_s` to skip the window.
2. **Split the block.** Add two `session_blocks` entries (`block_name_part1`, `block_name_part2`) bracketing the exclusion window.
3. **Relabel the extra person.** Run `video-annotate-tracks` over the exclusion window; press `2` or `3` (anything other than `0`/`1`) on the experimenter; save; run `video-correct-tracks`; then `video-filter-tracks --n-keep 2` to drop them.

Option 1 is simplest and recommended unless the exclusion window splits a critical event.

---

## EEG inputs (optional, sync workflow only)

Only needed if you're using `eeg-sync`. Skip otherwise.

- OpenBCI raw `.txt` or pre-cleaned `*fixed_irBlaster.csv` (CSV is preferred — script auto-detects).
- IR blaster connected to **Analog Channel 0**, baseline value 257.
- Sample rate 250 Hz (the default for OpenBCI Cyton).

Details in the top-level [README](../README.md#optional-eeg-video-synchronization).
