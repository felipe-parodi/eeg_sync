# Templates

Copy any file here, rename it, and fill in your own values.

| File | Used by | Notes |
|---|---|---|
| `session_config.example.json` | `video-pose-metrics`, `video-gaze-metrics`, `video-gaze-snapshots`, `gaze-infer`, `gaze-synchrony`, `gaze-plot` | Copy to repo root as `session_config.json` (or pass `--session-config <path>`). Defines analysis blocks and parent/child track IDs per camera. |
| `segment_timings.example.csv` | Input from the colleague — no CLI consumes this directly. | The format we ask collaborators to send when they note start/end times of analyzable segments. Convert to `session_blocks` in `session_config.json`. |
| `exclusion_windows.example.csv` | Input from the colleague — no CLI consumes this directly. | Times to drop because someone other than parent + child was present. Apply manually via `video-annotate-tracks` (relabel the third person to track 2 or 3) followed by `video-correct-tracks`, then re-run `video-filter-tracks --n-keep 2`. |

## Quick conversion: segment_timings.csv → session_config.json

A `start_time` of `13:26` in `MM:SS` = `806` seconds. The pipeline expects seconds in `session_blocks`. Either compute by hand or use:

```python
def mmss_to_s(t):
    parts = t.split(":")
    return int(parts[0]) * 60 + float(parts[1])
```
