# Quickstart: from session videos to results

**Audience**: research assistants processing parent-child sessions.
**You will**: compress the videos, submit them in the Video Processing Portal,
label the parent in a few frames, and download the results.

The portal does all the processing. You need a web browser, the portal link
and password, and the two GoPro videos of the session.

If you want to run the steps by hand instead (for example with an LLM), see
[By hand](#by-hand) at the end.

---

## 1. Before recording

The numbers are only as good as what the cameras see.

- **Keep the lens clear.** Nothing should sit close in front of a camera. In
  one session an object a few centimetres from camera A covered half the view
  and the tracker took it for a person.
- **Frame both people in full** for as much of each task as possible. When the
  child leaves the view, that time cannot be measured.
- **Use both cameras.** When one view is blocked, the other often has the
  data.

## 2. Write down the times

For each video, note in that video's own clock:

- **Tasks**: name, start, end (for example `grocery 13:26 23:40`).
- **Extra-person windows**: start, end and a reason, for any time someone
  other than the parent and child is in view (for example the experimenter).
  These times are removed from the analysis.

The two cameras' clocks differ, so camera B's times are not camera A's.

## 3. Compress the videos

Raw GoPro files are several GB. Compress each one first with the portal
compressor (it makes 10-frames-per-second copies of a few tens of MB):

```powershell
python scripts\compress_videos_for_portal.py --video "C:\Videos\1021_A.mov" --video "C:\Videos\1021_B.mov"
```

Details: [COLLEAGUE_VIDEO_COMPRESSION.md](COLLEAGUE_VIDEO_COMPRESSION.md).

## 4. Submit the session

Log in and click **New session**. The form has six steps:

1. **Session ID**, e.g. `1021`. If that ID already has results, the portal
   asks before replacing them; earlier results are moved to an archive
   folder, never deleted.
2. **Videos**: drop the compressed Video A and Video B. One camera is fine if
   only one recorded.
3. **Tasks** for each video, one row per task.
4. **Extra-person windows** for each video. The preview underneath shows
   exactly what will be analysed. A task cut by a window becomes pieces
   (`grocery_part1`, `grocery_part2`, ...); pieces shorter than 10 s are
   skipped and listed.
5. **Gaze**: tick to also measure where people look (takes longer).
6. **Review and start**.

The form checks the times as you type (for example `1:75` or a time past the
end of the video) and tells you what to fix.

## 5. Label the parent

After a few minutes the job page says it is waiting for you. Click the link
and, for each frame shown, pick the box around the **parent** (or skip the
frame if neither box is the parent). Every task gets at least two frames.
This takes about two minutes and decides who is parent and who is child for
the whole recording, so take care.

Processing then continues by itself. The job page shows which stage it is
at; you can close the browser and come back.

## 6. Read the results

When the job is done, click **Open results** on the job page (or open the
session from the **Sessions** page). The same report is `report.html` in the
results zip. It starts with one row per camera and task, with the verdict
and the reasons for it; the pieces of a task are under **Per piece**, and
the overlay clips play in the page.

**Check the clips first.** Each block has short overlay clips (start and end)
with skeletons drawn. Make sure the boxes follow the parent and the child,
not someone else or an object.

**The verdict** next to each task says whether there is enough data:

| Verdict | Meaning |
|---|---|
| Usable | Both people tracked for at least 2 minutes and at least 10 unbroken 20-second stretches for movement synchrony. |
| Check | Both people tracked less than half the time, or fewer than 10 unbroken 20-second stretches (movement synchrony is then not shown). Watch the clips. |
| Not enough data | Both people tracked for less than 2 minutes in total. |

**Recommended inclusion rule** (use it for every session, decided before
looking at results):

1. For each task, use the camera with more usable time. Do not average the
   two cameras; distances in pixels differ between them.
2. Include distance and gaze for a task when both people are tracked for at
   least 2 minutes (`valid_seconds_both` >= 120).
3. Include movement synchrony only with at least 10 windows
   (`sync_n_windows` >= 10).
4. Exclude any task whose clips show the boxes on the wrong person or an
   object, whatever its numbers.
5. Report usable time (or coverage) as a covariate; missing time is not
   random, because the child is lost mostly when moving around.

Why these numbers: on sessions 1011-1020, 2 minutes of data rank dyads'
distance almost like the whole task (r = 0.85), and 10 synchrony windows do
the same for synchrony (r ~ 0.87); with 3 windows it is only r = 0.66.

**The tables** (in the session folder and the zip):

| File | One row per | Use it for |
|---|---|---|
| `task_metrics.csv` | camera and task, pieces joined back together | analyses at the task level |
| `session_metrics.csv` | camera and block (piece) | checking individual pieces |
| `all_sessions_task_metrics.csv` (home page, **Download study task table**) | every session, camera and task | the study table |

Key columns:

- `coverage_both`, `valid_seconds_both`: share and seconds of the time with
  both people tracked.
- `proximity_scaled_median` (and `_mean`): torso-normalized distance, the
  distance between the two torsos divided by the pair's typical torso length
  on that camera. Use this one. It is not a distance in meters.
  `proximity_mean`, `proximity_median` are the same in image pixels, which
  depend on zoom and are not comparable across cameras.
- Why the two cameras can still disagree: each camera is a flat picture. It
  sees distance across the image, never toward or away from it. Two people
  standing one behind the other, as seen by a camera, look close together even
  when they are far apart.
- Choosing a camera, decided before looking at results: per dyad, prefer a
  camera that is `usable` for both tasks; among those, take the one with more
  both-tracked seconds (`valid_seconds_both`) summed over the tasks; on an
  exact tie, camera A. If no camera is usable for both tasks, flag the dyad.
  Use that camera for every measure, include camera as a fixed effect, and
  repeat key analyses with one fixed camera as a check.
- `sync_mean` and `sync_chance_mean`: movement synchrony (how closely the
  amount of whole-body movement of the two rises and falls together, in
  20-second windows) and the same measure with the child's movement taken
  from other moments (averaged over several time shifts; empty when the task
  is too short for at least 3). Only `sync_minus_chance` above zero reflects
  the interaction. With split tasks, also look at how the windows are spread
  across the pieces in `session_metrics.csv`.
- `gaze_<category>_pct` and `gaze_<category>_chance_pct`: share of frames in
  each gaze category and its chance level. A category means something only
  when it clearly exceeds chance.
- `quality_verdict`, `quality_reasons`: the verdict above.
- `metrics_version`: rows with different versions were computed differently;
  do not pool them. Re-submit older sessions instead.

Pieces of a task are joined by pooling, not averaging: distances from all
frames, synchrony from all 20-second windows, gaze shares weighted by frames.
A window never spans the gap between pieces.

Every column is described in [OUTPUTS.md](OUTPUTS.md); the zip also has a
`README.md`.

## 7. When something looks off

| You see | Likely cause | What to do |
|---|---|---|
| Low coverage in one camera only | That camera's view was blocked or the child was out of view | Use the other camera for that task |
| Boxes on the wrong person in a clip | A third person, or parent/child mixed up | Add an extra-person window, or re-submit and label more carefully |
| A box on an object | Something close to the lens | Exclude that block; fix camera placement next time |
| The job failed | Shown on the job page with the stage that failed | Open "technical details" and send them to Felipe |

---

## By hand

The portal runs a fixed chain of command-line steps (pose inference with the
offline two-person tracker, parent labels, keypoint refinement, gap filling,
smoothing, metrics, report). To run them yourself, follow
[PIPELINE.md](PIPELINE.md); output columns are in [OUTPUTS.md](OUTPUTS.md) and
input formats in [INPUTS.md](INPUTS.md). Install with `uv`
(`uv pip install -e ".[dev,video,gaze]"`); ffmpeg must be on `PATH`. Always
compress raw videos first.

EEG synchronization is a separate workflow: see the
[README](../README.md#optional-eeg-video-synchronization).
