# Portal Launch And Public Access

The fastest public-facing setup is to expose the local FastAPI portal through a
temporary public tunnel. This keeps all videos and GPU processing on the
workstation, but gives collaborators a normal HTTPS URL.

## What Happens After Submit

The browser does not start inference immediately after the user clicks
**Submit Processing Job**. The order is:

1. Create a job record.
2. Upload Video A and Video B in parallel 50 MiB chunks.
3. Write each chunk directly into the final camera video file on the workstation.
4. Verify all chunks arrived.
5. Compress videos with NVIDIA NVENC if either one exceeds the portal size
   threshold.
6. Extract frames only from post-exclusion analysis windows.
7. Run YOLO11m-pose inference, filtering, smoothing, metrics, overlays,
   optional gaze, result packaging, and notification.

If the tab spinner is active before the job page appears, the browser is usually
still uploading chunks or waiting for the server to queue finalization. Compression
starts only after both videos have uploaded and the job has moved to the server
side.

## Speed Guidance For Collaborators

For full GoPro recordings, the fastest v1 workflow is to compress videos before
upload. Server-side compression is still useful, but it cannot reduce the time
needed to send a raw 3.5 GB file across the network. A non-technical workflow
should use a one-click local compressor or a short drag-and-drop compression
utility, not CLI instructions.

Upload speed is driven by the slowest part of the path:

- collaborator laptop disk read speed,
- collaborator Wi-Fi or Ethernet upload speed,
- VPN or campus network routing,
- public tunnel throughput,
- workstation network receive speed,
- browser and TLS overhead,
- concurrent network traffic.

Increasing chunk size beyond 50 MiB usually does not change the true bandwidth
limit. Larger chunks mostly reduce request overhead, but they increase retry
cost and are more likely to hit proxy request-size limits.

## Saved Run Metadata

Each submitted run writes metadata into the job directory before processing
starts:

```text
video_inference/output/portal_jobs/<job_id>/submission_metadata.json
video_inference/output/portal_jobs/<job_id>/upload_meta.json
```

The result zip also includes the metadata under:

```text
<session_id>/portal_metadata/submission_metadata.json
<session_id>/portal_metadata/job_config.json
<session_id>/portal_metadata/rapid_compression_summary.json
<session_id>/portal_metadata/job.log
```

`submission_metadata.json` records the session ID, email, original video
filenames and sizes, raw analysis-block text, parsed analysis blocks,
extra-person windows, post-exclusion analysis blocks, upload chunk settings, and
processing settings such as pose backend, tracker, frame rate, compression
threshold, required encoder, whether analysis-window-only inference was used,
and whether gaze analysis was requested.

## Analysis-Window-Only Inference

The portal removes extra-person windows from the submitted analysis blocks before
running inference. It then passes those post-exclusion blocks to `video-infer
run --analysis-windows`, so the workstation only extracts and processes frames
inside valid analysis time.

The extracted frame files are still numbered sequentially for compatibility, but
`frame_index.csv`, `tracks_2d.csv`, and `pose_3d.csv` retain original
source-video `timestamp_s` values. Downstream filtering, metrics, gaze inputs,
and overlays use those timestamps.

## Human Launch Steps

Run the portal on the workstation:

```powershell
cd "C:\Users\Felipe Parodi\Documents\ext_repos\eeg_sync"
.\start-portal.ps1
```

The script prompts for the portal password, starts a public tunnel in the
background, prints the URL to share with collaborators, then launches the
portal in the foreground. Tunnel selection:

- **Prefers `cloudflared`** (Cloudflare Quick Tunnel, `*.trycloudflare.com`).
  The URL is stable for the lifetime of the cloudflared process, so it does
  not rotate mid-session. Install from
  https://github.com/cloudflare/cloudflared/releases.
- **Falls back to `ssh` + `localhost.run`** (`*.lhr.life`) if cloudflared is
  not installed. Anonymous `lhr.life` tunnels rotate roughly hourly; a
  collaborator connected past that point will need a refreshed URL.
- **Falls back to local-only** if neither tool is available.

On the first run the script generates `.portal_secret_key` (gitignored)
for session cookie signing and reuses it on every subsequent run so
existing browser sessions survive restarts.
**Every job from a prior run is wiped on startup** (uploads, completed
downloads, and failed-job logs alike). Restart only when no in-flight
job results need to be preserved.

The tunnel URL is fresh every run; do not bookmark it. Send the URL and
portal password to the collaborator through a private channel.

For same-machine local HTTP testing only, run with `-NoTunnel` and set
`$env:PORTAL_COOKIE_SECURE = "false"` before launching:

```powershell
$env:PORTAL_COOKIE_SECURE = "false"
.\start-portal.ps1 -NoTunnel
```

Do not commit `.portal_secret_key` or share the portal password in GitHub
issues. The default cookie mode requires HTTPS, which is why the tunnel
(TLS-terminated by Cloudflare or localhost.run) is the supported
collaborator path.

The portal uses one upload path: authenticated browser chunk uploads under
`/uploads/*`. The older direct multipart `/submit` path is intentionally not
registered. Authenticated POST requests require a per-session CSRF token that
the portal injects into the upload page JavaScript.

Only one processing job runs at a time in a portal process. If two researchers
submit jobs together, the later job stays queued until the active GPU job
finishes.

To run without the script (e.g., from CI or a managed service), set
`PORTAL_PASSWORD` and `PORTAL_SECRET_KEY` in the environment and invoke
`video-processing-portal` directly.

## LLM Launch Steps

When an LLM or automation launches the portal, prefer a background server and a
separate background tunnel:

```powershell
$env:PORTAL_PASSWORD = "<choose-a-shared-password>"
$env:PORTAL_SECRET_KEY = [Convert]::ToBase64String((1..32 | ForEach-Object { Get-Random -Maximum 256 }))
Start-Process -FilePath '.\.venv\Scripts\python.exe' -ArgumentList @('-m','portal.app') -WorkingDirectory (Resolve-Path -LiteralPath '.').Path -WindowStyle Hidden
```

Then start the tunnel:

```powershell
$log = "C:\tmp\localhostrun_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$script = "& 'C:\Windows\System32\OpenSSH\ssh.exe' -o StrictHostKeyChecking=no -R 80:127.0.0.1:8080 nokey@localhost.run *> '$log'"
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($script))
Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-EncodedCommand',$encoded) -WindowStyle Hidden
Get-Content $log
```

The printed `https://...lhr.life` URL is the collaborator-facing URL.

## URL Naming

Anonymous tunnel URLs are intentionally random and temporary. That is why the
testing URL looks like `https://da07776d795174.lhr.life`.

For a short stable URL, use one of these paths:

- **Fastest stable option:** create a localhost.run account/custom-domain setup
  and map a short name to this workstation tunnel.
- **More production-like option:** create a named Cloudflare Tunnel and map a
  public hostname such as `https://video-processing.<your-domain>` to
  `http://127.0.0.1:8080`.

Both options keep the portal application and GPU processing on the workstation.

## Cloudflare Tunnel Alternative

Cloudflare Tunnel is also a reasonable option:

```powershell
cloudflared tunnel --url http://127.0.0.1:8080
```

Cloudflare prints a temporary `https://...trycloudflare.com` URL. Send that URL
and the password to the collaborator. In the first local test, `localhost.run`
forwarded the portal correctly, while the Cloudflare quick tunnel registered but
returned a 404 from the generated hostname. A named Cloudflare Tunnel may still
be useful later for a stable production URL.

## Why This Path

- No inbound firewall rule is required.
- Penn VPN is not required for the collaborator.
- The GPU workstation still does all processing locally.
- The portal uploads videos in 50 MiB chunks, which avoids the common 100 MB
  request-body limit on free public tunnel/proxy setups.
- The job page refreshes automatically during server-side processing.

## Later Stable URL

Temporary `trycloudflare.com` and anonymous `lhr.life` URLs change when the
tunnel restarts. For a stable production URL, create a named Cloudflare Tunnel
and map it to a hostname such as:

```text
https://video-processing.<your-domain>
```

That requires a Cloudflare account and a domain, but the backend application can
stay the same.

## GitHub Pages Option

GitHub Pages can host a static frontend later. The browser would still call this
same FastAPI backend through the Cloudflare Tunnel. That adds CORS and separate
frontend deployment, so it is not the fastest path for v1.
