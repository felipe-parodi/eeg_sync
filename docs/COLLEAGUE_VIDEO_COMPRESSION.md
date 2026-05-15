# Colleague Video Compression

Use `scripts/compress_videos_for_portal.py` before uploading large GoPro files
to the Video Processing Portal.

The script is standalone Python and uses only the standard library. It requires
`ffmpeg` on the collaborator laptop. If `ffmpeg.exe` is placed next to the
script, the script will use that copy automatically.

## One Video

```powershell
python scripts\compress_videos_for_portal.py --video "C:\Videos\session_A.mov"
```

## Two Videos

```powershell
python scripts\compress_videos_for_portal.py --video "C:\Videos\session_A.mov" --video "C:\Videos\session_B.mov"
```

## Folder Of Videos

```powershell
python scripts\compress_videos_for_portal.py --input-dir "C:\Videos"
```

Outputs go to `portal_compressed/` by default. The defaults are portal-compatible:

- 8 FPS
- maximum width 854 px
- no audio
- aggressive H.264 compression
- hardware encoders tried first, CPU fallback if needed

The script writes `portal_compressed/compression_summary.json` with input paths,
output paths, encoder used, and file sizes.
