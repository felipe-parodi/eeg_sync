# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See [AGENTS.md](AGENTS.md) for full project context, architecture, and conventions.

## Claude Code-Specific Notes

- Default to `pytest -q` for concise output.
- Use `ruff check --fix . && black .` for auto-formatting.
- CPU is the default device -- never assume GPU availability.
- Never commit `video_inference/data/` or raw video recordings.
- The primary user is non-technical -- use plain language in all CLI output and error messages.

## IMPORTANT: Video Compression Requirement

**NEVER run video inference (`video-infer`) on raw/uncompressed video files.** Raw GoPro recordings can be 2-10 GB each and will cause extremely slow frame extraction and excessive disk usage.

Always compress videos first using `video-compress-rapid` or the pipeline's built-in compression (do NOT pass `--skip-compress` on raw files). Target file size should be **under 50 MB**. The compressed videos in `video_inference/compressed/` are already prepared and safe to use with `--skip-compress`.

```bash
# Compress first (if not already done)
video-compress-rapid --input-dir video_inference/data/ --output-dir video_inference/compressed/

# Then run inference on the compressed video
video-infer run --video-a video_inference/compressed/video_compressed.mp4 ...
```
