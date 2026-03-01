# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See [AGENTS.md](AGENTS.md) for full project context, architecture, and conventions.

## Claude Code-Specific Notes

- Default to `pytest -q` for concise output.
- Use `ruff check --fix . && black .` for auto-formatting.
- CPU is the default device -- never assume GPU availability.
- Never commit `video_inference/data/` or raw video recordings.
- The primary user is non-technical -- use plain language in all CLI output and error messages.
