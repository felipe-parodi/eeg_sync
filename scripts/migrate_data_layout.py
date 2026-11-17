"""One-time move of local data into the data/ + models/ + logs/ layout.

Before 2026-11-17, data lived inside the code folders
(``video_inference/output``, ``pose_training/data`` ...) and model weights and
logs sat in the repo root. After pulling the ``src/`` layout, stop the portal
and run::

    uv run python scripts/migrate_data_layout.py --dry-run   # show the plan
    uv run python scripts/migrate_data_layout.py             # do it

Rules: nothing is overwritten (folders are merged item by item; a clash is
reported and left in place), a root ``*.pt`` is removed only when an identical
copy is already in ``models/``, and running it twice is harmless. Moves are
renames on the same drive, so they are instant even for large folders.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path

# (old, new) relative to the repo root, folders merged into folders.
FOLDER_MOVES: list[tuple[str, str]] = [
    ("video_inference/output", "data/output"),
    ("video_inference/data/portal_uploads", "data/uploads"),
    ("video_inference/data", "data/videos"),
    ("video_inference/compressed", "data/compressed"),
    ("pose_training/data", "data/training"),
    ("pose_training/runs", "data/runs"),
    ("compressed_video", "data/videos"),
    ("runs", "data/runs"),
]
LOG_FILES = ["portal_server.log", ".portal_tunnel.log", ".portal_tunnel.err.log"]
# Old code folders that only hold caches once their data is moved out.
LEFTOVER_DIRS = ["video_inference", "pose_training"]
CACHE_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _merge(src: Path, dst: Path, report: dict, dry_run: bool) -> None:
    """Move ``src`` into ``dst``; merge folders; never overwrite."""
    if not dst.exists():
        report["moved"].append(f"{src} -> {dst}")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
        return
    if src.is_dir() and dst.is_dir():
        for child in sorted(src.iterdir()):
            _merge(child, dst / child.name, report, dry_run)
        if not dry_run and not any(src.iterdir()):
            src.rmdir()
        return
    report["conflicts"].append(f"{src} (already at {dst}; left in place)")


def _only_caches(folder: Path) -> bool:
    for path in folder.rglob("*"):
        if path.is_file() and not any(part in CACHE_NAMES for part in path.parts):
            return False
    return True


def migrate(root: Path, dry_run: bool = False) -> dict[str, list[str]]:
    """Move old-layout data under ``root`` into the new layout.

    Args:
        root: Repository root.
        dry_run: Only report what would happen.

    Returns:
        ``{"moved": [...], "removed": [...], "conflicts": [...]}``.
    """
    root = Path(root)
    report: dict[str, list[str]] = {"moved": [], "removed": [], "conflicts": []}
    for old, new in FOLDER_MOVES:
        src = root / old
        if src.exists():
            _merge(src, root / new, report, dry_run)
    for weights in sorted(root.glob("*.pt")):
        dst = root / "models" / weights.name
        if dst.exists():
            if filecmp.cmp(weights, dst, shallow=False):
                report["removed"].append(f"{weights} (identical copy in models/)")
                if not dry_run:
                    weights.unlink()
            else:
                report["conflicts"].append(f"{weights} (differs from {dst}; left in place)")
        else:
            _merge(weights, dst, report, dry_run)
    for name in LOG_FILES:
        src = root / name
        if src.exists():
            _merge(src, root / "logs" / name, report, dry_run)
    for name in LEFTOVER_DIRS:
        folder = root / name
        if folder.is_dir() and _only_caches(folder):
            report["removed"].append(f"{folder} (only caches left)")
            if not dry_run:
                shutil.rmtree(folder)
    return report


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="Show the plan only.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: this script's repo).",
    )
    args = parser.parse_args(argv)
    report = migrate(args.root, dry_run=args.dry_run)
    verb = "Would move" if args.dry_run else "Moved"
    for line in report["moved"]:
        print(f"{verb}: {line}")
    for line in report["removed"]:
        print(f"{'Would remove' if args.dry_run else 'Removed'}: {line}")
    for line in report["conflicts"]:
        print(f"Left in place: {line}")
    if not any(report.values()):
        print("Nothing to move: the folder already uses the new layout.")


if __name__ == "__main__":
    main()
