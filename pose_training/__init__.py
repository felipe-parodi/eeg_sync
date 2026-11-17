"""Pseudo-labeling and training tools for a domain-adapted YOLO11 pose model.

This package turns the project's compressed session videos into a YOLO pose
training set (sampling -> teacher pseudo-labeling -> QC/filtering -> dataset
export), fine-tunes a student model, and evaluates it against the current
portal model. See docs/plans/pseudo-label-pose-train_260531.md for the full plan.

The keypoint layout stays COCO-17 throughout so every downstream stage (gaze
head bbox, overlays, metrics) works unchanged.
"""

from __future__ import annotations

__all__ = ["coco"]
