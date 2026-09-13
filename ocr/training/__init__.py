"""Training: optimization loops, checkpoint I/O, and live metrics tracking."""

from ocr.training.checkpoints import load_checkpoint, save_checkpoint
from ocr.training.loops import (
    train_detector,
    train_recognizer,
    validate_detector,
    validate_recognizer,
)
from ocr.training.tracking import RunTracker, default_run_name

__all__ = [
    "RunTracker",
    "default_run_name",
    "load_checkpoint",
    "save_checkpoint",
    "train_detector",
    "train_recognizer",
    "validate_detector",
    "validate_recognizer",
]
