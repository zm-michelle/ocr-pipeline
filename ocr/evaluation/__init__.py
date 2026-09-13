"""Evaluation: CER/WER/detection metrics and the benchmark report."""

from ocr.evaluation.benchmark import benchmark
from ocr.evaluation.metrics import (
    box_iou,
    compute_cer,
    compute_wer,
    detection_f1,
    exact_match_ratio,
)

__all__ = [
    "benchmark",
    "box_iou",
    "compute_cer",
    "compute_wer",
    "detection_f1",
    "exact_match_ratio",
]
