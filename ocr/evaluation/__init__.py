"""Evaluation: CER/WER/detection metrics and the benchmark report."""

from ocr.evaluation.benchmark import benchmark
from ocr.evaluation.e2e import evaluate_end_to_end
from ocr.evaluation.metrics import (
    box_iou,
    compute_cer,
    compute_wer,
    corpus_cer,
    corpus_wer,
    detection_counts,
    detection_f1,
    exact_match_ratio,
    prf_from_counts,
)

__all__ = [
    "benchmark",
    "evaluate_end_to_end",
    "box_iou",
    "compute_cer",
    "compute_wer",
    "detection_f1",
    "detection_counts",
    "prf_from_counts",
    "corpus_cer",
    "corpus_wer",
    "exact_match_ratio",
]
