"""End-to-end evaluation: page in, text out, scored against the manifest.

This is the number that describes the product. Component metrics can improve
while OCR gets worse - a detector that merges two lines scores well on pixels
and hands the recognizer garbage - and only running the whole pipeline on
held-out pages catches that, along with reading-order mistakes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ocr.evaluation.breakdown import ErrorBreakdown
from ocr.evaluation.metrics import (
    compute_cer,
    corpus_cer,
    corpus_wer,
    detection_counts,
    prf_from_counts,
)
from ocr.utils import ProgressLogger

if TYPE_CHECKING:  # the pipeline imports training, which imports evaluation - keep this hint-only
    from ocr.inference.pipeline import OCRPipeline


def _load_pages(manifest_path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(manifest_path).read_text())
    return data.get("pages", data) if isinstance(data, dict) else data


def page_ground_truth(page: dict[str, Any]) -> str:
    """Lines joined in manifest order, matching how the pipeline joins its output."""
    lines = page.get("lines") or []
    return "\n".join(str(line.get("text", "")) for line in lines)


def evaluate_end_to_end(
    pipeline: OCRPipeline,
    manifest_path: str | Path,
    root_dir: str | Path | None = None,
    limit: int | None = None,
    log_every: int = 100,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    root = Path(root_dir) if root_dir else manifest_path.parent
    pages = _load_pages(manifest_path)
    if limit:
        pages = pages[:limit]

    predictions: list[str] = []
    targets: list[str] = []
    per_page_cer: list[float] = []
    box_tp = box_fp = box_fn = 0
    fallbacks = 0
    breakdown = ErrorBreakdown()
    progress = ProgressLogger("e2e eval", len(pages), log_every)

    for step, page in enumerate(pages, start=1):
        image_path = Path(page.get("image") or page.get("image_path"))
        if not image_path.is_absolute():
            image_path = root / image_path
        result = pipeline.read_path(image_path)
        target = page_ground_truth(page)

        predictions.append(result.text)
        targets.append(target)
        per_page_cer.append(compute_cer(result.text, target))
        fallbacks += int(result.used_line_split_fallback)

        gt_boxes = [tuple(b) for b in (page.get("boxes") or [l["box"] for l in page.get("lines", [])])]
        tp, fp, fn = detection_counts([line.box for line in result.lines], gt_boxes)
        box_tp, box_fp, box_fn = box_tp + tp, box_fp + fp, box_fn + fn
        breakdown.add_page(
            str(page.get("image")),
            [{"box": list(line.box), "text": line.text} for line in result.lines],
            page.get("lines") or [{"box": list(b), "text": ""} for b in gt_boxes],
            per_page_cer[-1],
        )

        if progress.should_log(step):
            progress.log(step, f"cer {corpus_cer(predictions, targets):.3f} page-mean cer {sum(per_page_cer) / step:.3f}")

    box = prf_from_counts(box_tp, box_fp, box_fn)
    return {
        "num_pages": len(pages),
        "cer": corpus_cer(predictions, targets),
        "wer": corpus_wer(predictions, targets),
        "page_mean_cer": sum(per_page_cer) / len(per_page_cer) if per_page_cer else 0.0,
        "page_exact": sum(p == t for p, t in zip(predictions, targets)) / len(pages) if pages else 0.0,
        "det_precision": box["precision"],
        "det_recall": box["recall"],
        "det_f1": box["f1"],
        "line_split_fallbacks": fallbacks,
        "unclip_ratio": pipeline.unclip_ratio,
        "breakdown": breakdown.report(),
        "breakdown_summary": breakdown.summary(),
    }
