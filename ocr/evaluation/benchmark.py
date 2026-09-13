"""Score a predictions.json against optional text / detection label manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ocr.evaluation.metrics import compute_cer, compute_wer, detection_f1, exact_match_ratio


def _load_items(path: str | Path, *keys: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        for key in keys:
            if data.get(key):
                return data[key]
        return []
    return data


def benchmark(
    predictions_path: str | Path,
    labels_path: str | Path | None = None,
    detection_labels_path: str | Path | None = None,
) -> dict[str, Any]:
    predictions = _load_items(predictions_path, "predictions")
    report: dict[str, Any] = {"num_predictions": len(predictions)}
    print(f"benchmark loaded {len(predictions)} prediction(s)")

    if labels_path is None:
        report["gt_text_metrics"] = "not computed; no labels manifest was provided"
        print("benchmark text metrics: no ground-truth labels provided")
    else:
        label_items = _load_items(labels_path, "samples", "items", "labels")
        labels = {Path(item["image"]).name: item.get("text", "") for item in label_items}
        pairs = [(p["text"], labels[p["image"]]) for p in predictions if p["image"] in labels]
        if pairs:
            report["cer"] = float(np.mean([compute_cer(p, t) for p, t in pairs]))
            report["wer"] = float(np.mean([compute_wer(p, t) for p, t in pairs]))
            report["exact_match"] = exact_match_ratio([p for p, _ in pairs], [t for _, t in pairs])
            print(
                "benchmark text metrics: "
                f"cer {report['cer']:.3f} wer {report['wer']:.3f} exact {report['exact_match']:.3f}"
            )
        else:
            report["gt_text_metrics"] = "not computed; no prediction filenames matched labels"
            print("benchmark text metrics: no prediction filenames matched labels")

    if detection_labels_path is not None:
        label_items = _load_items(detection_labels_path, "pages", "items")
        labels = {Path(item["image"]).name: item.get("boxes", []) for item in label_items}
        scores = []
        for pred in predictions:
            if pred["image"] not in labels:
                continue
            pred_boxes = [tuple(r["box"]) for r in pred.get("regions", [])]
            gt_boxes = [tuple(b) for b in labels[pred["image"]]]
            scores.append(detection_f1(pred_boxes, gt_boxes))
        if scores:
            report["det_precision"] = float(np.mean([s["precision"] for s in scores]))
            report["det_recall"] = float(np.mean([s["recall"] for s in scores]))
            report["det_f1"] = float(np.mean([s["f1"] for s in scores]))
            print(
                "benchmark detection metrics: "
                f"precision {report['det_precision']:.3f} recall {report['det_recall']:.3f} f1 {report['det_f1']:.3f}"
            )
    return report
