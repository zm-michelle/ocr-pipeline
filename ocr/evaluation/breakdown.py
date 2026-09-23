"""Where do the errors come from? Slices and confusions for an end-to-end run.

An aggregate CER says how bad; this says what kind of bad: which fonts, which
line lengths, which characters get swapped, which pages are the outliers.
Lines are matched between prediction and ground truth by box IoU, so detector
misses and spurious boxes are counted separately from recognition mistakes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ocr.evaluation.metrics import _edit_distance, box_iou, compute_cer

Box = tuple[int, int, int, int]


def align_chars(pred: str, target: str) -> list[tuple[str, str]]:
    """Levenshtein backtrace -> list of (target_char, pred_char); '' marks insert/delete."""
    n, m = len(target), len(pred)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (target[i - 1] != pred[j - 1]))
    pairs: list[tuple[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (target[i - 1] != pred[j - 1]):
            pairs.append((target[i - 1], pred[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            pairs.append((target[i - 1], ""))  # deletion: target char missing from prediction
            i -= 1
        else:
            pairs.append(("", pred[j - 1]))  # insertion: prediction has an extra char
            j -= 1
    pairs.reverse()
    return pairs


def match_lines(pred_lines: list[dict], gt_lines: list[dict], iou_threshold: float = 0.3) -> tuple[list[tuple[dict, dict]], int, int]:
    """Greedy IoU matching. Returns (pairs, unmatched_gt, unmatched_pred)."""
    used: set[int] = set()
    pairs: list[tuple[dict, dict]] = []
    for pred in pred_lines:
        best, best_iou = -1, 0.0
        for idx, gt in enumerate(gt_lines):
            if idx in used:
                continue
            iou = box_iou(tuple(pred["box"]), tuple(gt["box"]))
            if iou > best_iou:
                best, best_iou = idx, iou
        if best >= 0 and best_iou >= iou_threshold:
            used.add(best)
            pairs.append((pred, gt_lines[best]))
    return pairs, len(gt_lines) - len(used), len(pred_lines) - len(pairs)


def _length_bucket(n: int) -> str:
    if n < 20:
        return "<20 chars"
    if n < 40:
        return "20-39"
    if n < 60:
        return "40-59"
    return "60+"


class ErrorBreakdown:
    def __init__(self) -> None:
        self.by_font: dict[str, list[int]] = defaultdict(lambda: [0, 0])      # edits, chars
        self.by_family: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.by_style: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.by_length: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.confusions: Counter = Counter()
        self.deleted: Counter = Counter()
        self.inserted: Counter = Counter()
        self.lines_matched = self.lines_missed = self.lines_spurious = 0
        self.page_scores: list[tuple[float, str]] = []

    def add_page(self, image: str, pred_lines: list[dict], gt_lines: list[dict], page_cer: float) -> None:
        pairs, missed, spurious = match_lines(pred_lines, gt_lines)
        self.lines_matched += len(pairs)
        self.lines_missed += missed
        self.lines_spurious += spurious
        self.page_scores.append((page_cer, image))
        for pred, gt in pairs:
            target, output = gt.get("text", ""), pred.get("text", "")
            edits = _edit_distance(output, target)
            font = Path(gt["font_file"]).stem if gt.get("font_file") else "unknown"
            style = ("bold " if gt.get("bold_like") else "") + ("italic" if gt.get("italic_like") else "") or "regular"
            for table, key in (
                (self.by_font, font),
                (self.by_family, gt.get("font_family", "unknown")),
                (self.by_style, style.strip()),
                (self.by_length, _length_bucket(len(target))),
            ):
                table[key][0] += edits
                table[key][1] += len(target)
            for t, p in align_chars(output, target):
                if t and p and t != p:
                    self.confusions[f"{t}->{p}"] += 1
                elif t and not p:
                    self.deleted[t] += 1
                elif p and not t:
                    self.inserted[p] += 1

    @staticmethod
    def _rates(table: dict[str, list[int]], min_chars: int = 200) -> list[dict[str, Any]]:
        rows = [
            {"key": k, "cer": e / c, "chars": c}
            for k, (e, c) in table.items()
            if c >= min_chars
        ]
        return sorted(rows, key=lambda r: -r["cer"])

    def report(self, worst: int = 10) -> dict[str, Any]:
        return {
            "lines": {"matched": self.lines_matched, "missed_by_detector": self.lines_missed, "spurious_boxes": self.lines_spurious},
            "cer_by_family": self._rates(self.by_family),
            "cer_by_style": self._rates(self.by_style),
            "cer_by_length": self._rates(self.by_length),
            "cer_by_font": self._rates(self.by_font, min_chars=500)[:15],
            "top_confusions": [{"pair": k, "count": v} for k, v in self.confusions.most_common(20)],
            "top_deleted": [{"char": k, "count": v} for k, v in self.deleted.most_common(10)],
            "top_inserted": [{"char": k, "count": v} for k, v in self.inserted.most_common(10)],
            "worst_pages": [{"cer": round(c, 4), "image": im} for c, im in sorted(self.page_scores, reverse=True)[:worst]],
        }

    def summary(self) -> str:
        r = self.report()
        out = [f"lines: {r['lines']['matched']} matched, {r['lines']['missed_by_detector']} missed, {r['lines']['spurious_boxes']} spurious"]
        for name in ("cer_by_family", "cer_by_style", "cer_by_length"):
            out.append(f"{name[7:]:8s} " + "  ".join(f"{row['key']} {row['cer']*100:.1f}%" for row in r[name]))
        if r["cer_by_font"]:
            out.append("worst fonts " + "  ".join(f"{row['key']} {row['cer']*100:.1f}%" for row in r["cer_by_font"][:5]))
        out.append("confusions " + "  ".join(f"{c['pair']}x{c['count']}" for c in r["top_confusions"][:10]))
        out.append("deleted " + " ".join(f"{c['char']!r}x{c['count']}" for c in r["top_deleted"][:6]) + "   inserted " + " ".join(f"{c['char']!r}x{c['count']}" for c in r["top_inserted"][:6]))
        return "\n".join(out)
