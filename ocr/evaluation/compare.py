"""Score this pipeline against other systems on the same pages.

A CER only means something next to a reference. This runs any number of
"readers" - trained checkpoints of this pipeline, or an external engine such as
Tesseract - over one manifest and reports them side by side, so a change can be
judged against a baseline rather than against its own previous number.

Every reader gets the identical page list and the identical scoring: corpus CER
and WER (total edits / total units) over line texts joined in manifest order.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ocr.evaluation.e2e import _load_pages, page_ground_truth
from ocr.evaluation.metrics import compute_cer, corpus_cer, corpus_wer
from ocr.utils import ProgressLogger

Reader = Callable[[Path], str]


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def tesseract_reader(psm: int = 6, lang: str = "eng") -> Reader:
    """Plain Tesseract page text, one line per output line."""

    def read(path: Path) -> str:
        out = subprocess.run(
            ["tesseract", str(path), "stdout", "--psm", str(psm), "-l", lang],
            capture_output=True, text=True, check=True,
        ).stdout
        return "\n".join(line.strip() for line in out.splitlines() if line.strip())

    return read


def tesseract_version() -> str:
    try:
        first = subprocess.run(["tesseract", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
        return first.strip()
    except Exception:
        return "tesseract"


def pipeline_reader(pipeline: Any) -> Reader:
    """This project's own pipeline as a reader."""
    return lambda path: pipeline.read_path(path).text


def score_reader(
    name: str,
    reader: Reader,
    pages: list[dict[str, Any]],
    root: Path,
    log_every: int = 50,
) -> dict[str, Any]:
    predictions: list[str] = []
    targets: list[str] = []
    per_page: list[float] = []
    failures = 0
    progress = ProgressLogger(f"compare {name}", len(pages), log_every)
    started = time.perf_counter()

    for step, page in enumerate(pages, start=1):
        image = Path(page.get("image") or page.get("image_path"))
        if not image.is_absolute():
            image = root / image
        target = page_ground_truth(page)
        try:
            text = reader(image)
        except Exception as exc:  # one unreadable page must not lose the whole comparison
            failures += 1
            print(f"  {name}: failed on {image.name}: {type(exc).__name__}: {exc}")
            text = ""
        predictions.append(text)
        targets.append(target)
        per_page.append(compute_cer(text, target))
        if progress.should_log(step):
            progress.log(step, f"cer {corpus_cer(predictions, targets):.4f}")

    elapsed = time.perf_counter() - started
    return {
        "system": name,
        "cer": corpus_cer(predictions, targets),
        "wer": corpus_wer(predictions, targets),
        "page_mean_cer": sum(per_page) / len(per_page) if per_page else 0.0,
        "page_exact": sum(p == t for p, t in zip(predictions, targets)) / len(pages) if pages else 0.0,
        "seconds_per_page": elapsed / max(1, len(pages)),
        "failed_pages": failures,
        "num_pages": len(pages),
    }


def compare_systems(
    manifest_path: str | Path,
    readers: Iterable[tuple[str, Reader]],
    root_dir: str | Path | None = None,
    limit: int | None = None,
    log_every: int = 50,
) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_path)
    root = Path(root_dir) if root_dir else manifest_path.parent
    pages = _load_pages(manifest_path)
    if limit:
        pages = pages[:limit]
    return [score_reader(name, reader, pages, root, log_every) for name, reader in readers]


def format_table(rows: list[dict[str, Any]], sort: bool = True) -> str:
    """Fixed-width table, best CER first, with each row's gap to the best."""
    if not rows:
        return "(nothing to compare)"
    rows = sorted(rows, key=lambda r: r["cer"]) if sort else rows
    best = rows[0]["cer"]
    width = max(len(r["system"]) for r in rows)
    head = f"{'system'.ljust(width)}   {'CER':>8}  {'WER':>8}  {'exact':>7}  {'s/page':>7}   vs best"
    lines = [head, "-" * len(head)]
    for r in rows:
        delta = "" if r["cer"] <= best else f"+{(r['cer'] - best) * 100:.2f} pts"
        if r["cer"] <= best:
            delta = "best"
        flag = f"  ({r['failed_pages']} failed)" if r["failed_pages"] else ""
        lines.append(
            f"{r['system'].ljust(width)}   {r['cer'] * 100:7.2f}%  {r['wer'] * 100:7.2f}%  "
            f"{r['page_exact'] * 100:6.1f}%  {r['seconds_per_page']:6.2f}s   {delta}{flag}"
        )
    lines.append("")
    lines.append(f"{rows[0]['num_pages']} pages, corpus CER/WER (total edits / total units)")
    return "\n".join(lines)
