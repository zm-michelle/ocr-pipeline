"""Ground truth for real scans that ship without transcripts, via their clean twins.

NoisyOffice pairs every degraded page with a pixel-aligned clean original.
Tesseract reads the clean page almost perfectly; those words and boxes then
label the degraded versions. Boxes come from the 2x-resolution clean scans
(better OCR) and are scaled to the 1x frame the noisy images use.

Output mirrors the synthetic manifests, so the same training and evaluation
code runs unchanged on real pages.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from ocr.data.text_source import sanitize

VERSION_RE = re.compile(r"_(Clean|Noise[a-z]+)_", re.IGNORECASE)


def tesseract_lines(image_path: Path, min_conf: float = 60.0, psm: int = 6) -> list[dict[str, Any]]:
    """Text lines with word-union boxes from Tesseract's TSV output."""
    if shutil.which("tesseract") is None:
        raise RuntimeError("tesseract is not installed (brew install tesseract / apt install tesseract-ocr)")
    out = subprocess.run(
        ["tesseract", str(image_path), "stdout", "--psm", str(psm), "-l", "eng", "tsv"],
        capture_output=True, text=True, check=True,
    ).stdout
    words: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE):
        if row.get("level") != "5" or not row.get("text", "").strip():
            continue
        key = (int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
        words[key].append(
            {
                "text": row["text"],
                "conf": float(row["conf"]),
                "box": (int(row["left"]), int(row["top"]), int(row["left"]) + int(row["width"]), int(row["top"]) + int(row["height"])),
            }
        )
    lines: list[dict[str, Any]] = []
    for key in sorted(words):
        ws = words[key]
        mean_conf = sum(w["conf"] for w in ws) / len(ws)
        text = sanitize(" ".join(w["text"] for w in ws)).strip()
        if not text or mean_conf < min_conf:
            continue
        x1 = min(w["box"][0] for w in ws); y1 = min(w["box"][1] for w in ws)
        x2 = max(w["box"][2] for w in ws); y2 = max(w["box"][3] for w in ws)
        lines.append({"text": text, "box": [x1, y1, x2, y2], "conf": round(mean_conf, 1)})
    return lines


def label_pairs(
    clean_dir: Path,
    noisy_dir: Path,
    output_dir: Path,
    split_by_suffix: dict[str, str] | None = None,
    min_conf: float = 60.0,
    pad: int = 2,
) -> dict[str, Path]:
    """Label every noisy page from its clean twin and write pages/lines manifests.

    clean_dir may hold higher-resolution scans than noisy_dir; boxes are scaled.
    split_by_suffix maps the dataset's own suffix (TR/VA/TE) to a split name.
    """
    split_by_suffix = split_by_suffix or {"TR": "train", "VA": "train", "TE": "test"}
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "line_crops"
    crops_dir.mkdir(exist_ok=True)

    def rel(path: Path) -> str:
        """Relative to the manifest dir (e.g. ../SimulatedNoisyOffice/x.png), so data/ is movable as a tree."""
        return os.path.relpath(path.resolve(), output_dir.resolve())

    noisy_by_key: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(noisy_dir.glob("*.png")):
        noisy_by_key[VERSION_RE.sub("_", p.stem)].append(p)

    pages: dict[str, list[dict]] = defaultdict(list)
    lines: dict[str, list[dict]] = defaultdict(list)
    stats = {"clean_pages": 0, "noisy_pages": 0, "lines": 0, "dropped_low_conf": 0}

    for clean in sorted(clean_dir.glob("*.png")):
        key = VERSION_RE.sub("_", clean.stem)
        twins = noisy_by_key.get(key, [])
        if not twins:
            continue
        stats["clean_pages"] += 1
        with Image.open(clean) as im:
            clean_w, clean_h = im.size
        raw = tesseract_lines(clean, min_conf=-1.0)
        kept = [l for l in raw if l["conf"] >= min_conf]
        stats["dropped_low_conf"] += len(raw) - len(kept)
        suffix = clean.stem.rsplit("_", 1)[-1]
        split = split_by_suffix.get(suffix, "train")

        for noisy in twins:
            with Image.open(noisy) as im:
                nw, nh = im.size
                sx, sy = nw / clean_w, nh / clean_h
                page_lines = []
                for l in kept:
                    x1, y1, x2, y2 = l["box"]
                    box = [max(0, int(x1 * sx) - pad), max(0, int(y1 * sy) - pad), min(nw, int(x2 * sx) + pad), min(nh, int(y2 * sy) + pad)]
                    if box[2] - box[0] < 8 or box[3] - box[1] < 4:
                        continue
                    page_lines.append({"text": l["text"], "box": box, "conf": l["conf"], "source": "tesseract:clean-twin"})
                    crop_path = crops_dir / f"{noisy.stem}_line_{len(page_lines) - 1:02d}.png"
                    im.crop(tuple(box)).save(crop_path)
                    lines[split].append({"image": rel(crop_path), "text": l["text"], "source_page": rel(noisy), "box": box})
            pages[split].append({"image": rel(noisy), "boxes": [l["box"] for l in page_lines], "lines": page_lines, "clean_twin": rel(clean)})
            stats["noisy_pages"] += 1
            stats["lines"] += len(page_lines)

    written: dict[str, Path] = {}
    for split in sorted(set(pages) | set(lines)):
        pp = output_dir / f"pages_manifest_{split}.json"; pp.write_text(json.dumps({"pages": pages[split]}, indent=1)); written[f"pages_{split}"] = pp
        lp = output_dir / f"lines_manifest_{split}.json"; lp.write_text(json.dumps({"samples": lines[split]}, indent=1)); written[f"lines_{split}"] = lp
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print("pseudo-label stats:", stats)
    for name, path in written.items():
        n = len(pages[name.split("_", 1)[1]]) if name.startswith("pages") else len(lines[name.split("_", 1)[1]])
        print(f"  {name}: {n} -> {path}")
    return written
