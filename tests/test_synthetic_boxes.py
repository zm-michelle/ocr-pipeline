"""Every line box a generated page reports must be croppable.

A page-level transform (skew, two-column offset) once produced an inverted box
like [3, 260, 99, 258]: PIL accepted it all the way to `crop`, which raised
`cannot write empty image` and killed a 40k-page generation run 11k pages in.
Seed 1337 page 4277 is that exact case.

Run: .venv/bin/python -m pytest tests/ -q     (or just execute this file)
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run directly, not just under pytest

import numpy as np

from ocr.data.synthetic import (
    MIN_LINE_W,
    MIN_LINE_H,
    DegradationConfig,
    FontResolver,
    _draw_document,
    _draw_two_columns,
    _generate_page,
    _skew_page,
)

PAGE = (540, 258)


def _valid(box: list[int], width: int, height: int) -> bool:
    x1, y1, x2, y2 = box
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and x2 - x1 >= MIN_LINE_W and y2 - y1 >= MIN_LINE_H


def test_transforms_keep_boxes_inside_the_page() -> None:
    resolver = FontResolver()
    for seed in range(300):
        random.seed(seed)
        two_col = random.random() < 0.5
        draw = _draw_two_columns if two_col else _draw_document
        image, lines = draw(PAGE, 7, 10, resolver, "noisyoffice")
        image, lines = _skew_page(image, lines, 1.5)
        for line in lines:
            assert _valid(line["box"], *image.size), f"seed {seed} two_col={two_col}: {line['box']} vs {image.size}"


def test_regression_seed_1337_page_4277() -> None:
    """The page that crashed the run: a line rotated below the bottom edge."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for sub in ("pages", "line_crops", "previews"):
            (out / sub).mkdir()
        page, crops = _generate_page(4277, str(out), PAGE, 7, 10, True, 0.20, 0.35, DegradationConfig(), False, 1337, "noisyoffice")
        assert page["lines"], "page produced no lines"
        assert all(_valid(b, *PAGE) for b in page["boxes"]), page["boxes"]
        assert len(crops) == len(list((out / "line_crops").glob("*.png")))


def test_pages_are_reproducible_regardless_of_process() -> None:
    """Per-page seeding is what makes parallel generation deterministic."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for sub in ("pages", "line_crops", "previews"):
            (out / sub).mkdir()
        args = (str(out), PAGE, 7, 10, False, 0.20, 0.35, DegradationConfig(), False, 99, "noisyoffice")
        first, _ = _generate_page(11, *args)
        first_pixels = (out / "pages" / "page_000011.png").read_bytes()
        random.seed(0)  # unrelated draws between the two calls must not matter
        np.random.seed(0)
        second, _ = _generate_page(11, *args)
        assert first["boxes"] == second["boxes"]
        assert first_pixels == (out / "pages" / "page_000011.png").read_bytes()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
