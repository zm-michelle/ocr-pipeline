"""Detector training targets in the style of DBNet (Liao et al., 2020).

Instead of asking the network to paint every pixel of a text box, we

  1. SHRINK each box inward by an offset D, so adjacent lines are separated by
     a wide, unambiguous background band in the target (the "probability map"
     target, `gt`), and
  2. build a THRESHOLD MAP target that is high on the box border and fades to
     a floor over the same distance D on both sides, so the network learns
     where the boundary is rather than just what is inside it.

At inference the detected (shrunk) boxes are expanded back out - "unclipped" -
by an offset derived from their own size. Everything here is for axis-aligned
rectangles, which is all the synthetic generator and the postprocessor produce.

The paper derives offsets from area/perimeter, which suits arbitrary polygons
but does not invert cleanly for thin rectangles (a text line shrunk by 8 px
came back out by 4). For axis-aligned boxes we use the short side instead:

    shrink:  D  = s * min(w, h)                 s = shrink_ratio (default 0.3)
    unclip:  D' = s / (1 - 2s) * min(w', h')    on the detected, shrunk box

which is an exact inverse: a box shrunk by D and unclipped by D' is the
original box, and s = 0 leaves boxes untouched. With s = 0.3 a line keeps a
core 40% of its height and the threshold band is D wide on either side.
"""

from __future__ import annotations

import numpy as np

Box = tuple[int, int, int, int]

DEFAULT_SHRINK_RATIO = 0.3
DEFAULT_UNCLIP_RATIO = DEFAULT_SHRINK_RATIO  # undoes the default shrink exactly
THRESH_MIN, THRESH_MAX = 0.3, 0.7  # threshold-map value range, as in the paper


def shrink_offset(box: Box, shrink_ratio: float = DEFAULT_SHRINK_RATIO) -> float:
    """How far to move each edge inward: a fraction of the short side."""
    x1, y1, x2, y2 = box
    return float(shrink_ratio * max(0, min(x2 - x1, y2 - y1)))


def unclip_offset(box: Box, unclip_ratio: float = DEFAULT_UNCLIP_RATIO) -> float:
    """How far to move each edge of a detected (shrunk) box back outward.

    `unclip_ratio` is expressed as the shrink ratio it undoes, so a detector
    trained with shrink_ratio=s is unclipped with unclip_ratio=s.
    """
    x1, y1, x2, y2 = box
    s = unclip_ratio
    if s <= 0 or s >= 0.5:
        return 0.0
    return float(s / (1.0 - 2.0 * s) * max(0, min(x2 - x1, y2 - y1)))


def shrink_box(box: Box, shrink_ratio: float = DEFAULT_SHRINK_RATIO) -> Box:
    d = shrink_offset(box, shrink_ratio)
    x1, y1, x2, y2 = box
    return (int(round(x1 + d)), int(round(y1 + d)), int(round(x2 - d)), int(round(y2 - d)))


def unclip_box(box: Box, unclip_ratio: float, width: int, height: int) -> Box:
    if unclip_ratio <= 0:
        return box
    d = unclip_offset(box, unclip_ratio)
    x1, y1, x2, y2 = box
    return (
        max(0, int(round(x1 - d))),
        max(0, int(round(y1 - d))),
        min(width, int(round(x2 + d))),
        min(height, int(round(y2 + d))),
    )


def build_db_targets(
    boxes: list[Box],
    size: tuple[int, int],
    shrink_ratio: float = DEFAULT_SHRINK_RATIO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (gt, thresh_map, thresh_mask), each float32 [H, W], for one page.

    gt          1 inside each SHRUNK box, else 0.
    thresh_map  in [THRESH_MIN, THRESH_MAX]; THRESH_MAX exactly on a box border,
                falling linearly to THRESH_MIN at distance D either side.
    thresh_mask 1 wherever thresh_map carries a real target (the border bands),
                so the L1 loss is only applied there.

    With shrink_ratio == 0 every offset is 0 and gt is the plain full-box mask.
    """
    width, height = size
    gt = np.zeros((height, width), dtype=np.float32)
    thresh_map = np.full((height, width), THRESH_MIN, dtype=np.float32)
    thresh_mask = np.zeros((height, width), dtype=np.float32)

    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box)
        if x2 <= x1 or y2 <= y1:
            continue
        d = shrink_offset((x1, y1, x2, y2), shrink_ratio)

        sx1, sy1, sx2, sy2 = shrink_box((x1, y1, x2, y2), shrink_ratio)
        if sx2 > sx1 and sy2 > sy1:
            gt[max(0, sy1) : min(height, sy2), max(0, sx1) : min(width, sx2)] = 1.0

        if d <= 0:
            continue
        # Signed distance to the ORIGINAL box border inside a window that
        # covers the band on both sides: positive inside, negative outside.
        wx1, wy1 = max(0, int(np.floor(x1 - d))), max(0, int(np.floor(y1 - d)))
        wx2, wy2 = min(width, int(np.ceil(x2 + d)) + 1), min(height, int(np.ceil(y2 + d)) + 1)
        if wx2 <= wx1 or wy2 <= wy1:
            continue
        ys = np.arange(wy1, wy2, dtype=np.float32)[:, None] + 0.5
        xs = np.arange(wx1, wx2, dtype=np.float32)[None, :] + 0.5
        inside = np.minimum(np.minimum(xs - x1, x2 - xs), np.minimum(ys - y1, y2 - ys))
        dx = np.maximum(np.maximum(x1 - xs, 0.0), xs - x2)
        dy = np.maximum(np.maximum(y1 - ys, 0.0), ys - y2)
        outside = np.sqrt(dx * dx + dy * dy)
        signed = np.where(inside >= 0, inside, -outside)

        band = np.abs(signed) <= d
        value = THRESH_MIN + (THRESH_MAX - THRESH_MIN) * (1.0 - np.abs(signed) / d)
        window = thresh_map[wy1:wy2, wx1:wx2]
        np.maximum(window, np.where(band, value, THRESH_MIN), out=window)
        thresh_mask[wy1:wy2, wx1:wx2] = np.maximum(thresh_mask[wy1:wy2, wx1:wx2], band.astype(np.float32))

    return gt, thresh_map, thresh_mask
