"""Turn a detector probability map (or a raw page) into sorted line boxes."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from ocr.config import DEFAULT_DETECTOR_SIZE
from ocr.data.targets import unclip_box
from ocr.data.transforms import normalize_tensor, pil_to_tensor, resize_page

Box = tuple[int, int, int, int]


def sort_boxes_reading_order(boxes: list[Box]) -> list[Box]:
    if not boxes:
        return []
    heights = [y2 - y1 for _, y1, _, y2 in boxes]
    row_tol = max(8, int(np.median(heights) * 0.6))
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    rows: list[list[Box]] = []
    for box in boxes:
        cy = (box[1] + box[3]) // 2
        for row in rows:
            row_cy = sum((b[1] + b[3]) // 2 for b in row) / len(row)
            if abs(cy - row_cy) <= row_tol:
                row.append(box)
                break
        else:
            rows.append([box])
    ordered: list[Box] = []
    for row in sorted(rows, key=lambda r: min(b[1] for b in r)):
        ordered.extend(sorted(row, key=lambda b: b[0]))
    return ordered


def merge_overlapping_boxes(boxes: list[Box], y_overlap: float = 0.45) -> list[Box]:
    if not boxes:
        return []
    boxes = sort_boxes_reading_order(boxes)
    merged: list[Box] = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue
        x1, y1, x2, y2 = box
        mx1, my1, mx2, my2 = merged[-1]
        inter_y = max(0, min(y2, my2) - max(y1, my1))
        min_h = max(1, min(y2 - y1, my2 - my1))
        gap = x1 - mx2
        if inter_y / min_h >= y_overlap and gap < max(18, min_h * 2):
            merged[-1] = (min(mx1, x1), min(my1, y1), max(mx2, x2), max(my2, y2))
        else:
            merged.append(box)
    return merged


def _projection_runs(values: np.ndarray, threshold: float, max_gap: int = 2) -> list[tuple[int, int]]:
    active = values > threshold
    runs: list[tuple[int, int]] = []
    start: int | None = None
    last = -1
    for idx, is_active in enumerate(active):
        if is_active:
            if start is None:
                start = idx
            last = idx
        elif start is not None:
            runs.append((start, last + 1))
            start = None
    if start is not None:
        runs.append((start, last + 1))

    if not runs:
        return []
    merged = [runs[0]]
    for y1, y2 in runs[1:]:
        prev_y1, prev_y2 = merged[-1]
        if y1 - prev_y2 <= max_gap:
            merged[-1] = (prev_y1, y2)
        else:
            merged.append((y1, y2))
    return merged


def _line_boxes_from_crop(
    image: Image.Image,
    box: Box,
    padding: int = 3,
    min_height: int = 6,
    min_width: int = 24,
) -> list[Box]:
    x1, y1, x2, y2 = box
    crop = np.asarray(image.crop(box).convert("L"))
    if crop.size == 0:
        return []

    blurred = cv2.GaussianBlur(crop, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    row_counts = binary.sum(axis=1) / 255.0
    if len(row_counts) >= 5:
        row_counts = cv2.GaussianBlur(row_counts.reshape(-1, 1), (1, 5), 0).ravel()

    threshold = max(2.0, crop.shape[1] * 0.018)
    runs = _projection_runs(row_counts, threshold=threshold, max_gap=2)

    line_boxes: list[Box] = []
    for local_y1, local_y2 in runs:
        if local_y2 - local_y1 < min_height:
            continue
        band = binary[local_y1:local_y2, :]
        col_counts = band.sum(axis=0) / 255.0
        col_runs = _projection_runs(col_counts, threshold=max(1.0, band.shape[0] * 0.08), max_gap=8)
        if col_runs:
            local_x1 = max(0, col_runs[0][0] - padding)
            local_x2 = min(crop.shape[1], col_runs[-1][1] + padding)
        else:
            local_x1, local_x2 = 0, crop.shape[1]
        if local_x2 - local_x1 < min_width:
            continue
        line_boxes.append(
            (
                max(0, x1 + local_x1),
                max(0, y1 + local_y1 - padding),
                min(image.width, x1 + local_x2),
                min(image.height, y1 + local_y2 + padding),
            )
        )
    return line_boxes


def split_regions_into_lines(
    image: Image.Image,
    boxes: list[Box],
    min_split_height: int = 42,
    min_region_width: int = 40,
    min_region_height: int = 8,
) -> list[Box]:
    """Split tall regions into single text lines using horizontal ink projection."""
    line_boxes: list[Box] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        height = y2 - y1
        if height >= min_split_height:
            split = _line_boxes_from_crop(image, box)
            if len(split) >= 2:
                line_boxes.extend(split)
                continue
        line_boxes.append(box)
    filtered = [
        box
        for box in line_boxes
        if box[2] - box[0] >= min_region_width and box[3] - box[1] >= min_region_height
    ]
    return sort_boxes_reading_order(filtered)


def _ridge(profile: np.ndarray, start: int, direction: int, max_steps: int) -> int:
    """Walk from `start` in `direction` (+1/-1) and return the index where `profile` peaks.

    The threshold map is trained to be highest exactly on a box border and to
    fall off on both sides, so its ridge marks the true edge of the line.
    """
    n = len(profile)
    best_idx, best_val = start, -1.0
    for step in range(max_steps + 1):
        idx = start + direction * step
        if idx < 0 or idx >= n:
            break
        value = float(profile[idx])
        if value > best_val:
            best_idx, best_val = idx, value
        elif value < best_val - 0.02 and best_idx != start:
            break  # past the first ridge; stop before we climb the neighbouring line's
    return best_idx


def refine_boxes_with_threshold_map(boxes: list[Box], thresh_map: np.ndarray, max_grow_factor: float = 2.5) -> list[Box]:
    """Snap each detected core box to the ridges of the learned threshold map.

    Used instead of the geometric unclip when the detector has a threshold
    head: the geometric rule multiplies any blur in the core's thickness by
    ~2.5x, which for 17 px lines turns a 5 px blur into a 13 px overshoot.
    The threshold map peaks on the real border, so we read the border off it.
    """
    height, width = thresh_map.shape[:2]
    refined: list[Box] = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        core_h = y2 - y1
        max_steps = int(max_grow_factor * core_h) + 4
        rows = thresh_map[:, x1:x2].mean(axis=1)
        top = _ridge(rows, y1, -1, max_steps)
        bottom = _ridge(rows, max(y1, y2 - 1), +1, max_steps)
        cols = thresh_map[top : bottom + 1, :].mean(axis=0)
        grow_x = int(0.5 * ((y1 - top) + (bottom + 1 - y2))) + 2  # borders sit about as far out sideways
        left = _ridge(cols, x1, -1, grow_x)
        right = _ridge(cols, max(x1, x2 - 1), +1, grow_x)
        refined.append((left, top, right + 1, bottom + 1))
    return refined


def boxes_from_prob_map(
    prob: np.ndarray,
    image: Image.Image,
    threshold: float = 0.35,
    unclip_ratio: float = 0.0,
    min_area: int = 80,
    padding: int = 4,
    split_lines: bool = True,
    thresh_map: np.ndarray | None = None,
) -> list[Box]:
    """Probability map (same size as `image`) -> reading-order line boxes.

    This is the one postprocessing path, shared by inference and by detector
    validation so the box-level metric scores exactly what OCR will get.
    `unclip_ratio` re-expands boxes from a detector trained on shrunk targets.
    `split_lines=False` skips the ink-projection rescue of tall regions, which
    validation uses to score the detector's own separation of lines.
    """
    height, width = prob.shape[:2]
    mask = (prob >= threshold).astype(np.uint8) * 255
    kernel_w = max(9, width // 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cores: list[Box] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area or h < 5 or w < 8:
            continue
        cores.append((x, y, x + w, y + h))

    if thresh_map is not None and unclip_ratio > 0:
        # Learned border: snap to the threshold-map ridge, then a light pad.
        grown = refine_boxes_with_threshold_map(cores, thresh_map)
        pad = max(1, padding // 2)
    else:
        grown = [unclip_box(b, unclip_ratio, width, height) if unclip_ratio > 0 else b for b in cores]
        pad = padding
    boxes: list[Box] = [
        (max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)) for x1, y1, x2, y2 in grown
    ]
    merged = merge_overlapping_boxes(boxes)
    if not split_lines:
        return sort_boxes_reading_order(merged)
    return split_regions_into_lines(image, merged)


def detect_text_regions(
    detector: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    image_size: tuple[int, int] = DEFAULT_DETECTOR_SIZE,
    threshold: float = 0.35,
    min_area: int = 80,
    padding: int = 4,
    unclip_ratio: float = 0.0,
    learned_threshold: bool = False,
    db_k: float = 50.0,
) -> tuple[list[Box], np.ndarray]:
    """Run the detector and return reading-order line boxes plus the map that was binarized.

    learned_threshold=False: cut the probability map P at the fixed `threshold`.
    learned_threshold=True:  use the detector's own threshold map T, i.e. keep
        pixels where P > T - the binarization the DB loss trained for. The
        returned map is then B = sigmoid(k (P - T)), cut at 0.5, and `threshold`
        is ignored. Only meaningful for a detector trained with the DB loss.
    """
    original = image.convert("L")
    original_w, original_h = original.size
    # Match the training width and let the height follow (rounded to the stride),
    # instead of squashing every page into 540x258: a 540x420 scan keeps its
    # ~20 px lines, which is the text size the detector was trained on. DBNet is
    # fully convolutional, so any height works at inference.
    target_w = image_size[0]
    target_h = max(32, int(round(original_h * target_w / original_w / 32)) * 32)
    model_input = resize_page(original, (target_w, target_h))
    tensor = normalize_tensor(pil_to_tensor(model_input)).unsqueeze(0).to(device)

    detector.eval()
    with torch.no_grad():
        if learned_threshold and hasattr(detector, "forward_maps"):
            prob_logits, thresh_logits = detector.forward_maps(tensor)
            prob = torch.sigmoid(prob_logits.float())
            thresh = torch.sigmoid(thresh_logits.float())
            binary = torch.sigmoid(db_k * (prob - thresh))
            score_map = binary[0, 0].cpu().numpy()
            thresh_np = cv2.resize(thresh[0, 0].cpu().numpy(), (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            cutoff = 0.5
        else:
            score_map = torch.sigmoid(detector(tensor))[0, 0].detach().cpu().numpy()
            thresh_np = None
            cutoff = threshold

    score_map = cv2.resize(score_map, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    boxes = boxes_from_prob_map(score_map, original, cutoff, unclip_ratio, min_area, padding, thresh_map=thresh_np)
    return boxes, score_map
