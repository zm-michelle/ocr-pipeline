"""Inference: detector postprocessing, recognizer decoding, and the full pipeline."""

from ocr.inference.detection import (
    Box,
    detect_text_regions,
    merge_overlapping_boxes,
    sort_boxes_reading_order,
    split_regions_into_lines,
)
from ocr.inference.pipeline import (
    LineResult,
    OCRPipeline,
    PageResult,
    overlay_image,
    run_ocr_folder,
    run_ocr_image,
    save_overlay,
)
from ocr.inference.recognition import recognize_crop, recognize_crops

__all__ = [
    "Box",
    "detect_text_regions",
    "merge_overlapping_boxes",
    "sort_boxes_reading_order",
    "split_regions_into_lines",
    "recognize_crop",
    "recognize_crops",
    "OCRPipeline",
    "PageResult",
    "LineResult",
    "overlay_image",
    "save_overlay",
    "run_ocr_image",
    "run_ocr_folder",
]
