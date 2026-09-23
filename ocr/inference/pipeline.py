"""End-to-end OCR: detector -> line boxes -> recognizer -> text.

`OCRPipeline` holds the loaded models so a long-lived process (the web app) can
read many images without touching disk again. The `run_ocr_*` helpers wrap it for
the one-shot CLI modes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from ocr.config import (
    DEFAULT_CHARSET,
    DEFAULT_DETECTOR_SIZE,
    DEFAULT_RECOGNIZER_HEIGHT,
    DEFAULT_RECOGNIZER_WIDTH,
    IMAGE_EXTENSIONS,
    LEGACY_RECOGNIZER_SIZE,
)
from ocr.inference.detection import Box, detect_text_regions, split_regions_into_lines
from ocr.inference.recognition import recognize_crops
from ocr.models import CRNNRecognizer, DBNet
from ocr.training.checkpoints import load_checkpoint
from ocr.utils import ProgressLogger, resolve_device

# A page smaller than this is assumed to already be a single cropped line.
MIN_SPLITTABLE_SIZE = (200, 120)  # width, height


@dataclass
class LineResult:
    box: Box
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"box": list(self.box), "text": self.text}


@dataclass
class PageResult:
    text: str
    lines: list[LineResult] = field(default_factory=list)
    used_line_split_fallback: bool = False
    prob_map: np.ndarray | None = None
    image: str | None = None
    path: str | None = None

    def to_dict(self, include_regions: bool = True) -> dict[str, Any]:
        return {
            "image": self.image,
            "path": self.path,
            "text": self.text,
            "regions": [line.to_dict() for line in self.lines] if include_regions else [],
            "used_line_split_fallback": self.used_line_split_fallback,
        }


def save_overlay(image: Image.Image, boxes: list[Box], path: str | Path) -> None:
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for box in boxes:
        draw.rectangle(box, outline=(220, 40, 40), width=2)
    overlay.save(path)


def overlay_image(image: Image.Image, boxes: list[Box]) -> Image.Image:
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for box in boxes:
        draw.rectangle(box, outline=(220, 40, 40), width=2)
    return overlay


class OCRPipeline:
    """Detector + recognizer held in memory, ready to read images."""

    def __init__(
        self,
        recognizer: torch.nn.Module,
        detector: torch.nn.Module | None = None,
        device: torch.device | str = "cpu",
        charset: str = DEFAULT_CHARSET,
        detector_size: tuple[int, int] = DEFAULT_DETECTOR_SIZE,
        threshold: float = 0.35,
        rec_height: int = DEFAULT_RECOGNIZER_HEIGHT,
        rec_width: int = DEFAULT_RECOGNIZER_WIDTH,
        split_lines_without_detector: bool = False,
        unclip_ratio: float = 0.0,
        learned_threshold: bool = False,
        detector_info: dict[str, Any] | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.recognizer = recognizer.to(self.device).eval()
        self.detector = detector.to(self.device).eval() if detector is not None else None
        self.charset = charset
        self.detector_size = detector_size
        self.threshold = threshold
        self.rec_height = rec_height
        self.rec_width = rec_width
        self.split_lines_without_detector = split_lines_without_detector
        # >0 when the detector was trained on shrunk targets (stored in its checkpoint)
        self.unclip_ratio = unclip_ratio
        # binarize with the detector's own threshold map instead of a fixed cutoff
        self.learned_threshold = learned_threshold
        # what the detector checkpoint recorded about itself, for display
        self.detector_info = detector_info or {}

    @classmethod
    def load(
        cls,
        recognizer_ckpt: str | Path,
        detector_ckpt: str | Path | None = None,
        device: torch.device | str = "auto",
        charset: str = DEFAULT_CHARSET,
        **kwargs: Any,
    ) -> OCRPipeline:
        """Build a pipeline from checkpoint files on disk."""
        device = resolve_device(device)
        recognizer = CRNNRecognizer(charset=charset).to(device)
        rec_ckpt = load_checkpoint(recognizer_ckpt, recognizer, device=device, strict=False)
        # Feed the recognizer crops shaped the way it was trained; checkpoints from
        # before this metadata existed were all trained at the legacy 32x512.
        if "rec_height" in rec_ckpt:
            kwargs.setdefault("rec_height", int(rec_ckpt["rec_height"]))
            kwargs.setdefault("rec_width", int(rec_ckpt["rec_width"]))
        else:
            kwargs.setdefault("rec_height", LEGACY_RECOGNIZER_SIZE[0])
            kwargs.setdefault("rec_width", LEGACY_RECOGNIZER_SIZE[1])

        detector = None
        if detector_ckpt:
            detector = DBNet().to(device)
            ckpt = load_checkpoint(detector_ckpt, detector, device=device, strict=False)
            # A detector trained on shrunk targets records how to undo the shrink;
            # older checkpoints carry nothing and get unclip_ratio 0 (no change).
            kwargs.setdefault("unclip_ratio", float(ckpt.get("unclip_ratio", 0.0)))
            # A DB-trained detector has a threshold head worth using; default to it.
            is_db = ckpt.get("detector_loss") == "db"
            kwargs.setdefault("learned_threshold", is_db)
            kwargs.setdefault("detector_info", {
                "path": str(detector_ckpt),
                "loss": ckpt.get("detector_loss", "bce (pre-DB checkpoint)"),
                "epoch": ckpt.get("epoch"),
                "shrink_ratio": float(ckpt.get("shrink_ratio", 0.0)),
                "unclip_ratio": float(ckpt.get("unclip_ratio", 0.0)),
                "has_threshold_head": is_db,
            })

        return cls(recognizer=recognizer, detector=detector, device=device, charset=charset, **kwargs)

    @property
    def has_detector(self) -> bool:
        return self.detector is not None

    def _line_boxes(
        self,
        image: Image.Image,
        use_detector: bool = True,
        threshold: float | None = None,
        split_lines: bool | None = None,
        learned_threshold: bool | None = None,
    ) -> tuple[list[Box], np.ndarray | None, bool]:
        min_w, min_h = MIN_SPLITTABLE_SIZE
        splittable = image.width >= min_w and image.height >= min_h
        whole_page: list[Box] = [(0, 0, image.width, image.height)]
        split_lines = self.split_lines_without_detector if split_lines is None else split_lines

        if self.detector is None or not use_detector:
            if split_lines and splittable:
                split = split_regions_into_lines(image, whole_page)
                if split:
                    return split, None, True
            return whole_page, None, False

        boxes, prob_map = detect_text_regions(
            detector=self.detector,
            image=image,
            device=self.device,
            image_size=self.detector_size,
            threshold=self.threshold if threshold is None else threshold,
            unclip_ratio=self.unclip_ratio,
            learned_threshold=self.learned_threshold if learned_threshold is None else learned_threshold,
        )
        boxes = split_regions_into_lines(image, boxes or whole_page)
        if len(boxes) <= 1 and splittable:
            fallback = split_regions_into_lines(image, whole_page)
            if len(fallback) > len(boxes):
                return fallback, prob_map, True
        return boxes, prob_map, False

    def read_image(
        self,
        image: Image.Image,
        use_detector: bool = True,
        threshold: float | None = None,
        split_lines: bool | None = None,
        learned_threshold: bool | None = None,
    ) -> PageResult:
        """OCR a single in-memory image. Returns text plus per-line boxes.

        The keyword arguments override the pipeline defaults for this call only,
        so a caller serving many requests can vary them without reloading models.
        """
        image = image.convert("L")
        boxes, prob_map, fallback = self._line_boxes(image, use_detector, threshold, split_lines, learned_threshold)
        texts = recognize_crops(
            self.recognizer,
            [image.crop(box) for box in boxes],
            self.device,
            charset=self.charset,
            height=self.rec_height,
            width=self.rec_width,
        )
        return PageResult(
            text="\n".join(texts),
            lines=[LineResult(box=box, text=text) for box, text in zip(boxes, texts)],
            used_line_split_fallback=fallback,
            prob_map=prob_map,
        )

    def read_path(self, image_path: str | Path) -> PageResult:
        image_path = Path(image_path)
        result = self.read_image(Image.open(image_path))
        result.image = image_path.name
        result.path = str(image_path)
        return result


def _write_artifacts(
    image: Image.Image,
    result: PageResult,
    stem: str,
    output_dir: Path,
    save_crops: bool,
    save_visualizations: bool,
) -> None:
    if save_crops:
        crops_dir = output_dir / "crops" / stem
        crops_dir.mkdir(parents=True, exist_ok=True)
        for idx, line in enumerate(result.lines):
            image.crop(line.box).save(crops_dir / f"region_{idx:03d}.png")

    if save_visualizations:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_overlay(image, [line.box for line in result.lines], vis_dir / f"{stem}_boxes.png")
        if result.prob_map is not None:
            heat = (np.clip(result.prob_map, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(vis_dir / f"{stem}_prob.png"), heat)


def run_ocr_image(
    image_path: str | Path,
    pipeline: OCRPipeline,
    output_dir: str | Path | None = None,
    save_crops: bool = False,
    save_visualizations: bool = False,
) -> dict[str, Any]:
    image_path = Path(image_path)
    image = Image.open(image_path).convert("L")
    output_dir = Path(output_dir) if output_dir else image_path.parent / "ocr_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = pipeline.read_image(image)
    result.image = image_path.name
    result.path = str(image_path)
    _write_artifacts(image, result, image_path.stem, output_dir, save_crops, save_visualizations)

    include_regions = pipeline.has_detector or pipeline.split_lines_without_detector
    return result.to_dict(include_regions=include_regions)


def run_ocr_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    pipeline: OCRPipeline,
    save_crops: bool = False,
    save_visualizations: bool = False,
    log_every: int = 10,
) -> list[dict[str, Any]]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    text_dir = output_dir / "texts"
    text_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    results: list[dict[str, Any]] = []
    progress = ProgressLogger("ocr folder", len(image_paths), log_every)
    total_regions = 0
    fallback_count = 0

    for step, image_path in enumerate(image_paths, start=1):
        result = run_ocr_image(
            image_path=image_path,
            pipeline=pipeline,
            output_dir=output_dir,
            save_crops=save_crops,
            save_visualizations=save_visualizations,
        )
        (text_dir / f"{image_path.stem}.txt").write_text(result["text"])
        results.append(result)
        total_regions += len(result["regions"]) or 1
        fallback_count += int(bool(result.get("used_line_split_fallback")))
        if progress.should_log(step):
            progress.log(
                step,
                f"images done {step} avg_regions {total_regions / step:.1f} "
                f"line_split_fallbacks {fallback_count} loss n/a accuracy n/a"
                " (no OCR labels provided)",
            )

    (output_dir / "predictions.json").write_text(json.dumps({"predictions": results}, indent=2))
    return results
