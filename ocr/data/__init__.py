"""Datasets, image transforms, and synthetic page/line generation."""

from ocr.data.catalog import Collection, DatasetCatalog, GroundTruth
from ocr.data.datasets import (
    DetectionDataset,
    OCRFolderDataset,
    PrintedLineDataset,
    detection_collate,
    list_image_files,
    recognition_collate,
)
from ocr.data.synthetic import DegradationConfig, generate_synthetic_dataset
from ocr.data.transforms import DetectionTransform, RecognitionTransform

__all__ = [
    "Collection",
    "DatasetCatalog",
    "GroundTruth",
    "DetectionDataset",
    "OCRFolderDataset",
    "PrintedLineDataset",
    "detection_collate",
    "recognition_collate",
    "list_image_files",
    "DegradationConfig",
    "generate_synthetic_dataset",
    "DetectionTransform",
    "RecognitionTransform",
]
