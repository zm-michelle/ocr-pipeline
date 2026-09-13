"""OCR for degraded printed documents: detector -> line crops -> CRNN + CTC."""

from ocr.inference.pipeline import LineResult, OCRPipeline, PageResult

__version__ = "0.2.0"
__all__ = ["OCRPipeline", "PageResult", "LineResult", "__version__"]
