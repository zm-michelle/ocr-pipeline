"""Network definitions: the text detector and the line recognizer."""

from ocr.models.blocks import ConvBNAct, DepthwiseSeparableBlock
from ocr.models.detector import DBNet
from ocr.models.recognizer import CRNNRecognizer

__all__ = ["ConvBNAct", "DepthwiseSeparableBlock", "DBNet", "CRNNRecognizer"]
