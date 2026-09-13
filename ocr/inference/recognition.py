"""Run the CRNN recognizer over one or many line crops."""

from __future__ import annotations

import torch
from PIL import Image

from ocr.config import DEFAULT_CHARSET, DEFAULT_RECOGNIZER_HEIGHT, DEFAULT_RECOGNIZER_WIDTH
from ocr.ctc import decode_ctc_greedy
from ocr.data.transforms import RecognitionTransform


def recognize_crop(
    recognizer: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    charset: str = DEFAULT_CHARSET,
    height: int = DEFAULT_RECOGNIZER_HEIGHT,
    width: int = DEFAULT_RECOGNIZER_WIDTH,
) -> str:
    return recognize_crops(recognizer, [image], device, charset, height, width)[0]


def recognize_crops(
    recognizer: torch.nn.Module,
    images: list[Image.Image],
    device: torch.device,
    charset: str = DEFAULT_CHARSET,
    height: int = DEFAULT_RECOGNIZER_HEIGHT,
    width: int = DEFAULT_RECOGNIZER_WIDTH,
    batch_size: int = 16,
) -> list[str]:
    """Batched greedy-CTC recognition. No lexicon, spell, or language-model correction."""
    if not images:
        return []
    transform = RecognitionTransform(height=height, width=width, augment=False)
    recognizer.eval()
    texts: list[str] = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        tensor = torch.stack([transform(image) for image in chunk], dim=0).to(device)
        with torch.no_grad():
            logits = recognizer(tensor)
        texts.extend(decode_ctc_greedy(logits, charset=charset, time_first=True))
    return texts
