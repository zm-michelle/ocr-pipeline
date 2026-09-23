from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from ocr.config import DEFAULT_RECOGNIZER_HEIGHT, DEFAULT_RECOGNIZER_WIDTH


def to_grayscale(image: Image.Image) -> Image.Image:
    return image.convert("L")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(to_grayscale(image), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - 0.5) / 0.5


def resize_page(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return to_grayscale(image).resize(size, Image.BILINEAR)


def resize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    return mask.convert("L").resize(size, Image.NEAREST)


def fit_text_image(
    image: Image.Image,
    height: int = DEFAULT_RECOGNIZER_HEIGHT,
    width: int = DEFAULT_RECOGNIZER_WIDTH,
) -> Image.Image:
    image = to_grayscale(image)
    w, h = image.size
    if h <= 0 or w <= 0:
        return Image.new("L", (width, height), 255)

    new_w = max(1, min(width, int(round(w * (height / h)))))
    resized = image.resize((new_w, height), Image.BILINEAR)
    canvas = Image.new("L", (width, height), 255)
    canvas.paste(resized, (0, 0))
    return canvas


def random_affine_mild(image: Image.Image, max_translate: int = 2) -> Image.Image:
    tx = random.randint(-max_translate, max_translate)
    ty = random.randint(-max_translate, max_translate)
    return image.transform(image.size, Image.AFFINE, (1, 0, tx, 0, 1, ty), fillcolor=255)


def add_gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    arr += np.random.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_salt_pepper(image: Image.Image, amount: float) -> Image.Image:
    arr = np.asarray(image).copy()
    mask = np.random.random(arr.shape)
    arr[mask < amount / 2] = 0
    arr[(mask >= amount / 2) & (mask < amount)] = 255
    return Image.fromarray(arr.astype(np.uint8))


def add_stain(image: Image.Image, opacity: float = 0.18) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    h, w = arr.shape
    cx = random.randint(0, w)
    cy = random.randint(0, h)
    rx = random.randint(max(12, w // 12), max(16, w // 3))
    ry = random.randint(max(12, h // 16), max(16, h // 4))
    yy, xx = np.mgrid[:h, :w]
    blob = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    mask = np.clip(1 - blob, 0, 1) ** 2
    stain = random.uniform(90, 170)
    arr = arr * (1 - opacity * mask) + stain * (opacity * mask)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _blob_mask(h: int, w: int, cx: float, cy: float, radius: float, lobes: int = 5) -> np.ndarray:
    """Irregular blob: a union of jittered ellipses around (cx, cy), values in [0, 1]."""
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(lobes):
        ox, oy = random.uniform(-0.45, 0.45) * radius, random.uniform(-0.45, 0.45) * radius
        rx, ry = radius * random.uniform(0.55, 1.0), radius * random.uniform(0.55, 1.0)
        d = ((xx - cx - ox) / rx) ** 2 + ((yy - cy - oy) / ry) ** 2
        mask = np.maximum(mask, np.clip(1.0 - d, 0, 1))
    return mask


def add_coffee_stain(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """A heavy, dark, ringed blotch over the text, like NoisyOffice's coffee stains.

    Unlike `add_stain` (a faint tint), this can sit on top of the print: inside
    the blob the paper drops to 40-140 gray with mottled texture, and the rim -
    where a real stain dries - is darker still. Text under it stays visible but
    low-contrast, which is exactly the case the recognizer must learn.
    """
    arr = np.asarray(image).astype(np.float32)
    h, w = arr.shape
    for _ in range(random.randint(1, 2)):
        cx, cy = random.uniform(0, w), random.uniform(0, h)
        radius = random.uniform(min(h, w) * 0.25, min(h, w) * 0.7)
        blob = _blob_mask(h, w, cx, cy, radius)
        body = (blob > 0.15).astype(np.float32)
        body = cv2.GaussianBlur(body, (0, 0), 3)
        rim = np.clip(cv2.GaussianBlur((blob > 0.15).astype(np.float32), (0, 0), 6) - cv2.GaussianBlur((blob > 0.15).astype(np.float32), (0, 0), 1.5), 0, 1)
        rim = rim / (rim.max() + 1e-6)
        texture = cv2.GaussianBlur(np.random.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.5)
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-6)
        stain_gray = random.uniform(60, 150) * (0.75 + 0.5 * texture)
        opacity = random.uniform(0.35, 0.75) * strength
        alpha = np.clip(opacity * body + 0.5 * opacity * rim, 0, 0.92)
        arr = arr * (1 - alpha) + stain_gray * alpha
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_cup_ring(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """Just the rim of a cup: a thin dark annulus, sometimes broken."""
    arr = np.asarray(image).astype(np.float32)
    h, w = arr.shape
    cx, cy = random.uniform(0.1 * w, 0.9 * w), random.uniform(0.1 * h, 0.9 * h)
    r = random.uniform(min(h, w) * 0.25, min(h, w) * 0.55)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring = np.exp(-((d - r) ** 2) / (2 * random.uniform(2.0, 5.0) ** 2))
    if random.random() < 0.5:  # broken ring
        ang = np.arctan2(yy - cy, xx - cx)
        a0 = random.uniform(-np.pi, np.pi)
        ring *= (np.cos(ang - a0) > random.uniform(-0.6, 0.3)).astype(np.float32)
    alpha = np.clip(ring * random.uniform(0.5, 0.9) * strength, 0, 0.9)
    arr = arr * (1 - alpha) + random.uniform(40, 120) * alpha
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_uneven_illumination(image: Image.Image, strength: float = 0.18) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    h, w = arr.shape
    left = random.uniform(1 - strength, 1 + strength)
    right = random.uniform(1 - strength, 1 + strength)
    top = random.uniform(1 - strength, 1 + strength)
    bottom = random.uniform(1 - strength, 1 + strength)
    x_grad = np.linspace(left, right, w)[None, :]
    y_grad = np.linspace(top, bottom, h)[:, None]
    arr *= (x_grad + y_grad) / 2
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_smudge(image: Image.Image, strength: int = 7) -> Image.Image:
    arr = np.asarray(image)
    k = max(3, strength | 1)
    blurred = cv2.GaussianBlur(arr, (k, k), 0)
    mask = np.zeros_like(arr, dtype=np.float32)
    h, w = arr.shape
    for _ in range(random.randint(1, 3)):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2 = min(w - 1, max(0, x1 + random.randint(-w // 4, w // 4)))
        y2 = min(h - 1, max(0, y1 + random.randint(-h // 8, h // 8)))
        cv2.line(mask, (x1, y1), (x2, y2), 1.0, random.randint(5, 18))
    mask = cv2.GaussianBlur(mask, (21, 21), 0)[..., None] if mask.ndim == 3 else cv2.GaussianBlur(mask, (21, 21), 0)
    out = arr.astype(np.float32) * (1 - mask) + blurred.astype(np.float32) * mask
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def random_stroke_weight(image: Image.Image) -> Image.Image:
    """Erode or dilate ink by a pixel: bold-vs-light print, and scanner blooming."""
    arr = np.asarray(image)
    kernel = np.ones((2, 2), np.uint8) if random.random() < 0.7 else np.ones((3, 1), np.uint8)
    # text is dark on light, so "thicker ink" is a grayscale erosion
    out = cv2.erode(arr, kernel) if random.random() < 0.5 else cv2.dilate(arr, kernel)
    return Image.fromarray(out)


def random_skew(image: Image.Image, max_degrees: float = 2.0) -> Image.Image:
    angle = random.uniform(-max_degrees, max_degrees)
    return image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=255)


def random_scale_jitter(image: Image.Image, low: float = 0.85, high: float = 1.15) -> Image.Image:
    """Resize by a random factor before the final fit, so glyph size at 32 px varies."""
    f = random.uniform(low, high)
    w, h = image.size
    return image.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BILINEAR)


def mild_compression_artifacts(image: Image.Image, quality: int = 45) -> Image.Image:
    arr = np.asarray(image)
    ok, encoded = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return Image.fromarray(decoded)


@dataclass
class RecognitionTransform:
    height: int = DEFAULT_RECOGNIZER_HEIGHT
    width: int = DEFAULT_RECOGNIZER_WIDTH
    augment: bool = False

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = to_grayscale(image)
        if self.augment:
            # geometry first (what the scanner did), then photometry (what the paper did)
            if random.random() < 0.30:
                image = random_scale_jitter(image)
            if random.random() < 0.25:
                image = random_skew(image, 2.0)
            if random.random() < 0.25:
                image = random_affine_mild(image)
            if random.random() < 0.30:
                image = random_stroke_weight(image)
            if random.random() < 0.35:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.65, 1.35))
            if random.random() < 0.25:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.0)))
            if random.random() < 0.25:
                image = add_gaussian_noise(image, random.uniform(3, 12))
            if random.random() < 0.15:
                image = add_salt_pepper(image, random.uniform(0.002, 0.015))
            if random.random() < 0.20:
                image = mild_compression_artifacts(image, quality=random.randint(30, 70))
        return normalize_tensor(pil_to_tensor(fit_text_image(image, self.height, self.width)))


@dataclass
class DetectionTransform:
    """Augment (optionally) and resize a page to the detector input size.

    Targets are no longer resized alongside the image: the dataset builds them
    directly at `size` from the scaled boxes, so the threshold map stays exact.
    """

    size: tuple[int, int]
    augment: bool = False

    def augment_image(self, image: Image.Image) -> Image.Image:
        image = to_grayscale(image)
        if self.augment:
            if random.random() < 0.35:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
            if random.random() < 0.25:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.9)))
            if random.random() < 0.20:
                image = add_gaussian_noise(image, random.uniform(2, 10))
        return image

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return normalize_tensor(pil_to_tensor(resize_page(self.augment_image(image), self.size)))
