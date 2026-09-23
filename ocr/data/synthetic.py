from __future__ import annotations

import functools
import json
import math
import multiprocessing
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr.config import COMMON_FONT_DIRS, DEFAULT_PAGE_SIZE, FONT_FAMILIES, FONT_FILE_BLOCKLIST, FONT_FILE_HINTS
from ocr.data.transforms import (
    add_gaussian_noise,
    add_salt_pepper,
    add_smudge,
    add_coffee_stain,
    add_cup_ring,
    add_stain,
    add_uneven_illumination,
    mild_compression_artifacts,
)
from ocr.data.text_source import paragraph_words as _corpus_paragraph_words
from ocr.utils import ProgressLogger


OFFICE_WORDS = [
    "account",
    "analysis",
    "archive",
    "balance",
    "budget",
    "client",
    "contract",
    "department",
    "document",
    "estimate",
    "finance",
    "invoice",
    "ledger",
    "manager",
    "meeting",
    "office",
    "policy",
    "printed",
    "project",
    "quality",
    "receipt",
    "record",
    "report",
    "review",
    "schedule",
    "service",
    "shipping",
    "summary",
    "system",
    "total",
    "transfer",
    "vendor",
    "version",
]

PARAGRAPH_SENTENCES = [
    "There are several classic spatial filters for reducing image noise from scanned documents.",
    "The mean filter and the median filter replace each pixel using information from the neighborhood.",
    "This procedure reduces image noise but blurs the image and changes the appearance of small letters.",
    "The main goal was to train a neural network in a supervised way to recover a clean image from a noisy one.",
    "In this particular case, it was much easier to restore a noisy image from a clean one than to clean every image manually.",
    "A new printed document collection has recently been prepared for experiments with degraded office images.",
    "The database contains pages with several typefaces, font sizes, emphasized text, stains, folds, and low contrast.",
    "First of all, most documents do not contain the same amount of text or the same line spacing.",
    "Another important reason was to create restricted tasks commonly used in document analysis and recognition.",
    "The forms also contain a brief set of instructions given to the reader before the document was scanned.",
]


@dataclass
class DegradationConfig:
    gaussian_noise: bool = True
    salt_pepper: bool = True
    blur: bool = True
    low_contrast: bool = True
    uneven_illumination: bool = True
    smudges: bool = True
    stains: bool = True
    faded_print: bool = True
    speckle: bool = True
    compression: bool = False
    severity: float = 0.7


class FontResolver:
    def __init__(self) -> None:
        self._font_paths = self._scan_fonts()
        self._cache: dict[tuple[str, int, bool, bool], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    def _scan_fonts(self) -> list[Path]:
        paths: list[Path] = []
        for root in COMMON_FONT_DIRS:
            if root.exists():
                paths.extend(root.rglob("*.ttf"))
                paths.extend(root.rglob("*.otf"))
                paths.extend(root.rglob("*.ttc"))
        return paths

    @staticmethod
    def _style_of(path: Path) -> tuple[bool, bool]:
        """(bold, italic) read from the file name; URW uses Demi/BdIta, others Bold/Italic/Oblique."""
        n = path.name.lower().replace("-", " ").replace("_", " ")
        bold = any(t in n for t in ("bold", "demi", "bdita", "heavy", "black"))
        italic = any(t in n for t in ("italic", "oblique", "bdita"))
        return bold, italic

    def candidates(self, family: str, bold: bool = False, italic: bool = False) -> list[Path]:
        """Every installed face for `family` in the requested style, falling back to nearer styles.

        Tiers are tried in order - exact style, then drop italic, then drop bold, then
        regular - and the first non-empty tier is returned, so a request for bold
        italic on a machine with no bold-italic faces still lands on something close.
        """
        names = FONT_FAMILIES.get(family, FONT_FAMILIES["serif"])
        tiers: list[tuple[bool, bool]] = [(bold, italic)]
        if italic:
            tiers.append((bold, False))
        if bold:
            tiers.append((False, italic))
        tiers.append((False, False))

        matching: list[Path] = []
        for name in names:
            hints = [h.lower().replace("-", " ").replace("_", " ") for h in FONT_FILE_HINTS.get(name, [name.lower()])]
            for path in self._font_paths:
                normalized = path.name.lower().replace("-", " ").replace("_", " ")
                if any(b in normalized for b in FONT_FILE_BLOCKLIST):
                    continue
                if any(h in normalized for h in hints) and path not in matching:
                    matching.append(path)

        for want_bold, want_italic in tiers:
            tier = [p for p in matching if self._style_of(p) == (want_bold, want_italic)]
            if tier:
                return tier
        return []

    def _find_path(self, family: str, bold: bool = False, italic: bool = False) -> Path | None:
        found = self.candidates(family, bold, italic)
        return found[0] if found else None

    def pick(self, family: str, bold: bool = False, italic: bool = False) -> Path | None:
        """One random face for this family and style. Uses `random`, so it follows the dataset seed."""
        found = self.candidates(family, bold, italic)
        return random.choice(found) if found else None

    def load(self, path: Path | str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        key = (str(path), size)
        if key not in self._cache:
            try:
                self._cache[key] = ImageFont.truetype(str(path), size=size) if path else ImageFont.load_default()
            except OSError:
                self._cache[key] = ImageFont.load_default()
        return self._cache[key]

    def report(self) -> dict[str, list[Path]]:
        """Regular-weight faces available per family (empty means the bitmap fallback)."""
        return {family: self.candidates(family) for family in FONT_FAMILIES}

    def require_real_fonts(self) -> None:
        """Fail loudly instead of rendering a dataset with PIL's 8px bitmap font.

        That fallback is silent, and a run of 20k pages drawn with it looks like
        a dataset right up until the trained model reads nothing.
        """
        resolved = self.report()
        missing = [family for family, paths in resolved.items() if not paths]
        for family, paths in resolved.items():
            if paths:
                names = ", ".join(p.name for p in paths[:4]) + (", ..." if len(paths) > 4 else "")
                print(f"font {family:10s} -> {len(paths)} face(s): {names}")
            else:
                print(f"font {family:10s} -> NOT FOUND (bitmap fallback)")
        if missing:
            searched = ", ".join(str(d) for d in COMMON_FONT_DIRS if d.exists()) or "(none of the font dirs exist)"
            raise RuntimeError(
                f"no TrueType font found for {missing}. Searched: {searched}. "
                "Install DejaVu/Liberation (remote/fonts.sh does this without root) "
                "or set OCR_FONT_DIRS to a directory of .ttf files."
            )

    def get(
        self,
        family: str,
        size: int,
        bold: bool = False,
        italic: bool = False,
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """A random matching face at `size`. Prefer pick()+load() when the choice must be recorded."""
        return self.load(self.pick(family, bold, italic), size)


def _font_size_for_style(style: str) -> int:
    if style == "footnote":
        return random.randint(14, 18)
    if style == "large":
        return random.randint(28, 40)
    return random.randint(20, 28)


def _random_line(max_words: int) -> str:
    words = random.randint(4, max_words)
    pieces: list[str] = []
    for i in range(words):
        word = random.choice(OFFICE_WORDS)
        r = random.random()
        if r < 0.12:
            word = word.upper()
        elif r < 0.28:
            word = word.capitalize()
        if random.random() < 0.08:
            word += str(random.randint(1, 99))
        pieces.append(word)
        if i < words - 1 and random.random() < 0.12:
            pieces[-1] += random.choice([",", ";", ":", "-"])
    line = " ".join(pieces)
    if random.random() < 0.55:
        line += random.choice([".", ".", ":", ";"])
    return line


def _paragraph_words() -> list[str]:
    """Real sentences with numbers/dates/codes spliced in (see ocr/data/text_source.py)."""
    return _corpus_paragraph_words()


def _wrap_words_to_lines(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    target_width: int,
    n_lines: int,
) -> list[str]:
    words = _paragraph_words()
    lines: list[str] = []
    cursor = 0
    for _ in range(n_lines):
        if cursor >= len(words):
            words.extend(_paragraph_words())
        line_words: list[str] = []
        while cursor < len(words):
            candidate = " ".join(line_words + [words[cursor]])
            if line_words and _text_bbox(draw, candidate, font)[0] > target_width:
                break
            line_words.append(words[cursor])
            cursor += 1
        line = " ".join(line_words)
        if random.random() < 0.18:
            line = "  ".join(line.split(" ", 1)) if " " in line else line
        lines.append(line)
    return lines


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _add_paper_texture(image: Image.Image, strength: float = 0.55) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    h, w = arr.shape
    fine = np.random.normal(0, 4.5 * strength, arr.shape)
    coarse_small = np.random.normal(0, 11 * strength, (max(2, h // 18), max(2, w // 18))).astype(np.float32)
    coarse = cv2.resize(coarse_small, (w, h), interpolation=cv2.INTER_CUBIC)
    arr += fine + coarse
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _add_fold_or_wrinkle(image: Image.Image, strength: float = 0.45) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    h, w = arr.shape
    for _ in range(random.randint(1, 3)):
        if random.random() < 0.55:
            center = random.randint(0, w - 1)
            width = random.uniform(8, 28)
            x = np.arange(w)[None, :]
            shadow = np.exp(-((x - center) ** 2) / (2 * width**2))
            arr -= shadow * random.uniform(10, 28) * strength
        else:
            center = random.randint(0, h - 1)
            width = random.uniform(5, 18)
            y = np.arange(h)[:, None]
            shadow = np.exp(-((y - center) ** 2) / (2 * width**2))
            arr -= shadow * random.uniform(8, 22) * strength
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _apply_degradation(image: Image.Image, cfg: DegradationConfig) -> Image.Image:
    s = max(0.0, min(1.0, cfg.severity))
    out = image.convert("L")
    # Real scans (NoisyOffice) are white paper with heavy LOCAL damage; the old
    # stack dimmed whole pages to gray. Keep the paper light and do the damage locally.
    out = _add_paper_texture(out, strength=0.25 + s * 0.3)
    if random.random() < 0.40:
        out = _add_fold_or_wrinkle(out, strength=s)
    if cfg.faded_print and random.random() < 0.65:
        arr = np.asarray(out).astype(np.float32)
        dark = arr < 180
        arr[dark] = arr[dark] * (1 - 0.25 * s) + 255 * (0.25 * s)
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if cfg.low_contrast and random.random() < 0.45:
        arr = np.asarray(out).astype(np.float32)
        # fade toward the paper, not toward mid-gray, so the page stays white
        paper = float(np.percentile(arr, 90))
        arr = paper + (arr - paper) * random.uniform(0.6, 1.0)
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if cfg.uneven_illumination and random.random() < 0.55:
        out = add_uneven_illumination(out, strength=0.18 * s)
    if cfg.stains:
        r = random.random()
        if r < 0.30:
            out = add_coffee_stain(out, strength=s)  # heavy, over the text
        elif r < 0.42:
            out = add_cup_ring(out, strength=s)
        elif r < 0.65:
            out = add_stain(out, opacity=random.uniform(0.08, 0.22) * s)  # faint tint
    if cfg.smudges and random.random() < 0.35:
        out = add_smudge(out, strength=random.randint(3, 9))
    if cfg.blur and random.random() < 0.45:
        radius = random.uniform(0.25, 1.0 + s)
        arr = cv2.GaussianBlur(np.asarray(out), (0, 0), radius)
        out = Image.fromarray(arr)
    if cfg.gaussian_noise and random.random() < 0.75:
        out = add_gaussian_noise(out, sigma=random.uniform(3, 18) * s)
    if cfg.salt_pepper and random.random() < 0.55:
        out = add_salt_pepper(out, amount=random.uniform(0.002, 0.02) * s)
    if cfg.speckle and random.random() < 0.45:
        arr = np.asarray(out).copy()
        count = int(arr.size * random.uniform(0.0008, 0.004) * s)
        if count:
            ys = np.random.randint(0, arr.shape[0], count)
            xs = np.random.randint(0, arr.shape[1], count)
            arr[ys, xs] = np.random.randint(0, 80, count)
        out = Image.fromarray(arr)
    if cfg.compression and random.random() < 0.35:
        out = mild_compression_artifacts(out, quality=random.randint(35, 70))
    return out


def _visible_substring(text: str, font: ImageFont.ImageFont, start_px: int, end_px: int) -> str:
    if start_px <= 0 and end_px >= int(font.getlength(text)):
        return text
    spans: list[tuple[float, float, str]] = []
    cursor = 0.0
    for ch in text:
        next_cursor = cursor + float(font.getlength(ch))
        spans.append((cursor, next_cursor, ch))
        cursor = next_cursor
    chars = [ch for left, right, ch in spans if start_px <= (left + right) / 2 <= end_px]
    return "".join(chars)


def _crop_page_edges(
    image: Image.Image,
    lines: list[dict],
    crop_prob: float,
    resolver: FontResolver,
) -> tuple[Image.Image, list[dict]]:
    if random.random() >= crop_prob:
        return image, lines
    w, h = image.size
    left = random.randint(0, max(1, int(w * 0.08))) if random.random() < 0.55 else 0
    top = random.randint(0, max(1, int(h * 0.05))) if random.random() < 0.35 else 0
    right = w - (random.randint(0, max(1, int(w * 0.08))) if random.random() < 0.55 else 0)
    bottom = h - (random.randint(0, max(1, int(h * 0.05))) if random.random() < 0.35 else 0)
    cropped = image.crop((left, top, right, bottom))
    adjusted: list[dict] = []
    for line in lines:
        x1, y1, x2, y2 = line["box"]
        nx1, ny1 = max(0, x1 - left), max(0, y1 - top)
        nx2, ny2 = min(right - left, x2 - left), min(bottom - top, y2 - top)
        if nx2 - nx1 >= 8 and ny2 - ny1 >= 8:
            new_line = dict(line)
            text_x = line.get("origin", [x1, y1])[0]
            font = resolver.get(
                line["font_family"],
                line["font_size"],
                line["bold_like"],
                line.get("italic_like", False),
            )
            text_width = int(font.getlength(line["text"]))
            visible_left = max(0, left - text_x)
            visible_right = min(text_width, right - text_x)
            new_line["text"] = _visible_substring(line["text"], font, visible_left, visible_right)
            new_line["box"] = [nx1, ny1, nx2, ny2]
            if new_line["text"]:
                adjusted.append(new_line)
    return cropped, adjusted


def _draw_document(
    page_size: tuple[int, int],
    min_lines: int,
    max_lines: int,
    resolver: FontResolver,
    render_profile: str = "noisyoffice",
) -> tuple[Image.Image, list[dict]]:
    width, height = page_size
    background = random.randint(232, 247) if render_profile == "noisyoffice" else 255
    image = Image.new("L", (width, height), background)
    draw = ImageDraw.Draw(image)

    if render_profile == "noisyoffice":
        family = random.choices(["serif", "sans", "typewriter"], weights=[0.50, 0.30, 0.20])[0]
        # NoisyOffice lines are ~20-36 px tall; the old 14-23 px range left the
        # recognizer with no large-print experience, so weight "large" up and extend it.
        style = random.choices(["footnote", "normal", "large"], weights=[0.18, 0.52, 0.30])[0]
        font_size = {"footnote": random.randint(14, 16), "normal": random.randint(17, 22), "large": random.randint(23, 30)}[style]
        bold = random.random() < 0.18
        italic = family == "serif" and random.random() < 0.65
        font_file = resolver.pick(family, bold=bold, italic=italic)
        font = resolver.load(font_file, font_size)
        target_min = max(8, min_lines)
        target_max = max(target_min, min(10, max_lines))
        n_lines = random.randint(target_min, target_max)
        line_height = font_size + random.randint(8, 12)
        y = random.randint(-3, 5)
        wrap_width = random.randint(int(width * 0.96), int(width * 1.18))
        source_lines = _wrap_words_to_lines(draw, font, wrap_width, n_lines)
        x_base = random.randint(-22, 26)
    else:
        family = random.choice(list(FONT_FAMILIES))
        style = random.choices(["footnote", "normal", "large"], weights=[0.18, 0.68, 0.14])[0]
        font_size = _font_size_for_style(style)
        bold = random.random() < 0.22
        italic = False
        font_file = resolver.pick(family, bold=bold)
        font = resolver.load(font_file, font_size)
        margin_x = random.randint(max(18, width // 28), max(24, width // 10))
        margin_top = random.randint(max(18, height // 32), max(32, height // 10))
        line_gap = random.randint(max(4, font_size // 4), max(8, font_size))
        line_height = font_size + line_gap
        usable_h = height - margin_top - random.randint(24, 90)
        max_fit = max(min_lines, min(max_lines, usable_h // max(1, line_height)))
        n_lines = random.randint(min_lines, max(min_lines, max_fit))
        y = margin_top
        source_lines = [_random_line(max_words={"dense": 13, "normal": 10, "loose": 7}[random.choice(["dense", "normal", "loose"])]) for _ in range(n_lines)]
        x_base = margin_x

    lines: list[dict] = []

    for raw_text in source_lines:
        text = raw_text
        text_w, text_h = _text_bbox(draw, text, font)
        if render_profile == "document":
            max_text_w = width - 2 * x_base
            while text_w > max_text_w and " " in text:
                text = " ".join(text.split()[:-1])
                text_w, text_h = _text_bbox(draw, text, font)

        x = x_base + random.randint(-8, 12)
        if render_profile == "document":
            x = max(2, x)
            if random.random() < 0.10:
                x += random.randint(10, 45)

        fill = random.randint(12, 55) if render_profile == "noisyoffice" else random.randint(0, 45)
        draw.text((x, y), text, fill=fill, font=font)
        visible_left = max(0, -x)
        visible_right = min(int(font.getlength(text)), width - x)
        visible_text = _visible_substring(text, font, visible_left, visible_right)
        if not visible_text:
            y += line_height + random.randint(-1, 2)
            continue
        pad = random.randint(1, 3) if render_profile == "noisyoffice" else random.randint(2, 5)
        bx1 = max(0, x - pad)
        by1 = max(0, y - pad)
        bx2 = min(width, x + text_w + pad)
        by2 = min(height, y + text_h + pad)
        if bx2 <= bx1 or by2 <= by1:
            y += line_height + (random.randint(-1, 2) if render_profile == "noisyoffice" else random.randint(-2, 6))
            continue
        lines.append(
            {
                "text": visible_text,
                "box": [bx1, by1, bx2, by2],
                "font_family": family,
                "font_file": str(font_file) if font_file else None,
                "font_size": font_size,
                "bold_like": bold,
                "italic_like": italic,
                "origin": [max(0, x), y],
            }
        )
        y += line_height + (random.randint(-1, 2) if render_profile == "noisyoffice" else random.randint(-2, 6))
    return image, lines


MIN_LINE_W, MIN_LINE_H = 8, 4


def _clip_lines(lines: list[dict], width: int, height: int) -> list[dict]:
    """Clamp every box into the frame and drop lines left without a usable one.

    Any page-level transform (skew, column offset) can push a line partly or
    fully outside the page. Clamping one corner but not the other leaves an
    inverted box such as [3, 260, 99, 258], which survives all the way to
    PIL's crop and kills the run there; and a two-pixel sliver would keep the
    full line text as its label, which is pure label noise.
    """
    kept: list[dict] = []
    for line in lines:
        x1, y1, x2, y2 = line["box"]
        x1, x2 = min(max(0, x1), width), min(max(0, x2), width)
        y1, y2 = min(max(0, y1), height), min(max(0, y2), height)
        if x2 - x1 < MIN_LINE_W or y2 - y1 < MIN_LINE_H:
            continue
        line["box"] = [x1, y1, x2, y2]
        kept.append(line)
    return kept


def _draw_two_columns(
    page_size: tuple[int, int],
    min_lines: int,
    max_lines: int,
    resolver: FontResolver,
    render_profile: str,
) -> tuple[Image.Image, list[dict]]:
    """Two independently rendered columns side by side with a gutter.

    Teaches the detector that lines can end mid-page and that two lines can
    share a row, which a single paragraph never shows it.
    """
    width, height = page_size
    gutter = random.randint(max(12, width // 30), max(20, width // 14))
    col_w = (width - gutter) // 2
    left, left_lines = _draw_document((col_w, height), min_lines, max_lines, resolver, render_profile)
    right, right_lines = _draw_document((col_w, height), min_lines, max_lines, resolver, render_profile)
    page = Image.new("L", (width, height), int(np.asarray(left)[0, 0]))
    page.paste(left, (0, 0))
    x_off = col_w + gutter
    page.paste(right, (x_off, 0))
    for line in right_lines:
        x1, y1, x2, y2 = line["box"]
        line["box"] = [x1 + x_off, y1, x2 + x_off, y2]
    right_lines = _clip_lines(right_lines, width, height)
    # reading order: row by row across both columns is not what OCR of two
    # columns wants; keep column-major (all of left, then all of right)
    return page, left_lines + right_lines


def _skew_page(image: Image.Image, lines: list[dict], max_degrees: float = 1.5) -> tuple[Image.Image, list[dict]]:
    """Rotate the whole page a little, as a slightly crooked scan would be; boxes follow."""
    angle = random.uniform(-max_degrees, max_degrees)
    width, height = image.size
    background = int(np.asarray(image)[0, 0])
    rotated = image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=background)
    cx, cy = width / 2.0, height / 2.0
    theta = math.radians(-angle)  # PIL rotates counter-clockwise for positive angles, image y points down
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    for line in lines:
        x1, y1, x2, y2 = line["box"]
        xs, ys = [], []
        for px, py in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
            dx, dy = px - cx, py - cy
            xs.append(cx + dx * cos_t - dy * sin_t)
            ys.append(cy + dx * sin_t + dy * cos_t)
        line["box"] = [int(min(xs)), int(min(ys)), int(math.ceil(max(xs))), int(math.ceil(max(ys)))]
    return rotated, _clip_lines(lines, width, height)


def _save_overlay(image: Image.Image, lines: list[dict], path: Path) -> None:
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for line in lines:
        draw.rectangle(line["box"], outline=(220, 40, 40), width=2)
    overlay.save(path)


_WORKER_RESOLVER: FontResolver | None = None


def _worker_resolver() -> FontResolver:
    """One FontResolver per process (the font scan is done once, then cached)."""
    global _WORKER_RESOLVER
    if _WORKER_RESOLVER is None:
        _WORKER_RESOLVER = FontResolver()
    return _WORKER_RESOLVER


def _generate_page(
    idx: int,
    output_dir: str,
    page_size: tuple[int, int],
    min_lines: int,
    max_lines: int,
    make_line_crops: bool,
    line_crop_truncation_prob: float,
    page_crop_prob: float,
    degradation: DegradationConfig,
    preview: bool,
    seed: int | None,
    render_profile: str,
) -> tuple[dict, list[dict]]:
    """Render, degrade, and write one page plus its line crops; return manifest entries.

    Seeded per page (seed + idx) so the dataset is identical whether it was made
    by one process or eight, and any single page can be regenerated on its own.
    """
    if seed is not None:
        random.seed(seed * 1_000_003 + idx)
        np.random.seed((seed * 1_000_003 + idx) % (2**32))
    out = Path(output_dir)
    resolver = _worker_resolver()

    if random.random() < 0.15:
        clean, lines = _draw_two_columns(page_size, min_lines, max_lines, resolver, render_profile)
    else:
        clean, lines = _draw_document(page_size, min_lines, max_lines, resolver, render_profile)
    if random.random() < 0.20:
        clean, lines = _skew_page(clean, lines, 1.5)
    clean, lines = _crop_page_edges(clean, lines, page_crop_prob, resolver)
    degraded = _apply_degradation(clean, degradation)
    lines = _clip_lines(lines, *clean.size)
    page_name = f"page_{idx:06d}.png"
    page_path = out / "pages" / page_name
    degraded.save(page_path)
    rel_page = str(page_path.relative_to(out))
    page_entry = {"image": rel_page, "boxes": [l["box"] for l in lines], "lines": lines}

    if preview:
        _save_overlay(degraded, lines, out / "previews" / f"page_{idx:06d}_boxes.png")

    line_entries: list[dict] = []
    if make_line_crops:
        for line_idx, line in enumerate(lines):
            x1, y1, x2, y2 = line["box"]
            if x2 - x1 < MIN_LINE_W or y2 - y1 < MIN_LINE_H:
                continue  # belt and braces: one bad box must not kill a multi-hour job
            crop = degraded.crop((x1, y1, x2, y2))
            text = line["text"]
            if random.random() < line_crop_truncation_prob and crop.width > 40:
                cut_left = random.randint(0, max(1, int(crop.width * 0.20)))
                cut_right = crop.width - random.randint(0, max(1, int(crop.width * 0.20)))
                if cut_right - cut_left >= 20:
                    # the exact face this line was drawn with, so the visible-substring
                    # measurement matches the pixels (get() would pick a random face)
                    font = resolver.load(line.get("font_file"), line["font_size"]) if line.get("font_file") else resolver.get(
                        line["font_family"], line["font_size"], line["bold_like"], line.get("italic_like", False)
                    )
                    text = _visible_substring(text, font, cut_left, cut_right)
                    crop = crop.crop((cut_left, 0, cut_right, crop.height))
            if text:
                crop_path = out / "line_crops" / f"page_{idx:06d}_line_{line_idx:02d}.png"
                crop.save(crop_path)
                line_entries.append(
                    {"image": str(crop_path.relative_to(out)), "text": text, "source_page": rel_page, "box": line["box"]}
                )
    return page_entry, line_entries


def _generate_page_star(job, args: tuple[int, bool]) -> tuple[dict, list[dict]]:
    idx, preview = args
    return job(idx, preview=preview)


def generate_synthetic_dataset(
    output_dir: str | Path,
    num_samples: int,
    page_size: tuple[int, int] = DEFAULT_PAGE_SIZE,
    min_lines: int = 7,
    max_lines: int = 14,
    make_line_crops: bool = True,
    line_crop_truncation_prob: float = 0.20,
    page_crop_prob: float = 0.20,
    degradation: DegradationConfig | None = None,
    preview_count: int = 12,
    seed: int | None = None,
    log_every: int = 100,
    render_profile: str = "noisyoffice",
    val_split: float = 0.0,
    workers: int = 1,
) -> dict[str, Path]:
    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be in [0, 1)")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    output_dir = Path(output_dir)
    pages_dir = output_dir / "pages"
    lines_dir = output_dir / "line_crops"
    previews_dir = output_dir / "previews"
    pages_dir.mkdir(parents=True, exist_ok=True)
    lines_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    FontResolver().require_real_fonts()
    degradation = degradation or DegradationConfig()
    page_manifest: list[dict] = []
    line_manifest: list[dict] = []
    progress = ProgressLogger("synthetic generation", num_samples, log_every)

    job = functools.partial(
        _generate_page,
        output_dir=str(output_dir),
        page_size=page_size,
        min_lines=min_lines,
        max_lines=max_lines,
        make_line_crops=make_line_crops,
        line_crop_truncation_prob=line_crop_truncation_prob,
        page_crop_prob=page_crop_prob,
        degradation=degradation,
        seed=seed,
        render_profile=render_profile,
    )
    workers = max(1, int(workers))
    if workers == 1:
        results = (job(idx, preview=idx < preview_count) for idx in range(num_samples))
    else:
        pool = multiprocessing.get_context("spawn").Pool(workers)
        results = pool.imap(functools.partial(_generate_page_star, job), [(idx, idx < preview_count) for idx in range(num_samples)], chunksize=8)
    total_lines = 0
    for step, (page_entry, line_entries) in enumerate(results, start=1):
        page_manifest.append(page_entry)
        line_manifest.extend(line_entries)
        total_lines += len(page_entry["lines"])
        if progress.should_log(step):
            progress.log(step, f"pages {step} line_crops {len(line_manifest)} avg_lines {total_lines / step:.1f} loss n/a accuracy n/a")
    if workers > 1:
        pool.close()
        pool.join()

    pages_manifest_path = output_dir / "pages_manifest.json"
    lines_manifest_path = output_dir / "lines_manifest.json"
    meta_path = output_dir / "metadata.json"
    pages_manifest_path.write_text(json.dumps({"pages": page_manifest}, indent=2))
    lines_manifest_path.write_text(json.dumps({"samples": line_manifest}, indent=2))

    outputs: dict[str, Path] = {
        "pages_manifest": pages_manifest_path,
        "lines_manifest": lines_manifest_path,
    }
    # A requested split always yields at least one held-out page (given two or
    # more pages), so tiny smoke runs still produce the *_train/*_val manifests.
    num_val_pages = max(1, int(round(num_samples * val_split))) if val_split and num_samples >= 2 else 0
    if num_val_pages:
        # Split by page, not by crop: every line crop of a validation page goes to
        # validation with it, so the recognizer never trains on text it is scored on.
        # Pages are generated i.i.d., so the tail is as random as any other slice.
        val_pages = {p["image"] for p in page_manifest[-num_val_pages:]}
        splits = {
            "pages_manifest_train": ("pages", [p for p in page_manifest if p["image"] not in val_pages]),
            "pages_manifest_val": ("pages", [p for p in page_manifest if p["image"] in val_pages]),
            "lines_manifest_train": ("samples", [l for l in line_manifest if l["source_page"] not in val_pages]),
            "lines_manifest_val": ("samples", [l for l in line_manifest if l["source_page"] in val_pages]),
        }
        for name, (key, items) in splits.items():
            path = output_dir / f"{name}.json"
            path.write_text(json.dumps({key: items}, indent=2))
            outputs[name] = path

    meta_path.write_text(
        json.dumps(
            {
                "num_pages": num_samples,
                "num_line_crops": len(line_manifest),
                "page_size": page_size,
                "min_lines": min_lines,
                "max_lines": max_lines,
                "render_profile": render_profile,
                "degradation": asdict(degradation),
                "val_split": val_split,
                "num_val_pages": num_val_pages,
            },
            indent=2,
        )
    )
    outputs["metadata"] = meta_path
    outputs["previews"] = previews_dir
    return outputs
