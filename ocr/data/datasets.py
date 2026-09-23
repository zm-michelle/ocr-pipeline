from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from ocr.config import DEFAULT_CHARSET, DEFAULT_DETECTOR_SIZE, IMAGE_EXTENSIONS
from ocr.ctc import encode_text
from ocr.data.targets import DEFAULT_SHRINK_RATIO, build_db_targets
from ocr.data.transforms import DetectionTransform, RecognitionTransform


def list_image_files(root: str | Path, extensions: tuple[str, ...] = IMAGE_EXTENSIONS) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in extensions)


def load_manifest_spec(spec: str | Path, default_root: Path | None) -> list[dict[str, Any]]:
    """Comma-separated manifests, each optionally `:N` (repeat N times), `:0.x` (random subset) and `@root`.

        synthetic/lines_manifest_train.json,real/lines_manifest_train.json:8

    Every entry gets an absolute "image" path resolved against its own root
    (the manifest's directory unless given), so datasets from different places
    can be mixed in one loader. Repeating is how a small real set gets weight
    against a large synthetic one without a sampler.
    """
    merged: list[dict[str, Any]] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        repeat = 1
        root: Path | None = None
        if "@" in part:
            part, root_str = part.rsplit("@", 1)
            root = Path(root_str)
        fraction = 1.0
        if ":" in part:
            head, tail = part.rsplit(":", 1)
            if tail.isdigit():
                part, repeat = head, int(tail)
            elif tail.replace(".", "", 1).isdigit() and float(tail) < 1.0:
                part, fraction = head, float(tail)  # `:0.1` = a fixed random 10% subset
        path = Path(part)
        base = root or (default_root if default_root and len(str(spec).split(",")) == 1 else path.parent)
        resolved: list[dict[str, Any]] = []
        for item in _load_manifest(path):
            image = Path(item.get("image") or item.get("image_path"))
            if not image.is_absolute():
                item = {**item, "image": str((base / image).resolve())}
            resolved.append(item)
        if fraction < 1.0:
            rng = random.Random(0)
            resolved = rng.sample(resolved, max(1, int(len(resolved) * fraction)))
        merged.extend(resolved * repeat)
    return merged


def _load_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return data.get("samples") or data.get("pages") or data.get("items") or []
        return data
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported manifest format: {path}")


class PrintedLineDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        root_dir: str | Path | None = None,
        charset: str = DEFAULT_CHARSET,
        transform: RecognitionTransform | None = None,
    ) -> None:
        self.manifest_path = Path(str(manifest_path).split(",")[0])
        self.root_dir = Path(root_dir) if root_dir else self.manifest_path.parent
        self.samples = load_manifest_spec(manifest_path, self.root_dir)
        self.charset = charset
        self.transform = transform or RecognitionTransform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        image_path = Path(item.get("image") or item.get("image_path"))
        if not image_path.is_absolute():
            image_path = self.root_dir / image_path
        text = str(item.get("text") or item.get("transcript") or "")

        image = Image.open(image_path).convert("L")
        target = torch.tensor(encode_text(text, self.charset), dtype=torch.long)
        return {
            "image": self.transform(image),
            "target": target,
            "target_length": torch.tensor(len(target), dtype=torch.long),
            "text": text,
            "path": str(image_path),
        }


def recognition_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = [item["target"] for item in batch]
    target_lengths = torch.stack([item["target_length"] for item in batch])
    if targets:
        flat_targets = torch.cat(targets) if sum(len(t) for t in targets) else torch.empty(0, dtype=torch.long)
    else:
        flat_targets = torch.empty(0, dtype=torch.long)
    return {
        "images": images,
        "targets": flat_targets,
        "target_lengths": target_lengths,
        "texts": [item["text"] for item in batch],
        "paths": [item["path"] for item in batch],
    }


class DetectionDataset(Dataset):
    """Pages + line boxes -> detector input and DBNet-style targets.

    Each item carries, all at `image_size`:
      image        [1, H, W] normalized page
      mask         [1, H, W] probability-map target (boxes shrunk by shrink_ratio)
      thresh_map   [1, H, W] threshold-map target in [0.3, 0.7]
      thresh_mask  [1, H, W] where the threshold target applies (border bands)
      boxes        the line boxes scaled to image_size, for box-level scoring
    shrink_ratio=0 gives the plain full-box mask and an empty threshold band.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        root_dir: str | Path | None = None,
        image_size: tuple[int, int] = DEFAULT_DETECTOR_SIZE,
        transform: DetectionTransform | None = None,
        shrink_ratio: float = DEFAULT_SHRINK_RATIO,
    ) -> None:
        self.manifest_path = Path(str(manifest_path).split(",")[0])
        self.root_dir = Path(root_dir) if root_dir else self.manifest_path.parent
        self.samples = load_manifest_spec(manifest_path, self.root_dir)
        self.image_size = image_size
        self.transform = transform or DetectionTransform(size=image_size)
        self.shrink_ratio = shrink_ratio

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        image_path = Path(item.get("image") or item.get("image_path"))
        if not image_path.is_absolute():
            image_path = self.root_dir / image_path
        boxes = item.get("boxes") or [line["box"] for line in item.get("lines", [])]

        image = Image.open(image_path).convert("L")
        sx = self.image_size[0] / image.width
        sy = self.image_size[1] / image.height
        scaled = [
            (int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy)))
            for x1, y1, x2, y2 in boxes
        ]
        gt, thresh_map, thresh_mask = build_db_targets(scaled, self.image_size, self.shrink_ratio)
        return {
            "image": self.transform(image),
            "mask": torch.from_numpy(gt).unsqueeze(0),
            "thresh_map": torch.from_numpy(thresh_map).unsqueeze(0),
            "thresh_mask": torch.from_numpy(thresh_mask).unsqueeze(0),
            "boxes": scaled,
            "path": str(image_path),
        }


def detection_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "thresh_map": torch.stack([item["thresh_map"] for item in batch], dim=0),
        "thresh_mask": torch.stack([item["thresh_mask"] for item in batch], dim=0),
        "boxes": [item["boxes"] for item in batch],
        "paths": [item["path"] for item in batch],
    }


class OCRFolderDataset(Dataset):
    def __init__(self, input_dir: str | Path, extensions: tuple[str, ...] = IMAGE_EXTENSIONS) -> None:
        self.input_dir = Path(input_dir)
        self.paths = list_image_files(self.input_dir, extensions)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]
        return {"image": Image.open(path).convert("L"), "path": str(path)}
