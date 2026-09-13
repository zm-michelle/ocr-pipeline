"""Live training metrics: TensorBoard event files plus a plain metrics.jsonl.

Every run gets its own directory under --log_dir. Point TensorBoard at the parent
to compare runs side by side:

    tensorboard --logdir runs

The JSONL mirror means the numbers are still readable without TensorBoard, and
lets the web app or a notebook chart them later.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard is optional; the JSONL still gets written
    SummaryWriter = None  # type: ignore[assignment,misc]


def default_run_name(mode: str) -> str:
    return f"{mode}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"


class RunTracker:
    """Fan metrics out to TensorBoard and metrics.jsonl; a no-op when disabled."""

    def __init__(self, log_dir: str | Path | None, run_name: str, enabled: bool = True) -> None:
        self.enabled = bool(enabled and log_dir)
        self.run_dir: Path | None = None
        self._writer = None
        self._jsonl = None
        self._started = time.time()
        if not self.enabled:
            return

        self.run_dir = Path(log_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.run_dir / "metrics.jsonl").open("a")
        if SummaryWriter is not None:
            self._writer = SummaryWriter(log_dir=str(self.run_dir))

    @property
    def tensorboard_available(self) -> bool:
        return self._writer is not None

    # -- scalars ---------------------------------------------------------

    def scalar(self, tag: str, value: float, step: int) -> None:
        if not self.enabled:
            return
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)
        self._jsonl.write(json.dumps({"t": round(time.time() - self._started, 2), "step": step, tag: float(value)}) + "\n")

    def scalars(self, values: dict[str, float], step: int, prefix: str = "") -> None:
        if not self.enabled:
            return
        for name, value in values.items():
            if self._writer is not None:
                self._writer.add_scalar(f"{prefix}{name}", value, step)
        record: dict[str, Any] = {"t": round(time.time() - self._started, 2), "step": step}
        record.update({f"{prefix}{k}": float(v) for k, v in values.items()})
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()

    # -- richer views ----------------------------------------------------

    def text(self, tag: str, markdown: str, step: int) -> None:
        if self._writer is not None:
            self._writer.add_text(tag, markdown, step)

    def image_grid(self, tag: str, images: torch.Tensor, step: int) -> None:
        """images: [N, C, H, W] in [0, 1], stacked vertically into one image.

        Vertical because each row is already a wide input|target|prediction
        strip; tiling rows side by side would make a 6000px-wide sliver.
        """
        if self._writer is not None:
            stacked = torch.cat(list(images.clamp(0, 1)), dim=-2)  # [C, N*H, W]
            self._writer.add_image(tag, stacked, step)

    def hparams(self, params: dict[str, Any]) -> None:
        if not self.enabled:
            return
        (self.run_dir / "hparams.json").write_text(json.dumps(params, indent=2, default=str))
        if self._writer is not None:
            lines = "\n".join(f"| {k} | `{v}` |" for k, v in sorted(params.items()))
            self._writer.add_text("hparams", "| key | value |\n| --- | --- |\n" + lines, 0)

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()
        if self._jsonl is not None:
            self._jsonl.flush()

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
        if self._jsonl is not None:
            self._jsonl.close()

    def __enter__(self) -> RunTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
