"""Small cross-cutting helpers: device selection, seeding, progress logging."""

from __future__ import annotations

import random
import time

import numpy as np
import torch


def resolve_device(value: torch.device | str = "auto") -> torch.device:
    """Turn "auto"/"cuda"/"cpu"/"mps" into a concrete device.

    "auto" stays on cuda-or-cpu; ask for "mps" explicitly, since not every op the
    models use is guaranteed to be implemented on the Metal backend.
    """
    if isinstance(value, torch.device):
        return value
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ProgressLogger:
    def __init__(self, name: str, total: int, log_every: int = 50) -> None:
        self.name = name
        self.total = max(1, total)
        self.log_every = max(1, log_every)
        self.start = time.time()

    def eta(self, step: int) -> str:
        elapsed = time.time() - self.start
        rate = step / max(elapsed, 1e-6)
        remaining = max(0.0, (self.total - step) / max(rate, 1e-6))
        minutes, seconds = divmod(int(remaining), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}h{minutes:02d}m"
        return f"{minutes:02d}m{seconds:02d}s"

    def should_log(self, step: int) -> bool:
        return step == 1 or step == self.total or step % self.log_every == 0

    def log(self, step: int, message: str) -> None:
        print(f"{self.name} {step}/{self.total} eta {self.eta(step)} | {message}")
