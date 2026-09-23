from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ocr.models.blocks import ConvBNAct, DepthwiseSeparableBlock


class DBNet(nn.Module):
    """A compact DBNet-style text segmentation model.

    `forward` returns the text probability logit map, which is all inference
    needs; the postprocessor in ocr/inference/detection.py turns it into sorted
    line boxes. `forward_maps` additionally returns the threshold logit map used
    by the differentiable-binarization loss during training.

    The threshold head is a separate module so checkpoints from before it
    existed still load (strict=False leaves it at its init); the probability
    head keeps its original parameter names inside `fuse` for the same reason.
    """

    def __init__(self, in_channels: int = 1, inner_channels: int = 64) -> None:
        super().__init__()
        self.stem = ConvBNAct(in_channels, 16, 3, 2)
        self.stage2 = DepthwiseSeparableBlock(16, 24, 2)
        self.stage3 = DepthwiseSeparableBlock(24, 40, 2)
        self.stage4 = DepthwiseSeparableBlock(40, 80, 2)

        self.lat1 = nn.Conv2d(16, inner_channels, 1)
        self.lat2 = nn.Conv2d(24, inner_channels, 1)
        self.lat3 = nn.Conv2d(40, inner_channels, 1)
        self.lat4 = nn.Conv2d(80, inner_channels, 1)

        self.fuse = nn.Sequential(
            ConvBNAct(inner_channels * 4, inner_channels, 3),
            DepthwiseSeparableBlock(inner_channels, inner_channels),
            nn.Conv2d(inner_channels, 1, 1),  # probability head
        )
        self.thresh_head = nn.Conv2d(inner_channels, 1, 1)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.stem(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        target = c1.shape[-2:]
        feats = [
            self.lat1(c1),
            F.interpolate(self.lat2(c2), target, mode="bilinear", align_corners=False),
            F.interpolate(self.lat3(c3), target, mode="bilinear", align_corners=False),
            F.interpolate(self.lat4(c4), target, mode="bilinear", align_corners=False),
        ]
        fused = torch.cat(feats, dim=1)
        return self.fuse[1](self.fuse[0](fused))  # everything up to the heads

    def forward_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(probability logits, threshold logits), both [B, 1, H, W] at input size."""
        input_size = x.shape[-2:]
        shared = self._features(x)
        prob = F.interpolate(self.fuse[2](shared), input_size, mode="bilinear", align_corners=False)
        thresh = F.interpolate(self.thresh_head(shared), input_size, mode="bilinear", align_corners=False)
        return prob, thresh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        logits = self.fuse[2](self._features(x))
        return F.interpolate(logits, input_size, mode="bilinear", align_corners=False)
