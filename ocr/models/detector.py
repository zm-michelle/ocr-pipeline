from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ocr.models.blocks import ConvBNAct, DepthwiseSeparableBlock


class DBNet(nn.Module):
    """A compact DBNet-style text segmentation model.

    It predicts a single text probability/logit map. The postprocessor in
    ocr/inference/detection.py turns connected text regions into sorted line crops.
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
            nn.Conv2d(inner_channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
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
        logits = self.fuse(torch.cat(feats, dim=1))
        return F.interpolate(logits, input_size, mode="bilinear", align_corners=False)
