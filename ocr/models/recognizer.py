from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ocr.config import DEFAULT_CHARSET
from ocr.models.blocks import ConvBNAct


class CRNNRecognizer(nn.Module):
    """CNN feature extractor + BiLSTM + linear CTC head over a fixed charset."""

    def __init__(
        self,
        num_classes: int | None = None,
        charset: str = DEFAULT_CHARSET,
        hidden_size: int = 128,
        lstm_layers: int = 2,
    ) -> None:
        super().__init__()
        self.charset = charset
        self.num_classes = num_classes or (len(charset) + 1)

        self.cnn = nn.Sequential(
            ConvBNAct(1, 32, 3, 1),
            nn.MaxPool2d(2, 2),
            ConvBNAct(32, 64, 3, 1),
            nn.MaxPool2d(2, 2),
            ConvBNAct(64, 128, 3, 1),
            ConvBNAct(128, 128, 3, 1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            ConvBNAct(128, 256, 3, 1),
            ConvBNAct(256, 256, 3, 1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            ConvBNAct(256, 256, 3, 1),
        )
        self.sequence = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=False,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        features = F.adaptive_avg_pool2d(features, (1, features.shape[-1])).squeeze(2)
        sequence = features.permute(2, 0, 1).contiguous()
        sequence, _ = self.sequence(sequence)
        return self.classifier(sequence)
