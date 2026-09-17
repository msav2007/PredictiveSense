"""The crop classifier architecture (Phase 11 Part B section 10.1).

A small, from-scratch CNN, not a torchvision backbone: at the collection
scale this dataset actually reaches (tens of images per class), a compact
network trains in seconds on CPU and does not require downloading pretrained
ImageNet weights (no extra network dependency for a training run, and no
question of which pretrained-weights licence applies). Swappable later if a
larger, pretrained backbone is ever justified by measurement.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["CropClassifier", "IMAGE_SIZE"]

IMAGE_SIZE = 64  # matches predictivesense/training/synthetic_fixture.py


class CropClassifier(nn.Module):
    """Three conv blocks + global average pool + a linear head."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
