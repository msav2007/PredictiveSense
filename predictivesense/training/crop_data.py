"""Loads crop pixels for training/evaluation - the one place that turns a
:class:`~predictivesense.training.dataset.CropRecord` (a box reference into an
existing Studio sample image) into an actual tensor. Kept separate from
``dataset.py`` so that module stays torch-free (import cost / layering)."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from predictivesense.training.dataset import CropRecord
from predictivesense.training.model import IMAGE_SIZE

__all__ = ["CropDataset", "load_crop_tensor"]


def load_crop_tensor(record: CropRecord, *, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """Read the sample's own image, crop its box, resize, normalise to
    [0, 1] CHW float32 - no augmentation (used for eval / inference)."""

    image = cv2.imread(record.image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read {record.image_path}")
    x, y, w, h = record.box
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(image.shape[1], int(round(x + w))), min(image.shape[0], int(round(y + h)))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        crop = image  # degenerate box: fall back to the whole image rather than crash
    crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1)
    return tensor


class CropDataset(Dataset):
    """One split's crops, as ``(image_tensor, class_index)`` pairs.

    ``augment=True`` (train split only) applies a horizontal flip and small
    brightness jitter - cheap, label-preserving augmentation for a dataset
    this small. Never applied to val/test (section 15's comparison must be on
    unaltered held-out data)."""

    def __init__(
        self, records: list[CropRecord], class_map: dict[str, int], *,
        image_size: int = IMAGE_SIZE, augment: bool = False, seed: int = 0,
    ) -> None:
        self._records = records
        self._class_map = class_map
        self._image_size = image_size
        self._augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        record = self._records[idx]
        tensor = load_crop_tensor(record, image_size=self._image_size)
        if self._augment:
            if self._rng.random() < 0.5:
                tensor = torch.flip(tensor, dims=[2])
            jitter = float(self._rng.uniform(0.85, 1.15))
            tensor = torch.clamp(tensor * jitter, 0.0, 1.0)
        return tensor, self._class_map[record.object_id]
