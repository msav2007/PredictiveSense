"""Real training run: Studio samples -> validated splits -> trained artifact
-> ONNX export -> registry entry (Phase 11 Part B section 12).

No code editing between runs (section 12.1): every parameter is a function
argument / CLI flag. Every run records a full manifest and is versioned, never
overwriting a previous run (section 12.2/12.3).
"""

from __future__ import annotations

import hashlib
import platform
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader

from predictivesense.training.classifier_registry import ClassifierRegistry, ClassifierVersion
from predictivesense.training.crop_data import CropDataset
from predictivesense.training.dataset import (
    CropRecord,
    class_map_for,
    positive_records,
    rejection_records,
    validate_dataset,
)
from predictivesense.training.evaluate import evaluate_predictor, predictor_from_torch_model
from predictivesense.training.model import IMAGE_SIZE, CropClassifier
from predictivesense.training.splits import build_crop_splits
from predictivesense.telemetry.manifest import utc_now_iso

__all__ = ["TrainConfig", "TrainRunResult", "run_training"]

_ARCHITECTURE_NAME = "CropClassifier-3conv-gap"


@dataclass(frozen=True)
class TrainConfig:
    objects_root: str
    results_dir: str = "results"
    models_dir: str = "models/custom"
    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 1e-3
    seed: int = 0
    train_fraction: float = 0.6
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    min_samples_per_class: int = 4
    # Phase 12 section 9.1/9.2: imported external samples, combined with
    # (never replacing) objects_root's own Studio samples.
    external_manifests: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainRunResult:
    version_id: str
    manifest_path: str
    onnx_path: str
    registry_entry: ClassifierVersion
    test_metrics: dict[str, Any]


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _machine_fingerprint() -> str:
    return (
        f"{platform.platform()} | {platform.processor() or platform.machine()} | "
        f"{psutil.cpu_count(logical=True)} logical cores | Python {platform.python_version()}"
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_one_epoch(
    model: nn.Module, loader: DataLoader, *, optimizer: torch.optim.Optimizer | None
) -> tuple[float, float]:
    """Returns (mean_loss, accuracy). ``optimizer=None`` -> eval mode, no grad."""

    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    loss_fn = nn.CrossEntropyLoss()
    for images, labels in loader:
        if training:
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = loss_fn(logits, labels)
        total_loss += float(loss.item()) * images.size(0)
        correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
        n += images.size(0)
    return (total_loss / n if n else 0.0), (correct / n if n else 0.0)


def _export_onnx(model: nn.Module, path: Path, *, image_size: int) -> None:
    model.eval()
    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["crop"], output_names=["logits"],
        dynamic_axes={"crop": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        # dynamo=False (legacy TorchScript-based exporter, matches
        # dynamic_axes above per torch's own docs): the newer dynamo=True
        # exporter prints a unicode success banner that crashes on a cp1252
        # Windows console, and is unnecessary complexity for this small,
        # static-shaped CNN.
        dynamo=False,
    )


def run_training(config: TrainConfig, *, registry: ClassifierRegistry | None = None) -> TrainRunResult:
    """Studio samples (``config.objects_root``) -> a registered, unvalidated,
    inactive classifier version. Raises :class:`~predictivesense.training.dataset.DatasetError`
    if there is not enough data to train meaningfully (section 11.4)."""

    _seed_everything(config.seed)
    wall0 = time.monotonic()

    records, summary = validate_dataset(
        config.objects_root,
        min_samples_per_class=config.min_samples_per_class,
        external_manifests=config.external_manifests,
    )
    class_map = class_map_for(records)
    splits = build_crop_splits(
        records, train_fraction=config.train_fraction, val_fraction=config.val_fraction,
        test_fraction=config.test_fraction, seed=config.seed,
    )

    by_id: dict[str, CropRecord] = {r.crop_id: r for r in records}

    def _select(split_name: str) -> list[CropRecord]:
        return [by_id[cid] for cid in splits.crop_ids[split_name]]

    train_positive = positive_records(_select("train"))
    val_positive = positive_records(_select("val"))
    test_positive = positive_records(_select("test"))
    test_rejection = rejection_records(_select("test"))

    if not train_positive:
        from predictivesense.training.dataset import DatasetError

        raise DatasetError("no positive training crops after splitting - collect more samples")

    train_ds = CropDataset(train_positive, class_map, augment=True, seed=config.seed)
    val_ds = CropDataset(val_positive, class_map, augment=False) if val_positive else None
    g = torch.Generator()
    g.manual_seed(config.seed)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, generator=g, num_workers=0
    )
    val_loader = (
        DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
        if val_ds is not None else None
    )

    model = CropClassifier(num_classes=len(class_map))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    epoch_metrics: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        train_loss, train_acc = _run_one_epoch(model, train_loader, optimizer=optimizer)
        if val_loader is not None:
            val_loss, val_acc = _run_one_epoch(model, val_loader, optimizer=None)
        else:
            val_loss, val_acc = -1.0, -1.0
        epoch_metrics.append({
            "epoch": epoch, "train_loss": round(train_loss, 5), "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 5), "val_accuracy": round(val_acc, 4),
        })

    predict = predictor_from_torch_model(model, class_map)
    test_metrics = evaluate_predictor(predict, test_positive, test_rejection, sorted(class_map))

    version_id = f"crop-clf-{utc_now_iso().replace(':', '').replace('.', '')[:15]}-{uuid.uuid4().hex[:8]}"
    models_dir = Path(config.models_dir) / version_id
    onnx_path = models_dir / "model.onnx"
    _export_onnx(model, onnx_path, image_size=IMAGE_SIZE)
    artifact_sha256 = _sha256_file(onnx_path)

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version_id": version_id,
        "created_utc": utc_now_iso(),
        "class_map": class_map,
        "dataset_summary": summary.to_dict(),
        "dataset_content_hash": splits.content_hash,
        "split_content_hash": splits.content_hash,
        "split_fractions": splits.fractions,
        "architecture": _ARCHITECTURE_NAME,
        "image_size": IMAGE_SIZE,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "augmentation": "horizontal_flip_p0.5 + brightness_jitter[0.85,1.15]",
        "epoch_metrics": epoch_metrics,
        "test_metrics": test_metrics.to_dict(),
        "wall_seconds": round(time.monotonic() - wall0, 3),
        "machine_fingerprint": _machine_fingerprint(),
        "provider": "cpu (training always runs in PyTorch on CPU; serving is ONNX Runtime, see perception/classifier.py)",
        "artifact_path": str(onnx_path),
        "artifact_sha256": artifact_sha256,
    }
    manifest_path = results_dir / f"train_run_{version_id}.json"
    import json

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    entry = ClassifierVersion(
        version_id=version_id, created_utc=manifest["created_utc"], class_map=class_map,
        dataset_content_hash=splits.content_hash, split_content_hash=splits.content_hash,
        architecture=_ARCHITECTURE_NAME, image_size=IMAGE_SIZE, epochs=config.epochs,
        batch_size=config.batch_size, learning_rate=config.learning_rate, seed=config.seed,
        augmentation=manifest["augmentation"], metrics=manifest["test_metrics"],
        artifact_path=str(onnx_path), artifact_sha256=artifact_sha256,
        wall_seconds=manifest["wall_seconds"], machine_fingerprint=manifest["machine_fingerprint"],
        provider=manifest["provider"],
        dataset_human_confirmed_fraction=summary.human_confirmed_fraction,
        notes=(
            f"trained from {config.objects_root}"
            + (f" + external: {list(config.external_manifests)}" if config.external_manifests else "")
        ),
    )
    reg = registry if registry is not None else ClassifierRegistry()
    registered = reg.register(entry)

    return TrainRunResult(
        version_id=version_id, manifest_path=str(manifest_path), onnx_path=str(onnx_path),
        registry_entry=registered, test_metrics=test_metrics.to_dict(),
    )
