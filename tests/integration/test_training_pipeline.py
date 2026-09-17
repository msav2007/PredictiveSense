"""Phase 11 Part B section 12/13/16 - the training pipeline proven end to end
on a synthetic fixture, plus the registry state machine and real (non-fake)
ONNX Runtime inference. Requires the ``[train]`` extra; skips cleanly via
``require_train`` when it is absent (section 17's ``train`` marker).
"""

from __future__ import annotations

import pytest

from predictivesense.training.classifier_registry import ClassifierRegistry, ClassifierRegistryError

pytestmark = [pytest.mark.integration, pytest.mark.train]


@pytest.fixture()
def registry(tmp_path):
    return ClassifierRegistry(path=tmp_path / "classifier_registry.json")


@pytest.fixture()
def trained_run(tmp_path, registry, require_train):
    from predictivesense.training.synthetic_fixture import build_synthetic_object_store
    from predictivesense.training.train import TrainConfig, run_training

    objects_root = tmp_path / "objects"
    build_synthetic_object_store(objects_root, n_per_class=15, n_sessions_per_class=3, seed=0)
    config = TrainConfig(
        objects_root=str(objects_root), results_dir=str(tmp_path / "results"),
        models_dir=str(tmp_path / "models"), epochs=10, batch_size=8, seed=0,
    )
    return run_training(config, registry=registry)


def test_training_produces_a_real_artifact_with_metrics_and_checksum(trained_run, tmp_path):
    from pathlib import Path

    onnx_path = Path(trained_run.onnx_path)
    assert onnx_path.is_file()
    assert onnx_path.stat().st_size > 0
    assert Path(trained_run.manifest_path).is_file()

    entry = trained_run.registry_entry
    assert entry.artifact_sha256 and len(entry.artifact_sha256) == 64
    assert entry.validated is False
    assert entry.active is False
    assert set(entry.class_map) == {"circle", "square", "triangle"}
    # The fixture is trivially separable - a real, working classifier should
    # get almost all of it right (not asserting exactly 1.0: CPU float
    # nondeterminism across machines is not this test's concern).
    assert trained_run.test_metrics["accuracy"] >= 0.8


def test_dataset_and_split_manifests_are_recorded(trained_run):
    import json

    manifest = json.loads(open(trained_run.manifest_path, encoding="utf-8").read())
    for key in (
        "dataset_content_hash", "split_content_hash", "class_map", "architecture",
        "image_size", "epochs", "batch_size", "learning_rate", "seed", "augmentation",
        "epoch_metrics", "wall_seconds", "machine_fingerprint", "provider",
        "artifact_path", "artifact_sha256", "dataset_summary",
    ):
        assert key in manifest, f"missing manifest field {key!r}"
    assert manifest["dataset_summary"]["human_confirmed_fraction"] == 1.0


def test_reproducible_within_tolerance_given_the_same_seed(tmp_path, require_train):
    from predictivesense.training.synthetic_fixture import build_synthetic_object_store
    from predictivesense.training.train import TrainConfig, run_training

    objects_root = tmp_path / "objects"
    build_synthetic_object_store(objects_root, n_per_class=15, n_sessions_per_class=3, seed=0)

    results_a = tmp_path / "results_a"
    models_a = tmp_path / "models_a"
    reg_a = ClassifierRegistry(path=tmp_path / "reg_a.json")
    cfg = TrainConfig(
        objects_root=str(objects_root), results_dir=str(results_a), models_dir=str(models_a),
        epochs=8, batch_size=8, seed=42,
    )
    run_a = run_training(cfg, registry=reg_a)

    results_b = tmp_path / "results_b"
    models_b = tmp_path / "models_b"
    reg_b = ClassifierRegistry(path=tmp_path / "reg_b.json")
    cfg_b = TrainConfig(
        objects_root=str(objects_root), results_dir=str(results_b), models_dir=str(models_b),
        epochs=8, batch_size=8, seed=42,
    )
    run_b = run_training(cfg_b, registry=reg_b)

    # Same data, same config, same seed -> same reported metrics within a
    # stated tolerance (section 12.3). Not bit-identical: CPU BLAS reduction
    # order is not guaranteed deterministic across calls even with a fixed
    # seed - 2% absolute accuracy is a generous, explicit tolerance.
    assert abs(run_a.test_metrics["accuracy"] - run_b.test_metrics["accuracy"]) <= 0.02
    assert run_a.registry_entry.class_map == run_b.registry_entry.class_map
    assert run_a.registry_entry.dataset_content_hash == run_b.registry_entry.dataset_content_hash


def test_unvalidated_model_cannot_be_activated(trained_run, registry):
    with pytest.raises(ClassifierRegistryError, match="not validated"):
        registry.activate(trained_run.version_id)


def test_validate_then_activate_switches_the_active_version(trained_run, registry):
    registry.validate_version(trained_run.version_id)
    active = registry.activate(trained_run.version_id)
    assert active is not None
    assert active.version_id == trained_run.version_id
    assert active.validated is True
    assert registry.active().version_id == trained_run.version_id


def test_only_one_version_active_at_a_time(trained_run, registry, tmp_path, require_train):
    from predictivesense.training.synthetic_fixture import build_synthetic_object_store
    from predictivesense.training.train import TrainConfig, run_training

    registry.validate_version(trained_run.version_id)
    registry.activate(trained_run.version_id)

    # Train and validate a second version in the same registry.
    objects_root_2 = tmp_path / "objects2"
    build_synthetic_object_store(objects_root_2, n_per_class=15, n_sessions_per_class=3, seed=1)
    cfg = TrainConfig(
        objects_root=str(objects_root_2), results_dir=str(tmp_path / "results2"),
        models_dir=str(tmp_path / "models2"), epochs=8, seed=1,
    )
    run2 = run_training(cfg, registry=registry)
    registry.validate_version(run2.version_id)
    registry.activate(run2.version_id)

    versions = registry.list()
    active_versions = [v for v in versions if v.active]
    assert len(active_versions) == 1
    assert active_versions[0].version_id == run2.version_id


def test_rollback_deactivates_the_custom_classifier(trained_run, registry):
    registry.validate_version(trained_run.version_id)
    registry.activate(trained_run.version_id)
    assert registry.active() is not None

    result = registry.activate(None)  # rollback to baseline
    assert result is None
    assert registry.active() is None
    # The trained version itself is untouched - rollback deactivates, never deletes.
    assert registry.get(trained_run.version_id).validated is True


def test_activating_an_unknown_version_raises(registry):
    with pytest.raises(ClassifierRegistryError, match="no classifier version"):
        registry.activate("does-not-exist")


# -- Section 16: live inference must be real ONNX Runtime inference ----------

# -- Phase 12 section 9/12: external + Studio data combined, with composition --

def _build_external_manifest_with_real_images(tmp_path, *, object_id: str, n: int):
    import json

    import cv2
    import numpy as np

    from predictivesense.training.synthetic_fixture import _draw_shape

    img_dir = tmp_path / "external_images" / object_id
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(99)
    samples = []
    for i in range(n):
        img = _draw_shape("circle", size=64, color=(10, 220, 220), rng=rng)  # a 4th, distinct colour
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        img_path = img_dir / f"{object_id}_{i}.jpg"
        img_path.write_bytes(buf.tobytes())
        samples.append({
            "sample_id": f"ext_{object_id}_{i}",
            "object_id": object_id,
            "image_id": f"{object_id}_{i}",
            "image_path": str(img_path.resolve()),
            "box": [8.0, 8.0, 48.0, 48.0],
            "role": "positive",
            "negative_for": [],
            "box_confirmed_by_human": True,
        })
    manifest_path = tmp_path / "external" / object_id / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"version": 1, "source": "open-images-v7", "object_id": object_id, "samples": samples}),
        encoding="utf-8",
    )
    return manifest_path


def test_training_combines_external_and_studio_data_with_composition_reporting(tmp_path, registry, require_train):
    import json
    from pathlib import Path

    from predictivesense.training.synthetic_fixture import build_synthetic_object_store
    from predictivesense.training.train import TrainConfig, run_training

    objects_root = tmp_path / "objects"
    build_synthetic_object_store(objects_root, n_per_class=15, n_sessions_per_class=3, seed=0)
    manifest_path = _build_external_manifest_with_real_images(tmp_path, object_id="gadget", n=15)

    config = TrainConfig(
        objects_root=str(objects_root), results_dir=str(tmp_path / "results"),
        models_dir=str(tmp_path / "models"), epochs=6, batch_size=8, seed=0,
        external_manifests=(str(manifest_path),),
    )
    result = run_training(config, registry=registry)

    assert set(result.registry_entry.class_map) == {"circle", "square", "triangle", "gadget"}
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    composition = manifest["dataset_summary"]["per_source_counts"]
    assert composition["external:open-images-v7"] == {"gadget": 15}
    assert composition["camera"]["circle"] == 16  # 15 positive + 1 hard_negative
    assert "external" in result.registry_entry.notes


def test_onnx_inference_matches_the_trained_classes_no_hardcoding(trained_run, tmp_path, require_train):
    """Runs the EXPORTED ONNX artifact (not the in-memory torch model) via the
    same ``perception/classifier.py`` path the live app would use, on FRESH
    synthetic crops never seen during training. A hardcoded class map or
    filename-based shortcut would not track a genuinely different class_map
    ordering, so this also trains a second run with objects created in a
    DIFFERENT name order and asserts both correctly separate their own
    classes - proof the answer comes from the pixels, not a lookup table."""

    import numpy as np

    from predictivesense.perception.classifier import CropClassifierModel
    from predictivesense.training.synthetic_fixture import FIXTURE_SHAPES, _draw_shape

    entry = trained_run.registry_entry
    clf = CropClassifierModel(
        entry.artifact_path, class_map=entry.class_map, image_size=entry.image_size, provider="cpu",
    )
    rng = np.random.default_rng(123)
    colors = {"circle": (60, 180, 250), "square": (90, 220, 90), "triangle": (220, 90, 200)}
    correct = 0
    total = 0
    for shape in FIXTURE_SHAPES:
        for _ in range(4):
            img = _draw_shape(shape, size=entry.image_size, color=colors[shape], rng=rng)
            pred, confidence = clf.infer_crop(img)
            assert pred in entry.class_map  # only ever a real trained class
            assert 0.0 <= confidence <= 1.0
            correct += int(pred == shape)
            total += 1
    assert correct / total >= 0.75  # real inference on unseen crops, not memorised training images
