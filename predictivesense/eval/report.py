"""JSON + human-readable markdown writers for an evaluation run.

Every run records what would be needed to reproduce it: model name and hash,
policy config, split hash, git commit, machine, seed (BLOCK 3.2.10).
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from predictivesense.eval.metrics import EvalMetrics

__all__ = ["write_reports", "run_metadata"]


def run_metadata(
    *,
    model_name: str,
    model_hash: str,
    policy: dict[str, Any],
    split: str,
    split_hash: str,
    git_commit: str,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_sha256": model_hash,
        "policy": policy,
        "split": split,
        "split_hash": split_hash,
        "git_commit": git_commit,
        "seed": seed,
        "machine": platform.platform(),
        "python": platform.python_version(),
    }
    if extra:
        meta.update(extra)
    return meta


def write_reports(
    metrics: EvalMetrics,
    metadata: dict[str, Any],
    *,
    json_path: str | Path,
    md_path: str | Path,
) -> tuple[Path, Path]:
    jp, mp = Path(json_path), Path(md_path)
    jp.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "metrics": metrics.to_dict()}
    jp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    mp.write_text(_markdown(metrics, metadata), encoding="utf-8")
    return jp, mp


def _markdown(m: EvalMetrics, meta: dict[str, Any]) -> str:
    L: list[str] = []
    L.append(f"# Detection evaluation — {meta.get('model_name', '?')} · policy "
             f"{meta.get('policy', {}).get('enabled', '?')} · split `{meta.get('split', '?')}`")
    L.append("")
    L.append(f"- model sha256: `{str(meta.get('model_sha256', ''))[:16]}…`")
    L.append(f"- split hash: `{str(meta.get('split_hash', ''))[:16]}…`  ·  seed {meta.get('seed')}")
    L.append(f"- git commit: `{str(meta.get('git_commit', ''))[:12]}`  ·  {meta.get('machine', '')}")
    L.append(f"- generated: {meta.get('generated_utc', '')}")
    L.append("")
    L.append(f"**Samples:** {m.n_images} images · {m.n_gt} ground-truth boxes · "
             f"{m.n_pred} predictions · {m.n_matched} matched pairs (IoU ≥ {m.iou_threshold}).")
    L.append("")
    L.append("## Headline — false-class rate")
    L.append("")
    L.append("A prediction that overlaps a real object (IoU ≥ %.2f) but names it "
             "*wrongly* (and not `unknown`). This is the watch → donut failure." % m.iou_threshold)
    L.append("")
    L.append(f"| metric | value | n |")
    L.append(f"|---|---|---|")
    L.append(f"| **false-class rate** | **{m.false_class_rate:.4f}** | {m.false_class_count} / {m.n_matched} |")
    L.append(f"| unknown-on-object rate | {m.unknown_on_object_rate:.4f} | {m.unknown_on_object_count} / {m.n_matched} |")
    L.append(f"| localization recall (any class) | {m.localization_recall:.4f} | {m.n_matched} / {m.n_gt} |")
    L.append(f"| background false-positive rate | {m.background_fp_rate:.4f} | {m.background_fp_count} / {m.n_pred} |")
    L.append(f"| mAP@{m.iou_threshold} | {'—' if m.map50 is None else f'{m.map50:.4f}'} | classes w/ support |")
    L.append(f"| macro F1 (support > 0) | {m.macro_f1():.4f} | — |")
    L.append("")
    L.append("## Per class")
    L.append("")
    L.append("| class | support | TP | FP | FN | precision | recall | F1 | AP@%.2f |" % m.iou_threshold)
    L.append("|---|---|---|---|---|---|---|---|---|")
    for c in m.per_class:
        ap = "—" if c.ap50 is None else f"{c.ap50:.3f}"
        L.append(f"| {c.name} | {c.support} | {c.tp} | {c.fp} | {c.fn} | "
                 f"{c.precision:.3f} | {c.recall:.3f} | {c.f1:.3f} | {ap} |")
    L.append("")
    L.append("## Confusion matrix (rows = ground truth, cols = predicted)")
    L.append("")
    header = "| gt \\ pred | " + " | ".join(m.confusion_labels) + " |"
    L.append(header)
    L.append("|" + "---|" * (len(m.confusion_labels) + 1))
    for lab, row in zip(m.confusion_labels, m.confusion):
        L.append(f"| **{lab}** | " + " | ".join(str(v) for v in row) + " |")
    L.append("")
    L.append("## Most frequent confusions")
    L.append("")
    if m.top_confusions:
        L.append("| ground truth | predicted as | count | example image ids |")
        L.append("|---|---|---|---|")
        for row in m.top_confusions:
            ex = ", ".join(str(i) for i in row["example_image_ids"])
            L.append(f"| {row['ground_truth']} | {row['predicted_as']} | {row['count']} | {ex} |")
    else:
        L.append("_none_")
    L.append("")
    return "\n".join(L) + "\n"
