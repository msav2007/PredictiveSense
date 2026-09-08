"""Recognition policy layer (Phase 2.5).

Sits **after** the detector and **before** the snapshot. The detector is not
modified; this is a separate, independently-switchable filter that:

* rejects classes outside a configured ``domain_classes`` whitelist,
* rejects detections below a per-class confidence threshold (fitted on ``val``),
* emits ``unknown`` when the top-1 and top-2 class scores are within
  ``margin_min`` (the model is not sure *which* class it is), and
* rejects boxes below a minimum area fraction / outside an aspect-ratio bound.

Every input detection produces exactly one output detection - ``accepted``, or
``unknown``, or ``rejected`` - and the counts reconcile
(``accepted + unknown + rejected == input``). A rejected-but-real detection keeps
its box (``emit_unknown``) so a later phase still sees the object; only its
``class_name`` becomes ``"unknown"`` and ``policy_state`` records why.

Never raises into the analysis loop: :meth:`RecognitionPolicy.apply` catches its
own errors, counts the frame as a failure, and passes the detections through
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from predictivesense.config.settings import PolicyConfig
from predictivesense.core.types import Detection
from predictivesense.logging_setup import get_logger

__all__ = ["RecognitionPolicy", "PolicyOutcome", "PolicyCounts", "POLICY_STATES", "UNKNOWN"]

_LOG = get_logger(__name__)

UNKNOWN = "unknown"

POLICY_STATES = (
    "accepted",
    "unknown_low_confidence",
    "unknown_margin",
    "rejected_out_of_domain",
    "rejected_size",
)
_UNKNOWN_STATES = ("unknown_low_confidence", "unknown_margin")
_REJECTED_STATES = ("rejected_out_of_domain", "rejected_size")


@dataclass(frozen=True)
class PolicyCounts:
    """Per-rule tallies for one ``apply`` call (or accumulated). Reconciles."""

    input: int = 0
    accepted: int = 0
    unknown_low_confidence: int = 0
    unknown_margin: int = 0
    rejected_out_of_domain: int = 0
    rejected_size: int = 0
    errors: int = 0

    @property
    def unknown(self) -> int:
        return self.unknown_low_confidence + self.unknown_margin

    @property
    def rejected(self) -> int:
        return self.rejected_out_of_domain + self.rejected_size

    @property
    def reconciles(self) -> bool:
        return self.accepted + self.unknown + self.rejected == self.input

    def merged(self, other: "PolicyCounts") -> "PolicyCounts":
        return PolicyCounts(
            input=self.input + other.input,
            accepted=self.accepted + other.accepted,
            unknown_low_confidence=self.unknown_low_confidence + other.unknown_low_confidence,
            unknown_margin=self.unknown_margin + other.unknown_margin,
            rejected_out_of_domain=self.rejected_out_of_domain + other.rejected_out_of_domain,
            rejected_size=self.rejected_size + other.rejected_size,
            errors=self.errors + other.errors,
        )

    def as_metrics(self, prefix: str = "policy_") -> dict[str, float]:
        return {
            f"{prefix}input": float(self.input),
            f"{prefix}accepted": float(self.accepted),
            f"{prefix}unknown_low_confidence": float(self.unknown_low_confidence),
            f"{prefix}unknown_margin": float(self.unknown_margin),
            f"{prefix}rejected_out_of_domain": float(self.rejected_out_of_domain),
            f"{prefix}rejected_size": float(self.rejected_size),
            f"{prefix}errors": float(self.errors),
        }


@dataclass(frozen=True)
class PolicyOutcome:
    """Result of applying the policy to one frame's detections."""

    detections: list[Detection] = field(default_factory=list)
    counts: PolicyCounts = field(default_factory=PolicyCounts)
    # (raw_class_name, decided_class_name) for every input detection, in order -
    # backs the Diagnostics "raw vs decided" readout for the most recent frame.
    raw_to_decided: list[tuple[str, str]] = field(default_factory=list)


class RecognitionPolicy:
    """Stateless per-frame filter built from a :class:`PolicyConfig`."""

    def __init__(self, config: PolicyConfig) -> None:
        self._cfg = config
        self._domain = set(config.domain_classes)
        _LOG.info(
            "recognition policy: enabled=%s domain=%s margin_min=%.3f "
            "default_threshold=%.3f min_box_area_frac=%.5f per_class_thresholds=%d",
            config.enabled,
            config.domain_restriction,
            config.margin_min,
            config.default_threshold,
            config.min_box_area_frac,
            len(config.per_class_thresholds),
        )

    # -- introspection -------------------------------------------------

    def active_thresholds(self) -> dict[str, float]:
        """Effective per-class threshold for every domain class (read-only UI)."""

        return {
            name: float(self._cfg.per_class_thresholds.get(name, self._cfg.default_threshold))
            for name in sorted(self._domain)
        }

    def info(self) -> dict[str, object]:
        return {
            "enabled": self._cfg.enabled,
            "domain_restriction": self._cfg.domain_restriction,
            "margin_rule": self._cfg.margin_rule,
            "size_rule": self._cfg.size_rule,
            "per_class_threshold_rule": self._cfg.per_class_threshold_rule,
            "domain_classes": sorted(self._domain),
            "default_threshold": self._cfg.default_threshold,
            "margin_min": self._cfg.margin_min,
            "min_box_area_frac": self._cfg.min_box_area_frac,
            "emit_unknown": self._cfg.emit_unknown,
        }

    # -- application -------------------------------------------------

    def apply(
        self,
        detections: Iterable[Detection],
        *,
        frame_width: int,
        frame_height: int,
    ) -> PolicyOutcome:
        """Annotate every detection with a ``policy_state`` and a decided
        ``class_name``. Never raises: on an internal error the frame is passed
        through unchanged and ``counts.errors`` is incremented."""

        dets = list(detections)
        if not self._cfg.enabled:
            passthrough = [self._passthrough(d) for d in dets]
            return PolicyOutcome(
                detections=passthrough,
                counts=PolicyCounts(input=len(dets), accepted=len(dets)),
                raw_to_decided=[(d.class_name, d.class_name) for d in passthrough],
            )

        try:
            return self._apply(dets, frame_width, frame_height)
        except Exception as exc:  # noqa: BLE001 - a policy bug must not kill the loop
            _LOG.error("recognition policy raised; passing frame through: %r", exc)
            passthrough = [self._passthrough(d) for d in dets]
            return PolicyOutcome(
                detections=passthrough,
                counts=PolicyCounts(input=len(dets), accepted=len(dets), errors=1),
                raw_to_decided=[(d.class_name, d.class_name) for d in passthrough],
            )

    # -- internals -------------------------------------------------

    @staticmethod
    def _passthrough(d: Detection) -> Detection:
        raw = d.raw_class_name or d.class_name
        return d.model_copy(update={"raw_class_name": raw, "policy_state": "accepted"})

    def _apply(
        self, dets: list[Detection], frame_w: int, frame_h: int
    ) -> PolicyOutcome:
        frame_area = float(max(1, frame_w) * max(1, frame_h))
        cfg = self._cfg

        out: list[Detection] = []
        raw_to_decided: list[tuple[str, str]] = []
        tally = {k: 0 for k in POLICY_STATES}

        for d in dets:
            raw = d.raw_class_name or d.class_name
            x1, y1, x2, y2 = d.bbox
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area_frac = (w * h) / frame_area
            aspect = (w / h) if h > 0 else 0.0

            state = "accepted"

            if cfg.size_rule and (
                area_frac < cfg.min_box_area_frac or not self._aspect_ok(raw, aspect)
            ):
                state = "rejected_size"
            elif cfg.domain_restriction and raw not in self._domain:
                state = "rejected_out_of_domain"
            elif cfg.per_class_threshold_rule and d.score < self._threshold(raw):
                state = "unknown_low_confidence"
            elif (
                cfg.margin_rule
                and d.runner_up is not None
                and (d.score - float(d.runner_up[1])) < cfg.margin_min
            ):
                state = "unknown_margin"

            tally[state] += 1
            decided = raw if state == "accepted" else UNKNOWN
            if state != "accepted" and not cfg.emit_unknown:
                decided = raw  # keep the raw label but still flag the state
            out.append(
                d.model_copy(
                    update={
                        "raw_class_name": raw,
                        "class_name": decided,
                        "policy_state": state,
                    }
                )
            )
            raw_to_decided.append((raw, decided))

        counts = PolicyCounts(
            input=len(dets),
            accepted=tally["accepted"],
            unknown_low_confidence=tally["unknown_low_confidence"],
            unknown_margin=tally["unknown_margin"],
            rejected_out_of_domain=tally["rejected_out_of_domain"],
            rejected_size=tally["rejected_size"],
        )
        return PolicyOutcome(detections=out, counts=counts, raw_to_decided=raw_to_decided)

    def _threshold(self, class_name: str) -> float:
        return float(
            self._cfg.per_class_thresholds.get(class_name, self._cfg.default_threshold)
        )

    def _aspect_ok(self, class_name: str, aspect: float) -> bool:
        bound = self._cfg.aspect_ratio_bounds.get(class_name)
        if not bound or aspect <= 0.0:
            return True
        lo, hi = bound
        return lo <= aspect <= hi
