"""Paired failure classification and summary statistics for Q01 runs."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .evaluation import QueryComparison
from .maintenance import BuildCost, StateComparison


class FailureKind(str, Enum):
    STATE_MISMATCH = "state_mismatch"
    OLD_SOURCE_EXPOSURE = "old_source_exposure"
    SURVIVOR_LOSS = "survivor_loss"
    QUERY_MISMATCH = "query_mismatch"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    SUPPORT_MISMATCH = "support_mismatch"
    RELATION_MISMATCH = "relation_mismatch"
    RETRIEVAL_MISMATCH = "retrieval_mismatch"
    COST_REGRESSION = "cost_regression"
    TRACE_INCOMPLETE = "trace_incomplete"
    SNAPSHOT_UNSTABLE = "snapshot_unstable"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"


@dataclass(frozen=True)
class FailureRecord:
    scenario_id: str
    arm: str
    kind: FailureKind
    severity: str = "error"
    object_ids: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "arm": self.arm,
            "kind": self.kind.value,
            "severity": self.severity,
            "object_ids": list(self.object_ids),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class PairedObservation:
    """One arm result paired with the M0 reference for one scenario."""

    scenario_id: str
    arm: str
    comparison: StateComparison
    cost: BuildCost
    failures: tuple[FailureRecord, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def exact(self) -> bool:
        return self.comparison.equal and not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "arm": self.arm,
            "exact": self.exact,
            "comparison": {
                "equal": self.comparison.equal,
                "left_hash": self.comparison.left_hash,
                "right_hash": self.comparison.right_hash,
                "missing_ids": list(self.comparison.missing_ids),
                "extra_ids": list(self.comparison.extra_ids),
                "changed_ids": list(self.comparison.changed_ids),
                "reason": self.comparison.reason,
            },
            "cost": {
                "input_events": self.cost.input_events,
                "replayed_events": self.cost.replayed_events,
                "prefix_replayed_events": self.cost.prefix_replayed_events,
                "suffix_replayed_events": self.cost.suffix_replayed_events,
                "checkpoint_restore_events": self.cost.checkpoint_restore_events,
                "checkpoint_restore_bytes": self.cost.checkpoint_restore_bytes,
                "cost_scope": self.cost.cost_scope,
                "retrieved_candidates": self.cost.retrieved_candidates,
                "generated_objects": self.cost.generated_objects,
                "state_records": self.cost.state_records,
                "llm_calls": self.cost.llm_calls,
                "input_tokens": self.cost.input_tokens,
                "output_tokens": self.cost.output_tokens,
                "embedding_calls": self.cost.embedding_calls,
                "embedding_items": self.cost.embedding_items,
                "index_updates": self.cost.index_updates,
                "memory_reads": self.cost.memory_reads,
                "bytes_read": self.cost.bytes_read,
                "bytes_written": self.cost.bytes_written,
                "wall_time_ms": self.cost.wall_time_ms,
                "peak_memory_mb": self.cost.peak_memory_mb,
            },
            "failures": [failure.to_dict() for failure in self.failures],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PairedArmSummary:
    arm: str
    scenario_count: int
    exact_count: int
    exact_rate: float
    mean_replayed_events: float
    p95_replayed_events: float
    mean_retrieved_candidates: float
    failure_counts: Mapping[str, int]
    cost_savings_vs_reference: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "scenario_count": self.scenario_count,
            "exact_count": self.exact_count,
            "exact_rate": self.exact_rate,
            "mean_replayed_events": self.mean_replayed_events,
            "p95_replayed_events": self.p95_replayed_events,
            "mean_retrieved_candidates": self.mean_retrieved_candidates,
            "failure_counts": dict(self.failure_counts),
            "cost_savings_vs_reference": self.cost_savings_vs_reference,
        }


def classify_state_comparison(
    comparison: StateComparison,
    *,
    scenario_id: str,
    arm: str,
) -> list[FailureRecord]:
    """Turn a state diff into explicit, machine-readable failure records."""

    failures: list[FailureRecord] = []
    if comparison.equal:
        return failures
    if comparison.missing_ids or comparison.extra_ids or comparison.changed_ids:
        object_ids = tuple(sorted(set(comparison.missing_ids) | set(comparison.extra_ids) | set(comparison.changed_ids)))
        failures.append(
            FailureRecord(
                scenario_id=scenario_id,
                arm=arm,
                kind=FailureKind.STATE_MISMATCH,
                object_ids=object_ids,
                details={
                    "missing_ids": list(comparison.missing_ids),
                    "extra_ids": list(comparison.extra_ids),
                    "changed_ids": list(comparison.changed_ids),
                    "reason": comparison.reason,
                },
            )
        )
    return failures


def paired_observation(
    *,
    scenario_id: str,
    arm: str,
    comparison: StateComparison,
    cost: BuildCost,
    metadata: Mapping[str, Any] | None = None,
    extra_failures: Iterable[FailureRecord] = (),
) -> PairedObservation:
    failures = tuple(classify_state_comparison(comparison, scenario_id=scenario_id, arm=arm)) + tuple(extra_failures)
    return PairedObservation(
        scenario_id=scenario_id,
        arm=arm,
        comparison=comparison,
        cost=cost,
        failures=failures,
        metadata=dict(metadata or {}),
    )


def classify_query_comparison(
    comparison: QueryComparison,
    *,
    scenario_id: str,
    arm: str,
) -> list[FailureRecord]:
    failures: list[FailureRecord] = []
    if not comparison.answer_equal:
        failures.append(FailureRecord(scenario_id, arm, FailureKind.QUERY_MISMATCH))
    if not comparison.evidence_equal:
        failures.append(FailureRecord(scenario_id, arm, FailureKind.EVIDENCE_MISMATCH))
    if comparison.old_source_exposure:
        failures.append(
            FailureRecord(
                scenario_id,
                arm,
                FailureKind.OLD_SOURCE_EXPOSURE,
                object_ids=comparison.old_source_exposure,
            )
        )
    return failures


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def summarize_paired_observations(
    observations: Sequence[PairedObservation],
    *,
    reference_replayed_events: Mapping[str, float] | None = None,
) -> dict[str, PairedArmSummary]:
    """Summarize paired results by arm using history-level observations."""

    grouped: dict[str, list[PairedObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.arm, []).append(observation)
    summaries: dict[str, PairedArmSummary] = {}
    for arm, items in sorted(grouped.items()):
        replayed = [float(item.cost.replayed_events) for item in items]
        candidates = [float(item.cost.retrieved_candidates) for item in items]
        failures = Counter(
            failure.kind.value for item in items for failure in item.failures
        )
        savings: float | None = None
        if reference_replayed_events:
            ratios = []
            for item in items:
                # A cheap but incorrect state is not an efficiency result.
                # Only exact-to-reference observations may contribute to a
                # claimed maintenance saving.
                if not item.exact:
                    continue
                reference = float(reference_replayed_events.get(item.scenario_id, 0.0))
                if reference > 0:
                    ratios.append(1.0 - float(item.cost.replayed_events) / reference)
            if ratios:
                savings = sum(ratios) / len(ratios)
        summaries[arm] = PairedArmSummary(
            arm=arm,
            scenario_count=len(items),
            exact_count=sum(1 for item in items if item.exact),
            exact_rate=(sum(1 for item in items if item.exact) / len(items)) if items else 0.0,
            mean_replayed_events=(sum(replayed) / len(replayed)) if replayed else 0.0,
            p95_replayed_events=_percentile(replayed, 0.95),
            mean_retrieved_candidates=(sum(candidates) / len(candidates)) if candidates else 0.0,
            failure_counts=dict(sorted(failures.items())),
            cost_savings_vs_reference=savings,
        )
    return summaries


def write_failure_report(
    path: str | Path,
    observations: Sequence[PairedObservation],
    summaries: Mapping[str, PairedArmSummary] | None = None,
) -> Path:
    """Write a deterministic JSON report for review and failure replay."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "observations": [observation.to_dict() for observation in observations],
        "summaries": {
            key: value.to_dict()
            for key, value in sorted((summaries or summarize_paired_observations(observations)).items())
        },
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def paired_bootstrap_mean_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Return a deterministic paired bootstrap CI for ``mean(left-right)``."""

    if len(left) != len(right):
        raise ValueError("paired bootstrap inputs must have equal length")
    if not left:
        return {"n": 0, "estimate": 0.0, "lower": 0.0, "upper": 0.0, "samples": 0}
    if samples < 1:
        raise ValueError("samples must be positive")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(samples):
        draw = [differences[rng.randrange(len(differences))] for _ in differences]
        boot.append(sum(draw) / len(draw))
    alpha = max(0.0, min(1.0, (1.0 - confidence) / 2.0))
    return {
        "n": len(differences),
        "estimate": sum(differences) / len(differences),
        "lower": _percentile(boot, alpha),
        "upper": _percentile(boot, 1.0 - alpha),
        "samples": samples,
    }


def affected_set_precision_recall(
    predicted: Iterable[str],
    reference: Iterable[str],
) -> dict[str, float | int]:
    """Score a maintenance arm's affected-object prediction.

    ``reference`` is the actual object-level difference induced by the M0
    counterfactual (or another explicitly declared oracle).  This diagnostic
    is separate from state exactness: an arm may conservatively over-invalidate
    and still be correct, while a missed object is a correctness risk.
    """

    predicted_set = {str(item) for item in predicted}
    reference_set = {str(item) for item in reference}
    true_positive = len(predicted_set & reference_set)
    false_positive = len(predicted_set - reference_set)
    false_negative = len(reference_set - predicted_set)
    precision = true_positive / len(predicted_set) if predicted_set else (1.0 if not reference_set else 0.0)
    recall = true_positive / len(reference_set) if reference_set else (1.0 if not predicted_set else 0.0)
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "predicted_count": len(predicted_set),
        "reference_count": len(reference_set),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


__all__ = [
    "FailureKind",
    "FailureRecord",
    "PairedObservation",
    "PairedArmSummary",
    "classify_state_comparison",
    "paired_observation",
    "classify_query_comparison",
    "summarize_paired_observations",
    "write_failure_report",
    "paired_bootstrap_mean_difference",
    "affected_set_precision_recall",
]
