"""Canonical maintenance evaluation helpers.

This module joins the backend snapshot, decision trace and cost vector into
one result that can be compared across M0--M3.  It deliberately keeps the
oracle separate from any proposed maintenance algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .backend import MemorySnapshot, QueryResult, SourceRecord
from .maintenance import BuildCost, BuildResult, MaintenanceArm, StateComparison
from .state import (
    CanonicalQuery,
    CanonicalState,
    CanonicalStateComparator,
    ProvenanceEvent,
    SourceProvenance,
)


def snapshot_to_canonical(
    snapshot: MemorySnapshot,
    *,
    dataset: str = "unknown",
    instance_id: str = "unknown",
    events: Sequence[ProvenanceEvent] = (),
    queries: Sequence[CanonicalQuery] = (),
    source_kinds: Iterable[str] = ("source", "turn", "document"),
) -> CanonicalState:
    """Lift a backend snapshot while keeping physical IDs out of semantics.

    Source records become stable provenance entries. Derived records remain
    artifacts whose semantic IDs are derived from content/support/parents by
    :class:`CanonicalState`, so UUIDs allocated by a rebuild do not decide
    equality.
    """

    source_kind_set = set(source_kinds)
    source_records = [record for record in snapshot.records if record.kind in source_kind_set]
    if not source_records:
        # A backend that only exposes notes still needs a deterministic source
        # ledger for an engineering fixture; the record IDs are then explicit
        # inputs rather than inferred from wall-clock fields.
        source_records = list(snapshot.records)
    provenance = [
        SourceProvenance.from_record(
            record,
            dataset=dataset,
            instance_id=instance_id,
            ordinal=index,
        )
        for index, record in enumerate(sorted(source_records, key=lambda item: item.source_id))
    ]
    state = CanonicalState.from_snapshot(
        snapshot,
        dataset=dataset,
        instance_id=instance_id,
        source_records=provenance,
    )
    state.provenance_events = list(events)
    state.queries = list(queries)
    return state


@dataclass
class CanonicalArmResult:
    arm: MaintenanceArm | str
    snapshot: MemorySnapshot
    state: CanonicalState
    canonical_hash: str
    cost: BuildCost = field(default_factory=BuildCost)
    exact_to_reference: bool = False
    mismatch_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalMaintenanceReport:
    reference: CanonicalArmResult
    arms: dict[str, CanonicalArmResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "reference_hash": self.reference.canonical_hash,
            "arms": {
                str(key): {
                    "canonical_hash": value.canonical_hash,
                    "exact_to_reference": value.exact_to_reference,
                    "mismatch_counts": value.mismatch_counts,
                    "cost": value.cost.__dict__,
                }
                for key, value in self.arms.items()
            },
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class QueryComparison:
    """Paired answer/evidence comparison for one fixed question."""

    answer_equal: bool
    evidence_equal: bool
    old_source_exposure: tuple[str, ...] = ()

    @property
    def equivalent(self) -> bool:
        return self.answer_equal and self.evidence_equal and not self.old_source_exposure


def compare_query_results(
    left: QueryResult,
    right: QueryResult,
    *,
    deleted_source_ids: Iterable[str] = (),
) -> QueryComparison:
    deleted = set(deleted_source_ids)
    observed = (
        set(left.evidence.source_ids)
        | set(left.evidence.read_set)
        | set(right.evidence.source_ids)
        | set(right.evidence.read_set)
    )
    exposure = tuple(sorted(deleted.intersection(observed)))
    return QueryComparison(
        answer_equal=left.answer == right.answer,
        evidence_equal=(
            left.evidence.source_ids == right.evidence.source_ids
            and left.evidence.read_set == right.evidence.read_set
        ),
        old_source_exposure=exposure,
    )


def evaluate_maintenance_outputs(
    outputs: Mapping[Any, tuple[MemorySnapshot | BuildResult, Any]],
    *,
    dataset: str = "unknown",
    instance_id: str = "unknown",
    events_by_arm: Mapping[Any, Sequence[ProvenanceEvent]] | None = None,
    queries_by_arm: Mapping[Any, Sequence[CanonicalQuery]] | None = None,
    comparator: CanonicalStateComparator | None = None,
) -> CanonicalMaintenanceReport:
    """Compare M0--M3 outputs using the canonical state oracle.

    ``outputs`` accepts either the existing ``(BuildResult, comparison)``
    shape or ``(MemorySnapshot, metadata)``.  The second element is retained
    as metadata and is never treated as correctness evidence.
    """

    comparator = comparator or CanonicalStateComparator()
    normalized: dict[str, CanonicalArmResult] = {}
    for arm_key, pair in outputs.items():
        result_or_snapshot, auxiliary = pair
        if isinstance(result_or_snapshot, BuildResult):
            snapshot = result_or_snapshot.snapshot
            cost = result_or_snapshot.cost
            metadata = dict(result_or_snapshot.metadata)
            arm = result_or_snapshot.arm
        else:
            snapshot = result_or_snapshot
            cost = BuildCost()
            metadata = dict(auxiliary) if isinstance(auxiliary, Mapping) else {}
            arm = arm_key
        events = list((events_by_arm or {}).get(arm_key, ()))
        queries = list((queries_by_arm or {}).get(arm_key, ()))
        state = snapshot_to_canonical(
            snapshot,
            dataset=dataset,
            instance_id=instance_id,
            events=events,
            queries=queries,
        )
        normalized[str(getattr(arm, "value", arm))] = CanonicalArmResult(
            arm=arm,
            snapshot=snapshot,
            state=state,
            canonical_hash=state.canonical_hash,
            cost=cost,
            metadata=metadata,
        )
    reference = normalized.get(MaintenanceArm.M0_FULL_REBUILD.value)
    if reference is None:
        raise ValueError("maintenance outputs must contain M0_full_rebuild")
    for result in normalized.values():
        diff = comparator.compare(result.state, reference.state)
        result.exact_to_reference = diff.equivalent
        result.mismatch_counts = diff.mismatch_counts
    return CanonicalMaintenanceReport(reference=reference, arms=normalized)


def old_source_exposure(snapshot: MemorySnapshot, deleted_ids: Iterable[str]) -> list[str]:
    """Return active records whose support or lineage still cites a deleted source."""

    deleted = set(deleted_ids)
    exposed: list[str] = []
    for record in snapshot.records:
        if record.source_id in deleted or deleted.intersection(record.support_set) or deleted.intersection(record.derived_from):
            exposed.append(record.source_id)
    return sorted(set(exposed))


def survivor_loss(snapshot: MemorySnapshot, retained_source_ids: Iterable[str]) -> list[str]:
    active = {record.source_id for record in snapshot.records}
    return sorted(set(retained_source_ids) - active)


__all__ = [
    "CanonicalArmResult",
    "CanonicalMaintenanceReport",
    "QueryComparison",
    "compare_query_results",
    "evaluate_maintenance_outputs",
    "old_source_exposure",
    "snapshot_to_canonical",
    "survivor_loss",
]
