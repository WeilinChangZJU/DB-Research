"""Deterministic Q01 maintenance arms for offline validation.

The classes in this module deliberately operate on a tiny source/event builder,
not on an LLM.  They make the four required comparison arms executable before
an actual memory builder is instrumented:

* M0: full rebuild after removing the source set;
* M1: tombstone/filter of materialized state;
* M2: suffix replay from the earliest deleted source;
* M3: replay from the earliest step invalidated by a recorded read/control set.

This is a reference harness.  It is not an efficacy baseline for Bemula or a
claim that an arbitrary memory system supports exact incremental deletion.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .backend import MemorySnapshot, SourceRecord


class MaintenanceArm(str, Enum):
    """Fixed Q01 comparison arms."""

    M0_FULL_REBUILD = "M0_full_rebuild"
    M1_TOMBSTONE = "M1_tombstone_filter"
    M2_REPLAY = "M2_replay_repair_all"
    M3_READ_SET = "M3_exact_read_set"


@dataclass(frozen=True)
class SourceEvent:
    """One immutable external input event with a stable source identity."""

    event_id: str
    source_id: str
    content: str
    timestamp: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildStep:
    """One builder decision, including non-materialized candidate reads."""

    index: int
    event_id: str
    source_id: str
    operation: str
    read_set: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    output_ids: tuple[str, ...] = ()
    support_set: tuple[str, ...] = ()
    control_set: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildCost:
    """Counters used to compare maintenance work without wall-clock noise."""

    input_events: int = 0
    replayed_events: int = 0
    retrieved_candidates: int = 0
    generated_objects: int = 0
    state_records: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_calls: int = 0
    embedding_items: int = 0
    index_updates: int = 0
    memory_reads: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    wall_time_ms: float = 0.0
    peak_memory_mb: float | None = None
    # ``replayed_events`` is the total number of events actually executed by
    # the arm.  Prefix/suffix counters make the accounting auditable when a
    # nominal "suffix replay" first reconstructs a prefix.
    prefix_replayed_events: int = 0
    suffix_replayed_events: int = 0
    checkpoint_restore_events: int = 0
    checkpoint_restore_bytes: int = 0
    cost_scope: str = "maintenance"


@dataclass
class BuildResult:
    """Result of one deterministic build or maintenance arm."""

    arm: MaintenanceArm
    snapshot: MemorySnapshot
    trace: list[BuildStep]
    cost: BuildCost
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StateComparison:
    """Canonical comparison between two visible snapshots."""

    equal: bool
    left_hash: str
    right_hash: str
    missing_ids: tuple[str, ...] = ()
    extra_ids: tuple[str, ...] = ()
    changed_ids: tuple[str, ...] = ()
    reason: str = ""


def _token_set(text: str) -> set[str]:
    # ``\w`` must be interpreted by the regex engine; the old character
    # class matched only a literal backslash or ``w``.
    return {x for x in re.findall(r"\w+", text.lower()) if len(x) > 1}


def _content_key(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def compare_snapshots(left: MemorySnapshot, right: MemorySnapshot) -> StateComparison:
    """Compare visible records, content, support and derived lineage."""

    lm = {record.source_id: record for record in left.records}
    rm = {record.source_id: record for record in right.records}
    missing = tuple(sorted(set(rm) - set(lm)))
    extra = tuple(sorted(set(lm) - set(rm)))
    changed: list[str] = []
    for source_id in sorted(set(lm) & set(rm)):
        a = lm[source_id]
        b = rm[source_id]
        if a.model_dump(mode="json") != b.model_dump(mode="json"):
            changed.append(source_id)
    equal = not missing and not extra and not changed and left.tombstones == right.tombstones
    reason = "equal" if equal else "visible state differs"
    return StateComparison(
        equal=equal,
        left_hash=left.state_hash or "",
        right_hash=right.state_hash or "",
        missing_ids=missing,
        extra_ids=extra,
        changed_ids=tuple(changed),
        reason=reason,
    )


class DeterministicMemoryBuilder:
    """Small replayable builder used by offline maintenance tests.

    It materializes one source record per event and one derived ``fact`` record
    per normalized content value.  Repeated content updates the existing fact's
    support set and records the candidate/decision in the trace, which gives M3
    a concrete control-dependency signal.
    """

    name = "deterministic-reference-builder"

    def build(
        self,
        events: Sequence[SourceEvent],
        *,
        arm: MaintenanceArm = MaintenanceArm.M0_FULL_REBUILD,
        deleted_ids: Iterable[str] = (),
        start_index: int = 0,
        initial_snapshot: MemorySnapshot | None = None,
        replayed_events: int | None = None,
    ) -> BuildResult:
        deleted = set(deleted_ids)
        records: dict[str, SourceRecord] = {}
        tombstones: set[str] = set(initial_snapshot.tombstones if initial_snapshot else ())
        if initial_snapshot is not None:
            records = {record.source_id: record for record in initial_snapshot.records}
        trace: list[BuildStep] = []
        generated = 0
        candidates_read = 0
        selected = events[start_index:]
        for offset, event in enumerate(selected, start=start_index):
            if event.source_id in deleted:
                tombstones.add(event.source_id)
                continue
            fact_id = f"fact:{_content_key(event.content)}"
            candidate_ids = tuple(
                sorted(record.source_id for record in records.values() if record.kind == "fact")
            )
            read_set = tuple(
                sorted(
                    record.source_id
                    for record in records.values()
                    if record.kind == "fact" and _token_set(record.content) & _token_set(event.content)
                )
            )
            candidates_read += len(candidate_ids)
            source_record = SourceRecord(
                source_id=event.source_id,
                content=event.content,
                kind="source",
                timestamp=event.timestamp,
                source_hash=None,
                metadata=dict(event.metadata),
            )
            records[event.source_id] = source_record
            prior = records.get(fact_id)
            support = set(prior.support_set if prior else ())
            support.add(event.source_id)
            fact = SourceRecord(
                source_id=fact_id,
                content=event.content,
                kind="fact",
                timestamp=event.timestamp,
                support_set=sorted(support),
                derived_from=sorted(support),
                metadata={"normalized_key": _content_key(event.content)},
            )
            operation = "add_fact" if prior is None else "update_support"
            if prior is None:
                generated += 1
            records[fact_id] = fact
            trace.append(
                BuildStep(
                    index=offset,
                    event_id=event.event_id,
                    source_id=event.source_id,
                    operation=operation,
                    read_set=read_set,
                    candidate_ids=candidate_ids,
                    output_ids=(event.source_id, fact_id),
                    support_set=tuple(sorted(support)),
                    control_set=tuple(sorted(set(candidate_ids) | set(read_set))),
                    metadata={"fact_id": fact_id},
                )
            )
        snapshot = MemorySnapshot(
            state_id=f"state:{arm.value}:{len(trace):06d}",
            records=[records[key] for key in sorted(records)],
            tombstones=sorted(tombstones),
            metadata={"builder": self.name, "arm": arm.value, "start_index": start_index},
        )
        cost = BuildCost(
            input_events=len(events),
            replayed_events=len(selected) if replayed_events is None else replayed_events,
            retrieved_candidates=candidates_read,
            generated_objects=generated,
            state_records=len(records),
        )
        return BuildResult(arm=arm, snapshot=snapshot, trace=trace, cost=cost)

    def run_arm(
        self,
        events: Sequence[SourceEvent],
        deleted_ids: Iterable[str],
        arm: MaintenanceArm,
        *,
        baseline: BuildResult | None = None,
    ) -> BuildResult:
        """Run one fixed maintenance arm against the event history."""

        deleted = set(deleted_ids)
        if baseline is None:
            baseline = self.build(events, arm=MaintenanceArm.M0_FULL_REBUILD, replayed_events=len(events))
        if arm is MaintenanceArm.M0_FULL_REBUILD:
            kept = [event for event in events if event.source_id not in deleted]
            result = self.build(kept, arm=arm, deleted_ids=deleted, replayed_events=len(kept))
            result.cost = replace(result.cost, cost_scope="reference_rebuild")
            result.metadata["cost_scope"] = "reference_rebuild"
            return result
        if arm is MaintenanceArm.M1_TOMBSTONE:
            return self._tombstone_filter(baseline, deleted)
        if arm is MaintenanceArm.M2_REPLAY:
            # M2 is the conservative repair-all baseline: deletion of any
            # source invalidates the complete suffix after its first event.
            start = self._earliest_deleted_index(events, deleted)
            return self._replay_from_start(
                events, deleted, arm, start=start, mode="conservative_suffix"
            )
        if arm is MaintenanceArm.M3_READ_SET:
            # M3 uses the recorded read/candidate/control closure to choose a
            # safe invalidation point.  It is intentionally a separate method
            # and metadata contract so results cannot be mislabelled as M2.
            affected = [
                step.index
                for step in baseline.trace
                if deleted.intersection(
                    set(step.read_set)
                    | set(step.candidate_ids)
                    | set(step.support_set)
                    | set(getattr(step, "control_set", ()))
                    | {step.source_id}
                )
            ]
            fallback = self._earliest_deleted_index(events, deleted)
            start = min(affected) if affected else fallback
            result = self._replay_from_start(
                events, deleted, arm, start=start, mode="read_set_closure"
            )
            result.metadata.update(
                {
                    "affected_steps": affected,
                    "replay_start": start,
                    "dependency_baseline": "materialized_read_control_set",
                    "m3_status": "implemented_conservative_exact_reference",
                }
            )
            return result
        raise ValueError(f"unknown maintenance arm: {arm}")

    def _replay_from_start(
        self,
        events: Sequence[SourceEvent],
        deleted: set[str],
        arm: MaintenanceArm,
        *,
        start: int,
        mode: str,
    ) -> BuildResult:
        """Rebuild the prefix and replay the valid suffix with full accounting.

        This is not checkpoint restoration: the prefix is executed again.  The
        returned cost therefore includes both phases and records the scope so
        callers cannot interpret ``suffix_replayed_events`` as total work.
        """
        prefix = [event for event in events[:start] if event.source_id not in deleted]
        prefix_result = self.build(prefix, arm=arm, deleted_ids=deleted, replayed_events=len(prefix))
        suffix = [event for event in events[start:] if event.source_id not in deleted]
        result = self.build(
            suffix,
            arm=arm,
            deleted_ids=deleted,
            start_index=0,
            initial_snapshot=prefix_result.snapshot,
            replayed_events=len(suffix),
        )
        total_cost = BuildCost(
            input_events=len(events),
            replayed_events=prefix_result.cost.replayed_events + result.cost.replayed_events,
            prefix_replayed_events=prefix_result.cost.replayed_events,
            suffix_replayed_events=result.cost.replayed_events,
            retrieved_candidates=prefix_result.cost.retrieved_candidates + result.cost.retrieved_candidates,
            generated_objects=prefix_result.cost.generated_objects + result.cost.generated_objects,
            state_records=result.cost.state_records,
            cost_scope="maintenance_full_execution",
        )
        result.cost = total_cost
        result.metadata.update(
            {
                "replay_mode": mode,
                "replay_start": start,
                "prefix_events": len(prefix),
                "suffix_events": len(suffix),
                "cost_scope": "maintenance_full_execution",
                "checkpoint_restored": False,
            }
        )
        return result

    @staticmethod
    def _earliest_deleted_index(events: Sequence[SourceEvent], deleted: set[str]) -> int:
        for index, event in enumerate(events):
            if event.source_id in deleted:
                return index
        return len(events)

    @staticmethod
    def _tombstone_filter(baseline: BuildResult, deleted: set[str]) -> BuildResult:
        """Weak M1 arm: remove every materialized object touching a target."""

        records = [
            record
            for record in baseline.snapshot.records
            if record.source_id not in deleted and not deleted.intersection(record.support_set)
        ]
        snapshot = MemorySnapshot(
            state_id=f"state:{MaintenanceArm.M1_TOMBSTONE.value}:{len(records):06d}",
            records=records,
            tombstones=sorted(set(baseline.snapshot.tombstones) | deleted),
            metadata={"builder": baseline.snapshot.metadata.get("builder"), "arm": MaintenanceArm.M1_TOMBSTONE.value},
        )
        return BuildResult(
            arm=MaintenanceArm.M1_TOMBSTONE,
            snapshot=snapshot,
            trace=list(baseline.trace),
            cost=BuildCost(
                input_events=baseline.cost.input_events,
                replayed_events=0,
                retrieved_candidates=0,
                generated_objects=0,
                state_records=len(records),
            ),
        )


def run_all_arms(
    events: Sequence[SourceEvent],
    deleted_ids: Iterable[str],
    *,
    builder: DeterministicMemoryBuilder | None = None,
) -> dict[MaintenanceArm, tuple[BuildResult, StateComparison]]:
    """Run M0-M3 and compare each result with the M0 reference.

    ``builder`` is injectable so the paired runner can use the same harness
    with a builder-specific deterministic implementation.  The default stays
    the small reference builder used by the offline tests.
    """

    builder = builder or DeterministicMemoryBuilder()
    deleted = set(deleted_ids)
    reference = builder.run_arm(events, deleted, MaintenanceArm.M0_FULL_REBUILD)
    output: dict[MaintenanceArm, tuple[BuildResult, StateComparison]] = {}
    for arm in MaintenanceArm:
        result = reference if arm is MaintenanceArm.M0_FULL_REBUILD else builder.run_arm(
            events, deleted, arm, baseline=builder.build(events)
        )
        output[arm] = (result, compare_snapshots(result.snapshot, reference.snapshot))
    return output
