"""Deterministic deletion-scenario manifests for Q01 validation.

The benchmark datasets provide conversation and QA semantics, but they do not
provide a post-deletion canonical memory state.  This module creates explicit,
reproducible deletion masks over a source history.  A scenario is an input
condition; it is not an efficacy result.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .maintenance import BuildStep, SourceEvent


class DeletionScenarioKind(str, Enum):
    SINGLE = "single"
    ISOLATED = "isolated"
    SHARED_SUPPORT = "shared_support"
    RETRIEVAL_ONLY = "retrieval_only"
    TOP_K_BOUNDARY = "top_k_boundary"
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"
    BATCH = "batch"
    NOOP = "noop"
    REINSERT = "reinsert"
    LINKED_NEIGHBOR = "linked_neighbor"


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DeletionScenario:
    """One deterministic source deletion condition."""

    scenario_id: str
    kind: DeletionScenarioKind
    deleted_source_ids: tuple[str, ...]
    event_ids: tuple[str, ...] = ()
    query_ids: tuple[str, ...] = ()
    reinsert_source_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "kind": self.kind.value,
            "deleted_source_ids": list(self.deleted_source_ids),
            "event_ids": list(self.event_ids),
            "query_ids": list(self.query_ids),
            "reinsert_source_ids": list(self.reinsert_source_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DeletionScenarioManifest:
    """Serializable manifest binding scenarios to one frozen history."""

    manifest_id: str
    dataset: str
    history_id: str
    source_hash: str
    seed: int
    scenarios: tuple[DeletionScenario, ...]
    generator_version: str = "q01-scenarios-v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "dataset": self.dataset,
            "history_id": self.history_id,
            "source_hash": self.source_hash,
            "seed": self.seed,
            "generator_version": self.generator_version,
            "metadata": dict(self.metadata),
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target


def _source_hash(events: Sequence[SourceEvent]) -> str:
    return _stable_hash(
        [
            {
                "event_id": event.event_id,
                "source_id": event.source_id,
                "content": event.content,
                "timestamp": event.timestamp,
                "metadata": dict(event.metadata),
            }
            for event in events
        ]
    )


def _scenario(kind: DeletionScenarioKind, events: Sequence[SourceEvent], source_ids: Iterable[str], **metadata: Any) -> DeletionScenario:
    selected = tuple(dict.fromkeys(source_ids))
    event_ids = tuple(event.event_id for event in events if event.source_id in selected)
    identity = _stable_hash({"kind": kind.value, "source_ids": selected, "metadata": metadata})[:16]
    return DeletionScenario(
        scenario_id=f"delete:{kind.value}:{identity}",
        kind=kind,
        deleted_source_ids=selected,
        event_ids=event_ids,
        reinsert_source_ids=tuple(str(item) for item in metadata.get("reinsert_source_ids", ())),
        metadata=metadata,
    )


def generate_deletion_scenarios(
    events: Sequence[SourceEvent],
    *,
    dataset: str = "controlled",
    history_id: str = "history-0",
    seed: int = 0,
    max_single: int = 8,
    top_k: int = 3,
    trace: Sequence[BuildStep] = (),
    query_source_ids: Mapping[str, Sequence[str]] | None = None,
) -> DeletionScenarioManifest:
    """Create deterministic deletion masks from one source history.

    The generator uses only source order/content and an optional build trace.
    It never invents a post-deletion answer or treats a QA label as a state
    oracle.  Optional trace-aware scenarios are omitted when no corresponding
    dependency is observable.
    """

    if not events:
        raise ValueError("cannot generate deletion scenarios for an empty history")
    by_id = {event.source_id: event for event in events}
    ordered_ids = [event.source_id for event in events]
    scenarios: list[DeletionScenario] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    def add(kind: DeletionScenarioKind, source_ids: Iterable[str], **metadata: Any) -> None:
        selected = tuple(source_id for source_id in dict.fromkeys(source_ids) if source_id in by_id)
        if not selected:
            return
        key = (kind.value, selected)
        if key in seen:
            return
        seen.add(key)
        scenarios.append(_scenario(kind, events, selected, **metadata))

    for source_id in ordered_ids[: max(0, max_single)]:
        add(DeletionScenarioKind.SINGLE, [source_id], ordinal=ordered_ids.index(source_id))

    groups: dict[str, list[str]] = {}
    for event in events:
        groups.setdefault(_norm(event.content), []).append(event.source_id)
    for content_key, source_ids in sorted(groups.items()):
        if len(source_ids) > 1:
            add(DeletionScenarioKind.SHARED_SUPPORT, [source_ids[0]],
                support_group=source_ids, normalized_content=content_key)
            break
    unique_ids = [source_ids[0] for source_ids in groups.values() if len(source_ids) == 1]
    if unique_ids:
        add(DeletionScenarioKind.ISOLATED, [unique_ids[0]])

    positions = {
        DeletionScenarioKind.EARLY: 0,
        DeletionScenarioKind.MIDDLE: max(0, (len(events) - 1) // 2),
        DeletionScenarioKind.LATE: len(events) - 1,
    }
    for kind, position in positions.items():
        add(kind, [ordered_ids[position]], ordinal=position)
    if len(events) > 1:
        add(DeletionScenarioKind.BATCH, [ordered_ids[0], ordered_ids[-1]], batch_size=2)
    # An event ordinal is not a retrieval rank.  Emit this mechanism only
    # when a real query/ranking trace provides the ordered source IDs.
    ranked_traces: list[tuple[str, Sequence[str], str]] = []
    if query_source_ids:
        ranked_traces.extend((query_id, ranked, "query") for query_id, ranked in sorted(query_source_ids.items()))
    for step in trace:
        if step.candidate_ids:
            ranked_traces.append((f"trace:{step.index}", step.candidate_ids, "build"))
    if ranked_traces:
        for query_id, ranked_source_ids, trace_kind in ranked_traces:
            ranked = [sid for sid in ranked_source_ids if sid in by_id]
            if ranked:
                # A short trace still identifies its last observed boundary;
                # it must never be fabricated from event position.
                boundary = min(top_k - 1, len(ranked) - 1)
                add(
                    DeletionScenarioKind.TOP_K_BOUNDARY,
                    [ranked[boundary]],
                    top_k=top_k,
                    rank=boundary,
                    query_id=query_id,
                    trace_kind=trace_kind,
                    retrieval_trace=True,
                )

    for step in trace:
        if step.operation.lower() == "noop":
            add(DeletionScenarioKind.NOOP, [step.source_id], step_index=step.index)
        candidate_only = set(step.candidate_ids) - set(step.output_ids)
        if candidate_only:
            source_id = next((item for item in ordered_ids if item in candidate_only), None)
            if source_id:
                add(DeletionScenarioKind.RETRIEVAL_ONLY, [source_id], step_index=step.index)
        linked = set(step.read_set) - {step.source_id}
        if linked:
            source_id = next((item for item in ordered_ids if item in linked), None)
            if source_id:
                add(DeletionScenarioKind.LINKED_NEIGHBOR, [source_id], step_index=step.index)

    if ordered_ids:
        source_id = ordered_ids[0]
        add(
            DeletionScenarioKind.REINSERT,
            [source_id],
            reinsert_source_ids=[source_id],
            reinsert_after_delete=True,
        )

    # Dataset QA evidence is an overlay for scenario selection.  It is never
    # treated as the post-deletion state oracle; M0 still supplies that state.
    for query_id, source_ids in sorted((query_source_ids or {}).items()):
        add(
            DeletionScenarioKind.SINGLE,
            source_ids,
            query_id=query_id,
            evidence_overlay=True,
        )

    manifest_payload = {
        "dataset": dataset,
        "history_id": history_id,
        "source_hash": _source_hash(events),
        "seed": seed,
        "scenarios": [scenario.to_dict() for scenario in scenarios],
    }
    manifest_id = f"scenario-manifest:{_stable_hash(manifest_payload)[:16]}"
    return DeletionScenarioManifest(
        manifest_id=manifest_id,
        dataset=dataset,
        history_id=history_id,
        source_hash=_source_hash(events),
        seed=seed,
        scenarios=tuple(scenarios),
    )


def source_events_from_instance(instance: Any) -> list[SourceEvent]:
    """Adapt a ``ResearchInstance`` to the scenario generator's event model."""

    return [
        SourceEvent(
            event_id=turn.source_id,
            source_id=turn.source_id,
            content=turn.text,
            timestamp=turn.timestamp,
            metadata={
                "dataset": getattr(instance, "dataset", "unknown"),
                "instance_id": getattr(instance, "instance_id", "unknown"),
                "speaker": turn.speaker,
                "session_id": turn.metadata.get("session_id"),
            },
        )
        for turn in getattr(instance, "turns", [])
    ]


__all__ = [
    "DeletionScenarioKind",
    "DeletionScenario",
    "DeletionScenarioManifest",
    "generate_deletion_scenarios",
    "source_events_from_instance",
]
