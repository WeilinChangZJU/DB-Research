"""Dependency graph and affected-closure utilities for deletion maintenance."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .state import ProvenanceEvent


@dataclass(frozen=True)
class DependencyEdge:
    """One observed dependency from an upstream identity to a downstream one."""

    upstream: str
    downstream: str
    kind: str
    step: int | None = None
    query_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "upstream": self.upstream,
            "downstream": self.downstream,
            "kind": self.kind,
            "step": self.step,
            "query_id": self.query_id,
            "metadata": dict(self.metadata),
        }


class DependencyGraph:
    """Forward/reverse graph reconstructed from provenance events."""

    def __init__(self, *, complete: bool = False):
        self.complete = complete
        self.edges: list[DependencyEdge] = []
        self._forward: dict[str, set[str]] = defaultdict(set)
        self._reverse: dict[str, set[str]] = defaultdict(set)

    def add(self, edge: DependencyEdge) -> None:
        key = (edge.upstream, edge.downstream, edge.kind, edge.step, edge.query_id)
        if any((item.upstream, item.downstream, item.kind, item.step, item.query_id) == key for item in self.edges):
            return
        self.edges.append(edge)
        self._forward[edge.upstream].add(edge.downstream)
        self._reverse[edge.downstream].add(edge.upstream)

    def add_event(self, event: ProvenanceEvent) -> None:
        # A provenance event is the observation boundary for one operation.
        # Keep the event/operation identity explicit so a read in one event
        # can be connected to a later write even when the write has no direct
        # source edge of its own.  Older traces do not carry operation_id;
        # their deterministic event-based fallback remains compatible.
        operation_id = str(event.metadata.get("operation_id") or event.event_id)
        decision_id = str(
            event.metadata.get("decision_id")
            or f"decision:{event.step}:{event.event_type.value}"
        )
        operation_node = f"operation:{operation_id}"
        if operation_node != decision_id:
            self.add(DependencyEdge(operation_node, decision_id, "operation_decision", event.step, event.query_id))
        for source_id in event.source_ids:
            self.add(DependencyEdge(source_id, operation_node, "source_operation", event.step, event.query_id))
        for object_id in event.object_ids:
            self.add(DependencyEdge(decision_id, self._normalize_object_id(object_id), "decision_object", event.step, event.query_id))
        for candidate_id in event.candidate_ids:
            self.add(DependencyEdge(self._normalize_object_id(candidate_id), decision_id, "candidate_decision", event.step, event.query_id))
        for selected_id in event.selected_ids:
            self.add(DependencyEdge(self._normalize_object_id(selected_id), decision_id, "selected_decision", event.step, event.query_id))
        for control_id in event.control_dependencies:
            self.add(DependencyEdge(self._normalize_object_id(control_id), decision_id, "control_dependency", event.step, event.query_id))

    @staticmethod
    def _normalize_object_id(value: str) -> str:
        """Accept both legacy ``note:<id>`` and raw object IDs.

        The wrapper historically emitted ``note:<id>`` for candidates and
        raw ``<id>`` for writes, which disconnected read and write nodes in
        the graph.  The graph is deliberately tolerant so old traces remain
        useful while new traces use one identity namespace.
        """

        text = str(value)
        return text.removeprefix("note:")

    @classmethod
    def from_events(cls, events: Sequence[ProvenanceEvent], *, complete: bool = False) -> "DependencyGraph":
        graph = cls(complete=complete)
        for event in events:
            graph.add_event(event)
        return graph

    @property
    def forward(self) -> dict[str, list[str]]:
        return {key: sorted(value) for key, value in sorted(self._forward.items())}

    @property
    def reverse(self) -> dict[str, list[str]]:
        return {key: sorted(value) for key, value in sorted(self._reverse.items())}

    def affected_closure(self, roots: Iterable[str]) -> set[str]:
        """Return all downstream identities reachable from deletion roots."""

        affected: set[str] = set(roots)
        queue = deque(roots)
        while queue:
            current = queue.popleft()
            for downstream in sorted(self._forward.get(current, ())):
                if downstream not in affected:
                    affected.add(downstream)
                    queue.append(downstream)
        return affected

    def affected_steps(self, roots: Iterable[str]) -> list[int]:
        closure = self.affected_closure(roots)
        return sorted({
            edge.step
            for edge in self.edges
            if edge.step is not None and (edge.upstream in closure or edge.downstream in closure)
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "edges": [edge.to_dict() for edge in sorted(self.edges, key=lambda item: (item.step or -1, item.upstream, item.downstream, item.kind))],
            "forward": self.forward,
            "reverse": self.reverse,
        }


__all__ = ["DependencyEdge", "DependencyGraph"]
