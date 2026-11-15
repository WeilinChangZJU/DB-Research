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
        # Versioned object nodes retain their logical object identity for
        # deletion roots and legacy callers.  This is an alias index rather
        # than an extra graph edge, so it cannot introduce a decision/object
        # self-cycle in the execution graph.
        self._version_aliases: dict[str, str] = {}

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
        object_versions = {
            str(key): str(value)
            for key, value in (event.metadata.get("object_versions", {}) or {}).items()
        }
        candidate_versions = {
            str(key): str(value)
            for key, value in (event.metadata.get("candidate_versions", {}) or {}).items()
        }
        event_kind = event.event_type.value
        # READ/SELECT events consume object versions.  They must point from
        # the object into the decision; treating every ``object_ids`` field as
        # a write output creates a false decision -> object edge and, together
        # with the selected-object edge below, a pseudo-cycle for every read.
        read_event = event_kind in {"read", "select"}

        def versioned(value: str, versions: Mapping[str, str]) -> str:
            normalized = self._normalize_object_id(value)
            version = versions.get(str(value)) or versions.get(normalized)
            if not version:
                return normalized
            versioned_id = f"{normalized}@{version}"
            self._version_aliases[versioned_id] = normalized
            return versioned_id

        def candidate_node(value: str) -> str:
            """Resolve a candidate to the namespaced version node."""
            return versioned(value, candidate_versions)

        if operation_node != decision_id:
            self.add(DependencyEdge(operation_node, decision_id, "operation_decision", event.step, event.query_id))
        current_source_id = str(event.metadata.get("current_source_id", ""))
        content_support_ids = {
            str(value) for value in event.metadata.get("content_support_ids", [])
        }
        for source_id in event.source_ids:
            source_text = str(source_id)
            if source_text in content_support_ids and source_text != current_source_id:
                edge_kind = "content_support"
            elif source_text == current_source_id:
                edge_kind = "input_source"
            else:
                edge_kind = "source_operation"
            self.add(DependencyEdge(source_text, operation_node, edge_kind, event.step, event.query_id))
        for object_id in event.object_ids:
            normalized_object = self._normalize_object_id(object_id)
            versioned_object = versioned(object_id, object_versions)
            if read_event:
                self.add(DependencyEdge(versioned_object, decision_id, "read_object", event.step, event.query_id))
            else:
                self.add(DependencyEdge(decision_id, normalized_object, "decision_object", event.step, event.query_id))
                if versioned_object != normalized_object:
                    self.add(DependencyEdge(decision_id, versioned_object, "decision_object_version", event.step, event.query_id))
        output_ids = {self._normalize_object_id(value) for value in event.object_ids}
        for candidate_id in event.candidate_ids:
            normalized_candidate = self._normalize_object_id(candidate_id)
            has_version = bool(
                candidate_versions.get(str(candidate_id))
                or candidate_versions.get(normalized_candidate)
            )
            # Legacy write/update traces sometimes repeat the output ID in
            # candidate_ids without recording the prior object version.  An
            # unversioned input edge would create a false decision/object
            # cycle, so preserve the edge only when a prior version is
            # available (new traces) or when the candidate is distinct from
            # the materialized output.
            if event_kind in {"write", "update"} and normalized_candidate in output_ids and not has_version:
                continue
            self.add(DependencyEdge(candidate_node(candidate_id), decision_id, "candidate_decision", event.step, event.query_id))
        # ``selected_ids`` are inputs for read/select/query events, but the
        # wrapper also uses the field as a compatibility output list on write
        # and update events.  Treating those outputs as inputs creates an
        # unversioned decision <-> object cycle and is not an execution DAG.
        if event_kind not in {"write", "update"}:
            for selected_id in event.selected_ids:
                if read_event and self._normalize_object_id(selected_id) in {
                    self._normalize_object_id(value) for value in event.object_ids
                }:
                    # The object_ids edge above already captures this read;
                    # avoid a second edge with a competing semantic kind.
                    continue
                self.add(DependencyEdge(candidate_node(selected_id), decision_id, "selected_decision", event.step, event.query_id))
        for control_id in event.control_dependencies:
            normalized_control = self._normalize_object_id(control_id)
            has_version = bool(
                candidate_versions.get(str(control_id))
                or candidate_versions.get(normalized_control)
            )
            if event_kind in {"write", "update"} and normalized_control in output_ids and not has_version:
                continue
            self.add(DependencyEdge(candidate_node(control_id), decision_id, "control_dependency", event.step, event.query_id))

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

        root_values = [str(root) for root in roots]
        affected: set[str] = set(root_values)
        queue = deque(root_values)
        while queue:
            current = queue.popleft()
            # A caller may still provide a logical object ID (for example
            # ``n1``) while new traces use ``n1@<version>`` nodes.  Expand
            # all known versions of that logical ID without materialising a
            # graph alias edge.  The latter would make an in-place update
            # appear cyclic to dependency consumers.
            candidate_nodes = {current}
            candidate_nodes.update(
                versioned
                for versioned, logical in self._version_aliases.items()
                if logical == current
            )
            downstream_ids: set[str] = set()
            for node in candidate_nodes:
                downstream_ids.update(self._forward.get(node, ()))
            for downstream in sorted(downstream_ids):
                if downstream not in affected:
                    affected.add(downstream)
                    queue.append(downstream)
                logical = self._version_aliases.get(downstream)
                if logical and logical not in affected:
                    # Preserve the legacy logical identity in the closure;
                    # this keeps existing maintenance code and tests working
                    # while retaining version-level graph nodes.
                    affected.add(logical)
                    queue.append(logical)
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
