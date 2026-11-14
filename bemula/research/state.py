"""Canonical state and provenance models for Q01 experiments.

The original Bemula evaluator compares answers.  Q01 additionally needs a
state-level oracle: a rebuilt state and an incrementally maintained state must
be compared without treating storage addresses, UUIDs, or wall-clock fields as
semantic differences.  This module provides a small serialisable schema and a
deterministic comparator.  It intentionally does not prescribe a memory
implementation or claim that an LLM operation is deterministic.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .backend import MemorySnapshot, SourceRecord


DEFAULT_IGNORED_METADATA_KEYS = frozenset(
    {
        "created_at",
        "updated_at",
        "wall_clock",
        "wall_clock_time",
        "runtime_ns",
        "physical_id",
        "storage_key",
        # Run-local identifiers and arm labels are audit metadata, not
        # canonical visible memory semantics.
        "state_id",
        "event_count",
        "start_index",
        "arm",
        "maintenance_arm",
        "run_id",
        "trace_path",
    }
)


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ProvenanceEventType(str, Enum):
    """Lifecycle events that can affect a canonical state."""

    INGEST = "ingest"
    READ = "read"
    CANDIDATE = "candidate"
    SELECT = "select"
    WRITE = "write"
    UPDATE = "update"
    NOOP = "noop"
    FORGET = "forget"
    PUBLISH = "publish"
    PURGE = "purge"
    LLM_CALL = "llm_call"
    DECISION = "decision"
    INDEX_UPDATE = "index_update"


class SourceProvenance(BaseModel):
    """Stable identity and lifecycle metadata for one external source."""

    model_config = ConfigDict(extra="allow")

    source_id: str
    dataset: str = "unknown"
    instance_id: str = "unknown"
    source_type: str = "turn"
    content_hash: str
    ordinal: int | None = None
    session_id: str | None = None
    timestamp: str | None = None
    parent_source_ids: list[str] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)
    revoked: bool = False
    version: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_record(
        cls,
        record: SourceRecord,
        *,
        dataset: str = "unknown",
        instance_id: str = "unknown",
        ordinal: int | None = None,
    ) -> "SourceProvenance":
        """Create provenance without copying the raw source text."""

        metadata = dict(record.metadata)
        return cls(
            source_id=record.source_id,
            dataset=dataset,
            instance_id=instance_id,
            source_type=record.kind,
            content_hash=record.source_hash or _json_hash(record.content),
            ordinal=ordinal,
            session_id=metadata.pop("session_id", None),
            timestamp=record.timestamp,
            parent_source_ids=list(record.support_set),
            derived_from=list(record.derived_from),
            version=record.version,
            metadata=metadata,
        )


class ProvenanceEvent(BaseModel):
    """Auditable operation record for a write/read/decision boundary."""

    model_config = ConfigDict(extra="allow")

    event_id: str
    event_type: ProvenanceEventType
    step: int
    source_ids: list[str] = Field(default_factory=list)
    object_ids: list[str] = Field(default_factory=list)
    query_id: str | None = None
    input_hashes: list[str] = Field(default_factory=list)
    output_hashes: list[str] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)
    selected_ids: list[str] = Field(default_factory=list)
    control_dependencies: list[str] = Field(default_factory=list)
    decision: str | None = None
    version: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def event_hash(self) -> str:
        return _json_hash(self.model_dump(mode="json"))


class CanonicalArtifact(BaseModel):
    """A visible or derived memory object.

    ``object_id`` may be a physical storage identifier.  ``canonical_id`` is
    optional; when absent, the comparator derives one from semantic fields.
    """

    model_config = ConfigDict(extra="allow")

    object_id: str
    object_type: str = "memory"
    content: str = ""
    content_hash: str | None = None
    canonical_id: str | None = None
    version: int = 1
    support_set: list[str] = Field(default_factory=list)
    parent_ids: list[str] = Field(default_factory=list)
    embedding_hash: str | None = None
    relation_ids: list[str] = Field(default_factory=list)
    visible: bool = True
    status: str = "active"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def fill_content_hash(self) -> "CanonicalArtifact":
        if not self.content_hash:
            self.content_hash = _json_hash(self.content)
        return self

    @property
    def semantic_id(self) -> str:
        if self.canonical_id:
            return self.canonical_id
        return _json_hash(
            {
                "object_type": self.object_type,
                "content_hash": self.content_hash,
                "support_set": sorted(set(self.support_set)),
                "parent_ids": sorted(set(self.parent_ids)),
                "version": self.version,
            }
        )


class CanonicalRelation(BaseModel):
    """A semantic relation between canonical objects or sources."""

    model_config = ConfigDict(extra="allow")

    relation_id: str
    relation_type: str
    subject_id: str
    object_id: str
    support_set: list[str] = Field(default_factory=list)
    visible: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.relation_type,
            self.subject_id,
            self.object_id,
            tuple(sorted(set(self.support_set))),
            self.visible,
        )


class CanonicalQuery(BaseModel):
    """Historical query/decision state needed to detect new dependencies."""

    model_config = ConfigDict(extra="allow")

    query_id: str
    query_text_hash: str | None = None
    query_kind: str = "retrieval"
    visible_source_ids: list[str] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)
    selected_ids: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    top_k: int | None = None
    tie_break: str | None = None
    control_dependencies: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def semantic_key(self) -> str:
        return self.query_id


class CanonicalState(BaseModel):
    """Serializable state oracle input for full rebuild vs incremental runs."""

    model_config = ConfigDict(extra="allow")

    schema_version: int = 1
    state_id: str = ""
    sources: list[SourceProvenance] = Field(default_factory=list)
    artifacts: list[CanonicalArtifact] = Field(default_factory=list)
    relations: list[CanonicalRelation] = Field(default_factory=list)
    queries: list[CanonicalQuery] = Field(default_factory=list)
    provenance_events: list[ProvenanceEvent] = Field(default_factory=list)
    tombstones: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: MemorySnapshot,
        *,
        dataset: str = "unknown",
        instance_id: str = "unknown",
        source_records: Sequence[SourceProvenance] | None = None,
    ) -> "CanonicalState":
        """Lift the existing backend snapshot into the Q01 schema."""

        sources = list(source_records or [])
        if not sources:
            sources = [
                SourceProvenance.from_record(
                    record,
                    dataset=dataset,
                    instance_id=instance_id,
                    ordinal=index,
                )
                for index, record in enumerate(snapshot.records)
            ]
        artifacts = [
            CanonicalArtifact(
                object_id=record.source_id,
                object_type=record.kind,
                content=record.content,
                content_hash=record.source_hash,
                version=record.version,
                support_set=list(record.support_set),
                parent_ids=list(record.derived_from),
                visible=True,
                metadata=dict(record.metadata),
            )
            for record in snapshot.records
        ]
        return cls(
            state_id=snapshot.state_id,
            sources=sources,
            artifacts=artifacts,
            tombstones=list(snapshot.tombstones),
            metadata={"backend": snapshot.metadata.get("backend", "unknown"), **snapshot.metadata},
        )

    def _artifact_semantic_ids(self) -> dict[str, str]:
        """Resolve artifact references without depending on storage IDs."""

        by_object = {artifact.object_id: artifact for artifact in self.artifacts}
        resolved: dict[str, str] = {}
        visiting: set[str] = set()

        def resolve(object_id: str) -> str:
            if object_id in resolved:
                return resolved[object_id]
            artifact = by_object.get(object_id)
            if artifact is None:
                return object_id
            if artifact.canonical_id:
                resolved[object_id] = artifact.canonical_id
                return artifact.canonical_id
            if object_id in visiting:
                # Cyclic parent metadata is invalid for a strict DAG, but the
                # comparator should remain deterministic and report the cycle
                # as part of the resulting semantic ID rather than recurse.
                return f"cycle:{object_id}"
            visiting.add(object_id)
            parent_ids = sorted(resolve(parent_id) for parent_id in artifact.parent_ids)
            semantic_id = _json_hash(
                {
                    "object_type": artifact.object_type,
                    "content_hash": artifact.content_hash,
                    "support_set": sorted(set(artifact.support_set)),
                    "parent_ids": parent_ids,
                    "version": artifact.version,
                }
            )
            visiting.remove(object_id)
            resolved[object_id] = semantic_id
            return semantic_id

        for object_id in sorted(by_object):
            resolve(object_id)
        return resolved

    def canonical_payload(self, *, ignored_metadata_keys: Iterable[str] = ()) -> dict[str, Any]:
        """Return a deterministic payload containing semantic state only."""

        ignored = set(DEFAULT_IGNORED_METADATA_KEYS) | set(ignored_metadata_keys)

        def clean(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    str(key): clean(item)
                    for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                    if str(key) not in ignored
                }
            if isinstance(value, (list, tuple, set)):
                return [clean(item) for item in value]
            return value

        source_payload = []
        for source in sorted(self.sources, key=lambda item: item.source_id):
            data = source.model_dump(mode="json")
            data.pop("metadata", None)
            data["metadata"] = clean(source.metadata)
            source_payload.append(data)

        artifact_semantic_ids = self._artifact_semantic_ids()

        def map_reference(reference: str) -> str:
            return artifact_semantic_ids.get(reference, reference)

        artifact_payload = []
        for artifact in sorted(self.artifacts, key=lambda item: artifact_semantic_ids[item.object_id]):
            data = artifact.model_dump(mode="json")
            data.pop("object_id", None)
            data.pop("relation_ids", None)
            data["semantic_id"] = artifact_semantic_ids[artifact.object_id]
            data["support_set"] = sorted(set(artifact.support_set))
            data["parent_ids"] = sorted(map_reference(parent_id) for parent_id in artifact.parent_ids)
            data["metadata"] = clean(artifact.metadata)
            artifact_payload.append(data)

        relation_payload = []
        for relation in sorted(self.relations, key=lambda item: item.semantic_key):
            data = relation.model_dump(mode="json")
            data.pop("relation_id", None)
            data["subject_id"] = map_reference(relation.subject_id)
            data["object_id"] = map_reference(relation.object_id)
            data["support_set"] = sorted(set(relation.support_set))
            data["metadata"] = clean(relation.metadata)
            relation_payload.append(data)

        query_payload = []
        for query in sorted(self.queries, key=lambda item: item.semantic_key):
            data = query.model_dump(mode="json")
            data["visible_source_ids"] = sorted(set(query.visible_source_ids))
            # Candidate order is part of retrieval semantics: it can affect a
            # tie-break or a downstream top-k selection.  Selected order is
            # preserved for the same reason.  References to physical artifact
            # IDs are normalized to semantic IDs.
            data["candidate_ids"] = [map_reference(item) for item in query.candidate_ids]
            data["selected_ids"] = [map_reference(item) for item in query.selected_ids]
            # Scores are keyed by physical object IDs in most retrievers.
            # Normalize those keys through the same semantic-ID mapping as
            # candidates and selections; otherwise UUID churn appears as a
            # state difference even when the ranked objects are identical.
            data["scores"] = {
                map_reference(str(key)): value
                for key, value in sorted(query.scores.items(), key=lambda item: str(item[0]))
            }
            data["control_dependencies"] = sorted(map_reference(item) for item in query.control_dependencies)
            data["metadata"] = clean(query.metadata)
            query_payload.append(data)

        artifact_ids = {artifact.object_id: artifact.semantic_id for artifact in self.artifacts}
        event_payload = []
        for event in sorted(self.provenance_events, key=lambda item: (item.step, item.event_type.value, item.query_id or "", item.event_id)):
            data = event.model_dump(mode="json")
            # Event IDs and object IDs are often allocated by a storage layer.
            # Compare events by their deterministic execution position and map
            # physical object IDs to the artifact semantic IDs above.
            data.pop("event_id", None)
            data["event_key"] = f"{event.step}:{event.event_type.value}:{event.query_id or ''}"
            data["object_ids"] = [artifact_ids.get(object_id, object_id) for object_id in event.object_ids]
            data["candidate_ids"] = [map_reference(item) for item in event.candidate_ids]
            data["selected_ids"] = [map_reference(item) for item in event.selected_ids]
            data["control_dependencies"] = [map_reference(item) for item in event.control_dependencies]
            event_payload.append(data)
        return {
            "schema_version": self.schema_version,
            "sources": source_payload,
            "artifacts": artifact_payload,
            "relations": relation_payload,
            "queries": query_payload,
            "provenance_events": event_payload,
            "tombstones": sorted(set(self.tombstones)),
            "metadata": clean(self.metadata),
        }

    @property
    def canonical_hash(self) -> str:
        return _json_hash(self.canonical_payload())

    @property
    def visible_state_hash(self) -> str:
        """Hash the declared visible memory/query state only.

        This deliberately excludes runtime metadata and audit history. It is
        the hash to use for the primary ``M_inc == M_ref`` visible-state claim.
        """
        payload = self.canonical_payload()
        return _json_hash({
            "schema_version": payload["schema_version"],
            "sources": payload["sources"],
            "artifacts": payload["artifacts"],
            "relations": payload["relations"],
            "queries": payload["queries"],
            "tombstones": payload["tombstones"],
        })

    @property
    def runtime_state_hash(self) -> str:
        """Hash visible state plus service/runtime metadata."""
        payload = self.canonical_payload()
        return _json_hash({
            "visible": {
                key: payload[key]
                for key in ("schema_version", "sources", "artifacts", "relations", "queries", "tombstones")
            },
            "metadata": payload["metadata"],
        })

    @property
    def audit_trace_hash(self) -> str:
        """Hash the provenance trace independently of state semantics."""
        payload = self.canonical_payload()
        return _json_hash({"provenance_events": payload["provenance_events"]})


class StateMismatchKind(str, Enum):
    """Categories used by :class:`CanonicalStateComparator`."""

    SOURCE = "source"
    ARTIFACT = "artifact"
    RELATION = "relation"
    QUERY = "query"
    PROVENANCE = "provenance"
    TOMBSTONE = "tombstone"
    METADATA = "metadata"


class StateMismatch(BaseModel):
    kind: StateMismatchKind
    key: str
    field: str | None = None
    left: Any = None
    right: Any = None


class CanonicalStateDiff(BaseModel):
    equivalent: bool
    left_hash: str
    right_hash: str
    mismatches: list[StateMismatch] = Field(default_factory=list)

    @property
    def mismatch_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for mismatch in self.mismatches:
            counts[mismatch.kind.value] = counts.get(mismatch.kind.value, 0) + 1
        return counts


class CanonicalStateComparator:
    """Compare states while preserving semantic source and dependency fields."""

    def __init__(
        self,
        *,
        ignored_metadata_keys: Iterable[str] = (),
        include_provenance: bool = False,
    ):
        self.ignored_metadata_keys = frozenset(ignored_metadata_keys)
        self.include_provenance = include_provenance

    def compare(self, left: CanonicalState, right: CanonicalState) -> CanonicalStateDiff:
        mismatches: list[StateMismatch] = []
        left_payload = left.canonical_payload(ignored_metadata_keys=self.ignored_metadata_keys)
        right_payload = right.canonical_payload(ignored_metadata_keys=self.ignored_metadata_keys)
        # Provenance is an audit trace, not part of the default visible-state
        # oracle.  M0 and an incremental arm are allowed to take different
        # valid execution paths while producing the same canonical state.
        if not self.include_provenance:
            left_payload["provenance_events"] = []
            right_payload["provenance_events"] = []

        def compare_maps(kind: StateMismatchKind, key: str, lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> None:
            for item_key in sorted(set(lhs) | set(rhs)):
                if item_key not in lhs or item_key not in rhs:
                    mismatches.append(StateMismatch(kind=kind, key=f"{key}:{item_key}", left=lhs.get(item_key), right=rhs.get(item_key)))
                elif lhs[item_key] != rhs[item_key]:
                    mismatches.append(StateMismatch(kind=kind, key=f"{key}:{item_key}", left=lhs[item_key], right=rhs[item_key]))

        def index(items: list[dict[str, Any]], key_field: str) -> dict[str, Any]:
            """Index a sequence without silently discarding duplicates."""

            grouped: dict[str, list[dict[str, Any]]] = {}
            for item in items:
                grouped.setdefault(str(item.get(key_field)), []).append(item)
            indexed: dict[str, Any] = {}
            for key, values in grouped.items():
                for ordinal, value in enumerate(values):
                    indexed[f"{key}#{ordinal}"] = value
            return indexed

        compare_maps(
            StateMismatchKind.SOURCE,
            "source",
            index(left_payload["sources"], "source_id"),
            index(right_payload["sources"], "source_id"),
        )
        compare_maps(
            StateMismatchKind.ARTIFACT,
            "artifact",
            index(left_payload["artifacts"], "semantic_id"),
            index(right_payload["artifacts"], "semantic_id"),
        )
        compare_maps(
            StateMismatchKind.RELATION,
            "relation",
            index(
                [{"relation_key": _json_hash(item), **item} for item in left_payload["relations"]],
                "relation_key",
            ),
            index(
                [{"relation_key": _json_hash(item), **item} for item in right_payload["relations"]],
                "relation_key",
            ),
        )
        compare_maps(
            StateMismatchKind.QUERY,
            "query",
            index(left_payload["queries"], "query_id"),
            index(right_payload["queries"], "query_id"),
        )
        compare_maps(
            StateMismatchKind.PROVENANCE,
            "event",
            {str(item.get("event_key")): item for item in left_payload["provenance_events"]},
            {str(item.get("event_key")): item for item in right_payload["provenance_events"]},
        )
        if left_payload["tombstones"] != right_payload["tombstones"]:
            mismatches.append(
                StateMismatch(
                    kind=StateMismatchKind.TOMBSTONE,
                    key="tombstones",
                    left=left_payload["tombstones"],
                    right=right_payload["tombstones"],
                )
            )
        if left_payload["metadata"] != right_payload["metadata"]:
            mismatches.append(
                StateMismatch(
                    kind=StateMismatchKind.METADATA,
                    key="metadata",
                    left=left_payload["metadata"],
                    right=right_payload["metadata"],
                )
            )
        return CanonicalStateDiff(
            equivalent=not mismatches,
            left_hash=_json_hash(left_payload),
            right_hash=_json_hash(right_payload),
            mismatches=mismatches,
        )


def compare_canonical_states(
    left: CanonicalState,
    right: CanonicalState,
    *,
    ignored_metadata_keys: Iterable[str] = (),
    include_provenance: bool = False,
) -> CanonicalStateDiff:
    """Convenience function for one-off canonical state comparisons."""

    return CanonicalStateComparator(
        ignored_metadata_keys=ignored_metadata_keys,
        include_provenance=include_provenance,
    ).compare(left, right)

