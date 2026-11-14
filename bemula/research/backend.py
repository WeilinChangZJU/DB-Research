"""Research-facing memory backend contracts and a deterministic reference backend.

The public Bemula ``MemoryAdapter`` intentionally exposes only ``reset``,
``observe`` and ``query``.  Those methods are enough for black-box benchmark
scores, but they do not let a research protocol inspect source provenance,
deletion state, or retrieval evidence.  This module adds an optional layer for
backends that can expose those capabilities without changing existing
baseline adapters.

Backends may implement only the required observe/query methods.  Optional
operations raise ``NotImplementedError`` by default, making unsupported
mechanism-level claims explicit instead of silently fabricating evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bemula.baselines.base import MemoryAdapter
from bemula.schemas import Turn


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SourceRecord(BaseModel):
    """A stable source unit used for provenance-aware memory experiments."""

    model_config = ConfigDict(extra="allow")

    source_id: str
    content: str
    kind: str = "turn"
    speaker: str | None = None
    timestamp: str | None = None
    source_hash: str | None = None
    span: dict[str, int] | None = None
    version: int = 1
    support_set: list[str] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def fill_hash(self) -> "SourceRecord":
        if not self.source_hash:
            self.source_hash = _canonical_hash(
                {
                    "source_id": self.source_id,
                    "content": self.content,
                    "kind": self.kind,
                    "speaker": self.speaker,
                    "timestamp": self.timestamp,
                    "span": self.span,
                    "version": self.version,
                    "support_set": self.support_set,
                    "derived_from": self.derived_from,
                    "metadata": self.metadata,
                }
            )
        return self


class QueryEvidence(BaseModel):
    """Evidence returned by a provenance-aware query."""

    source_ids: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    read_set: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """Answer plus optional evidence.  ``answer`` remains the adapter output."""

    answer: str
    evidence: QueryEvidence = Field(default_factory=QueryEvidence)
    latency_ms: float | None = None
    usage: dict[str, Any] = Field(default_factory=dict)


class ForgetResult(BaseModel):
    """Audit record for a requested deletion or logical forget operation."""

    requested_ids: list[str] = Field(default_factory=list)
    removed_ids: list[str] = Field(default_factory=list)
    missing_ids: list[str] = Field(default_factory=list)
    operation: str = "forget"
    timestamp: str = Field(default_factory=_utc_now)
    details: dict[str, Any] = Field(default_factory=dict)


class MemorySnapshot(BaseModel):
    """Serializable state snapshot suitable for before/after comparisons."""

    state_id: str
    records: list[SourceRecord] = Field(default_factory=list)
    tombstones: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utc_now)
    state_hash: str | None = None

    @model_validator(mode="after")
    def fill_state_hash(self) -> "MemorySnapshot":
        if not self.state_hash:
            self.state_hash = _canonical_hash(
                {
                    "records": [r.model_dump(mode="json") for r in self.records],
                    "tombstones": sorted(self.tombstones),
                }
            )
        return self


class MemoryDiff(BaseModel):
    """Deterministic diff between two snapshots."""

    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    before_hash: str | None = None
    after_hash: str | None = None


class MemoryBackend(ABC):
    """Research backend contract layered under a Bemula adapter.

    ``observe`` and ``query`` are the only required operations.  The explicit
    mutation and introspection methods are intentionally optional: a backend
    that does not support them must fail loudly when a mechanism-level
    protocol requests them.
    """

    name = "research-backend"

    @abstractmethod
    def reset(self) -> None:
        """Clear the current user/instance state."""

    @abstractmethod
    def observe(self, turn: Turn, *, source_id: str | None = None) -> SourceRecord:
        """Ingest one conversation turn and return its source record."""

    @abstractmethod
    def query(self, question: str) -> QueryResult:
        """Answer a question without writing the probe back into memory."""

    def write(self, content: str, *, source_id: str | None = None, **metadata: Any) -> SourceRecord:
        raise NotImplementedError(f"{self.name} does not expose write()")

    def update(self, source_id: str, content: str, **metadata: Any) -> SourceRecord:
        raise NotImplementedError(f"{self.name} does not expose update()")

    def forget(self, source_ids: Sequence[str], *, reason: str | None = None) -> ForgetResult:
        raise NotImplementedError(f"{self.name} does not expose forget()")

    def delete(self, source_ids: Sequence[str], *, reason: str | None = None) -> ForgetResult:
        raise NotImplementedError(f"{self.name} does not expose delete()")

    def snapshot(self) -> MemorySnapshot:
        raise NotImplementedError(f"{self.name} does not expose snapshot()")

    def diff(self, before: MemorySnapshot, after: MemorySnapshot) -> MemoryDiff:
        before_map = {r.source_id: r.source_hash for r in before.records}
        after_map = {r.source_id: r.source_hash for r in after.records}
        before_ids, after_ids = set(before_map), set(after_map)
        return MemoryDiff(
            added=sorted(after_ids - before_ids),
            removed=sorted(before_ids - after_ids),
            updated=sorted(i for i in before_ids & after_ids if before_map[i] != after_map[i]),
            unchanged=sorted(i for i in before_ids & after_ids if before_map[i] == after_map[i]),
            before_hash=before.state_hash,
            after_hash=after.state_hash,
        )

    def introspect(self) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} does not expose introspect()")

    def explain(self, question: str) -> QueryEvidence:
        result = self.query(question)
        return result.evidence


class ResearchAdapter(MemoryAdapter):
    """Bridge a :class:`MemoryBackend` into Bemula's black-box adapter API."""

    name = "research"

    def __init__(self, backend: MemoryBackend):
        self.backend = backend

    def reset(self) -> None:
        self.backend.reset()

    def observe(self, turn: Turn) -> None:
        self.backend.observe(turn)

    def query(self, question: str) -> str:
        result = self.backend.query(question)
        return result.answer if isinstance(result, QueryResult) else str(result)

    def write(self, content: str, **metadata: Any) -> SourceRecord:
        return self.backend.write(content, **metadata)

    def update(self, source_id: str, content: str, **metadata: Any) -> SourceRecord:
        return self.backend.update(source_id, content, **metadata)

    def forget(self, source_ids: Sequence[str], **kwargs: Any) -> ForgetResult:
        return self.backend.forget(source_ids, **kwargs)

    def delete(self, source_ids: Sequence[str], **kwargs: Any) -> ForgetResult:
        return self.backend.delete(source_ids, **kwargs)

    def snapshot(self) -> MemorySnapshot:
        return self.backend.snapshot()

    def introspect(self) -> dict[str, Any]:
        return self.backend.introspect()


class InMemoryBackend(MemoryBackend):
    """Small deterministic backend used for offline protocol tests.

    It is deliberately lexical rather than semantic.  The implementation is
    a reference for lifecycle/provenance behavior, not a quality baseline.
    """

    name = "in-memory-reference"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._records: dict[str, SourceRecord] = {}
        self._tombstones: set[str] = set()
        self._counter = 0
        self._events: list[dict[str, Any]] = []

    def _next_id(self, prefix: str = "turn") -> str:
        self._counter += 1
        return f"{prefix}:{self._counter:06d}"

    def write(self, content: str, *, source_id: str | None = None, **metadata: Any) -> SourceRecord:
        sid = source_id or self._next_id("source")
        record = SourceRecord(source_id=sid, content=content, metadata=metadata)
        self._records[sid] = record
        self._tombstones.discard(sid)
        self._events.append({"event": "write", "source_id": sid, "hash": record.source_hash})
        return record

    def observe(self, turn: Turn, *, source_id: str | None = None) -> SourceRecord:
        sid = source_id or self._next_id("turn")
        record = SourceRecord(
            source_id=sid,
            content=turn.text,
            kind="turn",
            speaker=turn.speaker,
            timestamp=turn.timestamp,
            support_set=list(turn.linked_fact_ids),
        )
        self._records[sid] = record
        self._tombstones.discard(sid)
        self._events.append({"event": "observe", "source_id": sid, "hash": record.source_hash})
        return record

    def update(self, source_id: str, content: str, **metadata: Any) -> SourceRecord:
        previous = self._records.get(source_id)
        if previous is None:
            raise KeyError(f"unknown source_id: {source_id}")
        record = previous.model_copy(
            update={
                "content": content,
                "version": previous.version + 1,
                "metadata": {**previous.metadata, **metadata},
                "source_hash": None,
            }
        )
        self._records[source_id] = record
        self._events.append({"event": "update", "source_id": source_id, "hash": record.source_hash})
        return record

    def forget(self, source_ids: Sequence[str], *, reason: str | None = None) -> ForgetResult:
        requested = list(dict.fromkeys(source_ids))
        removed = [sid for sid in requested if sid in self._records]
        missing = [sid for sid in requested if sid not in self._records]
        for sid in removed:
            del self._records[sid]
            self._tombstones.add(sid)
        self._events.append(
            {"event": "forget", "requested_ids": requested, "removed_ids": removed, "reason": reason}
        )
        return ForgetResult(
            requested_ids=requested,
            removed_ids=removed,
            missing_ids=missing,
            operation="forget",
            details={"reason": reason} if reason else {},
        )

    def delete(self, source_ids: Sequence[str], *, reason: str | None = None) -> ForgetResult:
        result = self.forget(source_ids, reason=reason)
        result.operation = "delete"
        return result

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in re.findall(r"[\w]+", text.lower()) if len(t) > 1}

    def query(self, question: str) -> QueryResult:
        q = self._tokens(question)
        scored: list[tuple[float, SourceRecord]] = []
        for record in self._records.values():
            overlap = len(q & self._tokens(record.content))
            if overlap:
                score = overlap / max(len(q), 1)
                scored.append((score, record))
        scored.sort(key=lambda pair: (-pair[0], pair[1].source_id))
        selected = scored[:5]
        answer = "\n".join(record.content for _, record in selected)
        evidence = QueryEvidence(
            source_ids=[record.source_id for _, record in selected],
            scores={record.source_id: score for score, record in selected},
            read_set=[record.source_id for record in self._records.values()],
        )
        return QueryResult(answer=answer, evidence=evidence)

    def snapshot(self) -> MemorySnapshot:
        records = [self._records[sid] for sid in sorted(self._records)]
        return MemorySnapshot(
            state_id=f"state:{len(self._events):06d}",
            records=records,
            tombstones=sorted(self._tombstones),
            metadata={"backend": self.name, "event_count": len(self._events)},
        )

    def introspect(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "active_records": len(self._records),
            "tombstones": sorted(self._tombstones),
            "event_count": len(self._events),
            "supports": {
                "write": True,
                "update": True,
                "forget": True,
                "delete": True,
                "snapshot": True,
                "explain": True,
            },
        }

    def events(self) -> list[dict[str, Any]]:
        """Return a copy of the operation log for tests and research traces."""
        return list(self._events)

