"""Dataset-neutral ResearchProtocol orchestration layer.

Bemula's original runner is intentionally specialized to its synthetic
control/forget benchmark.  ``ResearchProtocol`` provides a small, explicit
contract for real datasets and mechanism-aware studies while keeping the
original runner untouched.  It can run a control ingestion, an optional
forget/delete intervention, and query items while retaining state snapshots
and provenance evidence.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bemula.schemas import Grading, Turn

from .backend import (
    ForgetResult,
    MemoryBackend,
    MemoryDiff,
    MemorySnapshot,
    QueryResult,
    SourceRecord,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


class ProtocolPhase(str, Enum):
    CONTROL = "control"
    FORGET = "forget"
    RELEARN = "relearn"
    QUERY = "query"


class ResearchTurn(BaseModel):
    """Dataset-neutral conversation turn with a stable source identity."""

    model_config = ConfigDict(extra="allow")

    dataset: str
    instance_id: str
    source_id: str
    speaker: str = "user"
    text: str
    timestamp: str = ""
    source_hash: str | None = None
    linked_fact_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def fill_hash(self) -> "ResearchTurn":
        if not self.source_hash:
            self.source_hash = _hash_payload(
                {
                    "dataset": self.dataset,
                    "instance_id": self.instance_id,
                    "source_id": self.source_id,
                    "speaker": self.speaker,
                    "text": self.text,
                    "timestamp": self.timestamp,
                    "linked_fact_ids": self.linked_fact_ids,
                }
            )
        return self

    def to_turn(self) -> Turn:
        return Turn(
            speaker=self.speaker,
            text=self.text,
            timestamp=self.timestamp,
            linked_fact_ids=list(self.linked_fact_ids),
        )


class ResearchItem(BaseModel):
    """Dataset-neutral question and scoring metadata."""

    model_config = ConfigDict(extra="allow")

    dataset: str
    instance_id: str
    item_id: str
    question: str
    gold_answer: str | None = None
    grading: Grading = Grading.LLM_JUDGE
    suite: str = "recall"
    related_source_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchInstance(BaseModel):
    """Canonical intermediate representation shared by dataset adapters."""

    model_config = ConfigDict(extra="allow")

    dataset: str
    instance_id: str
    split: str = "unknown"
    turns: list[ResearchTurn] = Field(default_factory=list)
    control_turns: list[ResearchTurn] | None = None
    items: list[ResearchItem] = Field(default_factory=list)
    forget_source_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def source_ids(self) -> list[str]:
        return [t.source_id for t in self.turns]

    @property
    def content_hash(self) -> str:
        return _hash_payload(self.model_dump(mode="json"))


class ProtocolQuery(BaseModel):
    phase: ProtocolPhase
    item_id: str
    instance_id: str
    question: str
    answer: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = None
    usage: dict[str, Any] = Field(default_factory=dict)


class ProtocolResult(BaseModel):
    dataset: str
    instance_id: str
    protocol_version: str = "research-v1"
    started_at: str
    finished_at: str
    control_queries: list[ProtocolQuery] = Field(default_factory=list)
    forget_queries: list[ProtocolQuery] = Field(default_factory=list)
    relearn_queries: list[ProtocolQuery] = Field(default_factory=list)
    control_snapshot: MemorySnapshot | None = None
    before_forget_snapshot: MemorySnapshot | None = None
    after_forget_snapshot: MemorySnapshot | None = None
    forget_result: ForgetResult | None = None
    state_diff: MemoryDiff | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
    maintenance: dict[str, Any] = Field(default_factory=dict)
    cost_summary: dict[str, Any] = Field(default_factory=dict)


class ResearchProtocol:
    """Run dataset-neutral lifecycle phases against a research backend."""

    version = "research-v1"

    def __init__(self, backend: MemoryBackend, *, trace: Any | None = None):
        self.backend = backend
        self.trace = trace

    def _emit(self, event: str, **payload: Any) -> None:
        if self.trace is None:
            return
        record = {"event": event, "timestamp": _utc_now(), **payload}
        callback = getattr(self.trace, "record_event", None) or getattr(self.trace, "record", None)
        if callback is not None:
            callback(record)

    def _ingest(self, turns: Iterable[ResearchTurn], phase: ProtocolPhase) -> list[SourceRecord]:
        records: list[SourceRecord] = []
        for turn in turns:
            self._emit("observe_start", phase=phase.value, source_id=turn.source_id)
            record = self.backend.observe(turn.to_turn(), source_id=turn.source_id)
            records.append(record)
            self._emit(
                "observe_end",
                phase=phase.value,
                source_id=turn.source_id,
                source_hash=record.source_hash,
            )
        return records

    def _query_items(
        self,
        items: Sequence[ResearchItem],
        phase: ProtocolPhase,
    ) -> list[ProtocolQuery]:
        results: list[ProtocolQuery] = []
        for item in items:
            self._emit("query_start", phase=phase.value, item_id=item.item_id)
            raw = self.backend.query(item.question)
            result = raw if isinstance(raw, QueryResult) else QueryResult(answer=str(raw))
            query = ProtocolQuery(
                phase=phase,
                item_id=item.item_id,
                instance_id=item.instance_id,
                question=item.question,
                answer=result.answer,
                evidence=result.evidence.model_dump(mode="json"),
                latency_ms=result.latency_ms,
                usage=result.usage,
            )
            results.append(query)
            self._emit(
                "query_end",
                phase=phase.value,
                item_id=item.item_id,
                source_ids=list(result.evidence.source_ids),
                latency_ms=result.latency_ms,
            )
        return results

    def run_instance(
        self,
        instance: ResearchInstance,
        *,
        control_items: Sequence[ResearchItem] | None = None,
        forget_items: Sequence[ResearchItem] | None = None,
        forget_source_ids: Sequence[str] | None = None,
        apply_explicit_forget: bool = False,
        relearn_turns: Sequence[ResearchTurn] | None = None,
        relearn_items: Sequence[ResearchItem] | None = None,
        maintenance: dict[str, Any] | None = None,
        cost_summary: dict[str, Any] | None = None,
    ) -> ProtocolResult:
        """Run control and forget phases with snapshots and provenance.

        ``apply_explicit_forget`` is opt-in.  If false, the caller can model a
        natural-language request by passing a different ``turns`` sequence.
        If true, the backend's explicit ``forget`` method is called after the
        forget turns are ingested; unsupported backends fail loudly.
        """

        started = _utc_now()
        # An explicit empty list is a real negative-control condition.
        control_turns = instance.turns if instance.control_turns is None else instance.control_turns
        c_items = list(control_items if control_items is not None else instance.items)
        f_items = list(forget_items if forget_items is not None else instance.items)
        self.backend.reset()
        self._emit("phase_start", phase=ProtocolPhase.CONTROL.value, instance_id=instance.instance_id)
        self._ingest(control_turns, ProtocolPhase.CONTROL)
        control_snapshot = self.backend.snapshot()
        control_queries = self._query_items(c_items, ProtocolPhase.CONTROL)
        self._emit("phase_end", phase=ProtocolPhase.CONTROL.value, instance_id=instance.instance_id)

        self.backend.reset()
        self._emit("phase_start", phase=ProtocolPhase.FORGET.value, instance_id=instance.instance_id)
        self._ingest(instance.turns, ProtocolPhase.FORGET)
        before_forget = self.backend.snapshot()
        forget_result: ForgetResult | None = None
        ids = list(forget_source_ids if forget_source_ids is not None else instance.forget_source_ids)
        if apply_explicit_forget and ids:
            forget_result = self.backend.forget(ids, reason="research_protocol")
            self._emit(
                "forget",
                phase=ProtocolPhase.FORGET.value,
                requested_ids=list(ids),
                removed_ids=list(forget_result.removed_ids),
            )
        after_forget = self.backend.snapshot()
        diff = self.backend.diff(before_forget, after_forget)
        forget_queries = self._query_items(f_items, ProtocolPhase.FORGET)
        self._emit("phase_end", phase=ProtocolPhase.FORGET.value, instance_id=instance.instance_id)
        relearn_queries: list[ProtocolQuery] = []
        if relearn_turns or relearn_items:
            self._emit("phase_start", phase=ProtocolPhase.RELEARN.value, instance_id=instance.instance_id)
            if relearn_turns:
                self._ingest(relearn_turns, ProtocolPhase.RELEARN)
            relearn_queries = self._query_items(
                list(relearn_items or f_items), ProtocolPhase.RELEARN
            )
            self._emit("phase_end", phase=ProtocolPhase.RELEARN.value, instance_id=instance.instance_id)
        finished = _utc_now()
        events: list[dict[str, Any]] = []
        if self.trace is not None and hasattr(self.trace, "events"):
            trace_events = getattr(self.trace, "events")
            events = list(trace_events() if callable(trace_events) else trace_events)
        return ProtocolResult(
            dataset=instance.dataset,
            instance_id=instance.instance_id,
            protocol_version=self.version,
            started_at=started,
            finished_at=finished,
            control_queries=control_queries,
            forget_queries=forget_queries,
            relearn_queries=relearn_queries,
            control_snapshot=control_snapshot,
            before_forget_snapshot=before_forget,
            after_forget_snapshot=after_forget,
            forget_result=forget_result,
            state_diff=diff,
            events=list(events),
            maintenance=dict(maintenance or {}),
            cost_summary=dict(cost_summary or {}),
        )

