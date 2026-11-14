"""Cost and build-operation accounting for Q01 research runs.

The Bemula evaluator reports answer-level behavior metrics.  Q01 additionally
needs a reproducible account of the work performed by each maintenance arm:
observations, retrievals, decisions, rebuilds, model calls, embeddings, index
updates, and other numeric resources.  ``CostRecorder`` stores only structured
metadata and numeric counters; it never needs to persist prompts or source
text.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


# These names are deliberately descriptive rather than prescriptive.  A
# builder may add more numeric counters, but these are the common cross-arm
# counters used in reports.
STANDARD_COUNTERS = (
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "embedding_calls",
    "embedding_items",
    "retrieval_calls",
    "retrieved_items",
    "decision_count",
    "memory_writes",
    "memory_updates",
    "memory_deletes",
    "index_updates",
    "rebuild_units",
    "bytes_read",
    "bytes_written",
)

_SENSITIVE_TOKENS = ("prompt", "content", "text", "secret", "token", "key", "password")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _safe_field(name: str, value: Any) -> Any:
    """Keep audit labels useful without persisting obvious secrets/content."""
    lowered = name.lower()
    if any(token in lowered for token in _SENSITIVE_TOKENS) and isinstance(value, str):
        return {"sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_field(name, item) for item in value]
    if isinstance(value, Mapping):
        return {str(k): _safe_field(str(k), v) for k, v in value.items()}
    return str(value)


def _numeric_counters(counters: Mapping[str, Any] | None) -> dict[str, int | float]:
    out: dict[str, int | float] = {}
    for key, value in (counters or {}).items():
        if isinstance(value, bool):
            out[str(key)] = int(value)
        elif isinstance(value, (int, float)):
            out[str(key)] = value
        else:
            raise TypeError(f"cost counter {key!r} must be numeric, got {type(value).__name__}")
    return out


class CostRecorder:
    """Thread-safe JSONL recorder and aggregate for Q01 operation costs.

    ``operation`` is one of ``build``, ``observe``, ``retrieve``, ``decision``,
    ``rebuild`` or a builder-specific label.  Counters are numeric and can be
    merged across maintenance arms.  An optional JSONL ``path`` makes the
    operation trace durable; without it the recorder is an in-memory fixture.
    """

    schema_version = 1

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @staticmethod
    def hash_text(value: str | None) -> str | None:
        return hashlib.sha256(value.encode("utf-8")).hexdigest() if value is not None else None

    def record(
        self,
        operation: str,
        *,
        scope: str = "unspecified",
        duration_ms: float | None = None,
        counters: Mapping[str, Any] | None = None,
        error: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Record one operation with numeric cost counters and safe labels."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "timestamp": _utc_now(),
            "monotonic_ns": time.monotonic_ns(),
            "kind": "cost_operation",
            "operation": str(operation),
            "scope": str(scope),
            "duration_ms": duration_ms,
            "counters": _numeric_counters(counters),
            "error": error,
        }
        payload.update({str(key): _safe_field(str(key), value) for key, value in fields.items()})
        with self._lock:
            self._events.append(payload)
            if self.path:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        return payload

    def record_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Bridge a pre-built event into the cost stream.

        Protocol events without numeric counters are retained with operation
        ``protocol:<event>`` so one stream can be joined by timestamp later.
        """
        payload = dict(event)
        operation = str(payload.pop("operation", payload.pop("event", "protocol")))
        counters = payload.pop("counters", None)
        duration_ms = payload.pop("duration_ms", None)
        return self.record(operation, duration_ms=duration_ms, counters=counters, **payload)

    @contextmanager
    def span(self, operation: str, **fields: Any) -> Iterator[dict[str, Any]]:
        """Measure an operation and record completion or failure."""
        started = time.perf_counter_ns()
        context: dict[str, Any] = {"operation": operation, **fields}
        try:
            yield context
        except Exception as exc:
            self.record(
                operation,
                duration_ms=(time.perf_counter_ns() - started) / 1e6,
                error=f"{type(exc).__name__}: {exc}",
                **fields,
            )
            raise
        else:
            self.record(
                operation,
                duration_ms=(time.perf_counter_ns() - started) / 1e6,
                **fields,
            )

    def record_llm(
        self,
        *,
        task: str,
        model: str | None = None,
        prompt: str | None = None,
        duration_ms: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost: float | None = None,
        request_id: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        counters: dict[str, int | float] = {"llm_calls": 1}
        if input_tokens is not None:
            counters["input_tokens"] = input_tokens
        if output_tokens is not None:
            counters["output_tokens"] = output_tokens
        return self.record(
            "llm",
            duration_ms=duration_ms,
            counters=counters,
            task=task,
            model=model,
            prompt_sha256=self.hash_text(prompt),
            cost=cost,
            request_id=request_id,
            error=error,
        )

    def record_memory(
        self,
        *,
        operation: str,
        duration_ms: float | None = None,
        source_ids: list[str] | None = None,
        before_hash: str | None = None,
        after_hash: str | None = None,
        counters: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        return self.record(
            operation,
            duration_ms=duration_ms,
            counters=counters,
            source_ids=source_ids,
            before_hash=before_hash,
            after_hash=after_hash,
            **fields,
        )

    def record_arm_cost(
        self,
        *,
        arm: str,
        cost: Any,
        phase: str = "maintenance",
        exact_to_reference: bool | None = None,
    ) -> dict[str, Any]:
        """Record the complete numeric cost of one Q01 maintenance arm.

        ``cost`` may be a ``BuildCost`` instance or any object exposing the
        same fields.  Missing values remain explicitly absent instead of being
        silently replaced with zero; callers can use ``null`` in their report
        when a backend did not measure a component.
        """
        names = (
            "input_events", "replayed_events", "prefix_replayed_events",
            "suffix_replayed_events", "checkpoint_restore_events",
            "checkpoint_restore_bytes", "retrieved_candidates",
            "generated_objects", "state_records", "llm_calls", "input_tokens",
            "output_tokens", "embedding_calls", "embedding_items",
            "index_updates", "memory_reads", "bytes_read", "bytes_written",
        )
        counters = {
            name: value for name in names
            if (value := getattr(cost, name, None)) is not None
        }
        # Backward-compatible report aliases used by existing Q01 summaries.
        if "replayed_events" in counters:
            counters.setdefault("rebuild_units", counters["replayed_events"])
        if "retrieved_candidates" in counters:
            counters.setdefault("retrieved_items", counters["retrieved_candidates"])
        if "generated_objects" in counters:
            counters.setdefault("memory_writes", counters["generated_objects"])
        fields: dict[str, Any] = {
            "arm": arm,
            "cost_scope": getattr(cost, "cost_scope", phase),
            "exact_to_reference": exact_to_reference,
        }
        duration_ms = getattr(cost, "wall_time_ms", None)
        return self.record(
            "maintenance_cost",
            scope=phase,
            duration_ms=duration_ms,
            counters=counters,
            **fields,
        )

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def load_existing(self) -> int:
        """Load previously persisted JSONL events into this recorder.

        Resume runs must include already completed arms in the final lifecycle
        cost summary.  Loading is explicit so a fresh in-memory recorder does
        not silently double-count or replay file contents.
        """
        if self.path is None or not self.path.exists():
            return 0
        loaded: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid cost JSONL at {self.path}:{line_no}") from exc
                if not isinstance(value, dict) or value.get("kind") != "cost_operation":
                    raise ValueError(f"invalid cost event at {self.path}:{line_no}")
                loaded.append(value)
        with self._lock:
            self._events = loaded
        return len(loaded)

    def merge(self, other: "CostRecorder", *, arm: str | None = None) -> int:
        """Copy another recorder's structured events into this recorder.

        The operation is useful when M0--M3 run in isolated workers. Numeric
        counters remain additive and raw prompt/content fields stay subject to
        the same redaction path as normal records.
        """

        count = 0
        for event in other.events:
            payload = dict(event)
            payload.pop("timestamp", None)
            payload.pop("monotonic_ns", None)
            operation = str(payload.pop("operation", "merged"))
            counters = payload.pop("counters", {})
            payload.pop("kind", None)
            payload.pop("schema_version", None)
            if arm is not None:
                payload["arm"] = arm
            self.record(operation, counters=counters, **payload)
            count += 1
        return count

    def summary(self) -> dict[str, Any]:
        """Return deterministic operation and counter aggregates."""
        by_operation: dict[str, dict[str, Any]] = {}
        by_scope: dict[str, dict[str, Any]] = {}
        totals: dict[str, int | float] = {}
        total_duration = 0.0
        errors = 0
        with self._lock:
            events = list(self._events)
        for event in events:
            operation = str(event["operation"])
            scope = str(event.get("scope", "unspecified"))
            bucket = by_operation.setdefault(
                operation,
                {"count": 0, "duration_ms": 0.0, "counters": {}, "errors": 0},
            )
            bucket["count"] += 1
            duration = event.get("duration_ms")
            if isinstance(duration, (int, float)):
                bucket["duration_ms"] += float(duration)
                total_duration += float(duration)
            if event.get("error"):
                bucket["errors"] += 1
                errors += 1
            for key, value in event.get("counters", {}).items():
                bucket["counters"][key] = bucket["counters"].get(key, 0) + value
                totals[key] = totals.get(key, 0) + value
            scope_bucket = by_scope.setdefault(scope, {"count": 0, "duration_ms": 0.0, "counters": {}, "errors": 0})
            scope_bucket["count"] += 1
            if isinstance(duration, (int, float)):
                scope_bucket["duration_ms"] += float(duration)
            if event.get("error"):
                scope_bucket["errors"] += 1
            for key, value in event.get("counters", {}).items():
                scope_bucket["counters"][key] = scope_bucket["counters"].get(key, 0) + value
        return {
            "schema_version": self.schema_version,
            "events": len(events),
            "operations": by_operation,
            "scopes": by_scope,
            "totals": totals,
            "duration_ms": total_duration,
            "errors": errors,
        }

    def write_summary(self, path: str | Path) -> None:
        """Atomically write the current aggregate summary."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".tmp")
        temp.write_text(json.dumps(self.summary(), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        # Open read/write for fsync.  Windows rejects fsync on a descriptor
        # opened read-only (the POSIX form used by the original implementation
        # raises ``OSError: [Errno 9] Bad file descriptor``).
        with temp.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
