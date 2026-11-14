"""Thread-safe JSONL trace hooks for LLM and memory operations."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class TraceRecorder:
    """Record structured events without writing secrets.

    ``path`` is optional.  When omitted, events remain in memory and can be
    inspected through :meth:`events` or :meth:`summary`, which makes the class
    convenient for offline unit tests.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @staticmethod
    def hash_text(value: str | None) -> str | None:
        return hashlib.sha256(value.encode("utf-8")).hexdigest() if value is not None else None

    def record(self, kind: str, **fields: Any) -> dict[str, Any]:
        event = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            "kind": kind,
            **fields,
        }
        with self._lock:
            self._events.append(event)
            if self.path:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        return event

    def record_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Record a pre-built protocol event on the same JSONL path."""
        payload = dict(event)
        kind = str(payload.pop("kind", payload.pop("event", "protocol_event")))
        payload.setdefault("protocol_event", True)
        return self.record(kind, **payload)

    @contextmanager
    def span(self, kind: str, **fields: Any) -> Iterator[dict[str, Any]]:
        started = time.perf_counter_ns()
        context = {"kind": kind, **fields}
        try:
            yield context
        except Exception as exc:
            self.record(kind, phase="error", duration_ms=(time.perf_counter_ns() - started) / 1e6,
                        error_type=type(exc).__name__, error=str(exc), **fields)
            raise
        else:
            self.record(kind, phase="complete", duration_ms=(time.perf_counter_ns() - started) / 1e6,
                        **fields)

    def record_llm(self, *, task: str, model: str | None = None, prompt: str | None = None,
                   duration_ms: float | None = None, usage: dict[str, Any] | None = None,
                   cost: float | None = None, request_id: str | None = None,
                   error: str | None = None) -> dict[str, Any]:
        return self.record(
            "llm_call", task=task, model=model, prompt_sha256=self.hash_text(prompt),
            duration_ms=duration_ms, usage=usage, cost=cost, request_id=request_id,
            error=error,
        )

    def record_memory(self, *, operation: str, instance_id: str | None = None,
                      source_ids: list[str] | None = None, duration_ms: float | None = None,
                      before_hash: str | None = None, after_hash: str | None = None,
                      **fields: Any) -> dict[str, Any]:
        return self.record("memory_operation", operation=operation, instance_id=instance_id,
                           source_ids=source_ids, duration_ms=duration_ms,
                           before_hash=before_hash, after_hash=after_hash, **fields)

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        durations: dict[str, float] = {}
        errors = 0
        with self._lock:
            for event in self._events:
                kind = str(event.get("kind"))
                counts[kind] = counts.get(kind, 0) + 1
                if event.get("phase") == "error" or event.get("error"):
                    errors += 1
                if isinstance(event.get("duration_ms"), (int, float)):
                    durations[kind] = durations.get(kind, 0.0) + float(event["duration_ms"])
        return {"events": len(self._events), "by_kind": counts, "duration_ms": durations, "errors": errors}
