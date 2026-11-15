"""Source-aware wrapper and offline history runner for A-MEM.

The stock Bemula A-MEM adapter exposes only reset/observe/query.  This module
adds an audit layer around that adapter without changing its behavior: source
IDs are bound to note IDs, note snapshots are serialised, retrieval selection
is captured by a temporary retriever hook, and note evolution decisions become
ProvenanceEvent records.  The wrapper deliberately fails loudly for explicit
forget unless the wrapped implementation provides a real deletion API. Q01
uses two independent M0 rebuilds and a forkable shared baseline for M1--M3;
checkpoint restore and suffix work are counted separately.
"""
from __future__ import annotations

import hashlib
import copy
import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Protocol, Sequence

from bemula.baselines.base import MemoryAdapter
from bemula.schemas import Turn

from .backend import (
    ForgetResult,
    MemoryBackend,
    MemorySnapshot,
    QueryEvidence,
    QueryResult,
    SourceRecord,
)
from .maintenance import (
    BuildCost,
    BuildResult,
    BuildStep,
    MaintenanceArm,
    SourceEvent,
    StateComparison,
    compare_snapshots,
)
from .state import (
    CanonicalArtifact,
    CanonicalQuery,
    CanonicalState,
    CanonicalStateComparator,
    ProvenanceEvent,
    ProvenanceEventType,
    SourceProvenance,
)
from .dependency import DependencyGraph


class AmemAdapterFactory(Protocol):
    def __call__(self) -> MemoryAdapter: ...


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        value = value.keys()
    if isinstance(value, (str, bytes)):
        return [str(value)]
    try:
        return [str(item) for item in value]
    except TypeError:
        return [str(value)]


def _as_sequence(value: Any) -> list[Any]:
    """Convert lists, tuples and numpy arrays without truth-value checks."""

    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


class AmemResearchWrapper(MemoryBackend):
    """Expose provenance/state introspection around a Bemula A-MEM adapter.

    The wrapper works with the current ``AmemAdapter`` (which stores the
    official system at ``_system``) and with a small fake adapter used by the
    offline tests.  It never writes raw prompt text into the decision trace.
    """

    name = "amem-research-wrapper"

    def __init__(self, adapter: MemoryAdapter, *, retrieve_k: int = 10, trace: Any | None = None):
        self.adapter = adapter
        self.retrieve_k = retrieve_k
        self.trace = trace
        self.reset()

    def reset(self) -> None:
        self.adapter.reset()
        self._source_records: dict[str, SourceRecord] = {}
        self._source_to_objects: dict[str, set[str]] = {}
        self._object_support: dict[str, set[str]] = {}
        self._tombstones: set[str] = set()
        self._events: list[ProvenanceEvent] = []
        self._step = 0
        self._llm_trace: list[dict[str, Any]] = []
        self._query_records: list[CanonicalQuery] = []
        self._logical_ids: dict[str, str] = {}
        # Stable creation identity is derived from the external source event
        # and output slot, never from a physical note/storage identifier.
        self._logical_origins: dict[str, str] = {}
        # Request-keyed provider responses are copied with a runtime fork.
        # Recheck policies may reuse an unchanged response, but always execute
        # the current write logic against the restored state.
        self._response_cache: dict[str, str] = {}
        # Physical object IDs can be updated in place by A-MEM.  Keep the
        # version visible in dependency traces so an update reads the old
        # version and writes a new one instead of creating a self-cycle.
        self._object_versions: dict[str, str] = {}
        self._trace_context: dict[str, Any] = {
            "instance_id": "unknown",
            "base_instance_id": "unknown",
            "scenario_id": "unknown",
            "arm": "unknown",
            "phase": "unknown",
        }
        self._checkpoints: dict[int, "AmemResearchWrapper"] = {}
        self._provider_fork_mode = getattr(self.adapter, "provider_fork_mode", "unknown")

    def fork(self, *, trace: Any | None = None) -> "AmemResearchWrapper":
        """Fork the already-built runtime without re-observing provider input.

        A-MEM has no public clone API.  The adapter may provide ``fork``; the
        offline and simple adapters are otherwise copied with ``deepcopy``.
        Failure is explicit because silently falling back to prefix replay
        would mislabel a full rebuild as an incremental checkpoint arm.
        """

        adapter_fork = getattr(self.adapter, "fork", None)
        if callable(adapter_fork):
            adapter = adapter_fork()
        else:
            try:
                adapter = copy.deepcopy(self.adapter)
            except Exception as exc:
                raise RuntimeError(
                    "adapter does not support an in-memory fork; checkpoint replay is unavailable"
                ) from exc
        clone = object.__new__(type(self))
        clone.adapter = adapter
        clone.retrieve_k = self.retrieve_k
        clone.trace = trace
        clone._source_records = copy.deepcopy(self._source_records)
        clone._source_to_objects = copy.deepcopy(self._source_to_objects)
        clone._object_support = copy.deepcopy(self._object_support)
        clone._tombstones = copy.deepcopy(self._tombstones)
        clone._events = copy.deepcopy(self._events)
        clone._step = self._step
        clone._llm_trace = copy.deepcopy(self._llm_trace)
        clone._query_records = copy.deepcopy(self._query_records)
        clone._logical_ids = copy.deepcopy(self._logical_ids)
        clone._logical_origins = copy.deepcopy(self._logical_origins)
        clone._response_cache = copy.deepcopy(self._response_cache)
        clone._object_versions = copy.deepcopy(self._object_versions)
        clone._trace_context = copy.deepcopy(self._trace_context)
        clone._checkpoints = {}
        clone._provider_fork_mode = getattr(adapter, "provider_fork_mode", "unknown")
        return clone

    def _save_checkpoint(self, event_count: int) -> None:
        self._checkpoints[event_count] = self.fork(trace=None)

    def fork_checkpoint(self, event_count: int, *, trace: Any | None = None) -> tuple["AmemResearchWrapper", int]:
        checkpoint = self._checkpoints.get(event_count)
        if checkpoint is None:
            raise RuntimeError(
                f"no runtime checkpoint exists at event boundary {event_count}; "
                "a provider-specific checkpoint/fork is required"
            )
        clone = checkpoint.fork(trace=trace)
        try:
            checkpoint_payload = json.dumps(
                checkpoint.canonical_snapshot().model_dump(mode="json"),
                ensure_ascii=False,
            ).encode("utf-8")
            checkpoint_bytes = len(checkpoint_payload)
        except Exception:
            checkpoint_bytes = len(repr(checkpoint._system()).encode("utf-8"))
        return clone, checkpoint_bytes

    def set_trace_context(self, **fields: Any) -> None:
        """Attach stable run/history/arm/phase identifiers to future events."""

        self._trace_context.update({key: value for key, value in fields.items() if value is not None})

    def _system(self) -> Any:
        return getattr(self.adapter, "_system", self.adapter)

    def _memories(self) -> dict[str, Any]:
        system = self._system()
        memories = getattr(system, "memories", None)
        if isinstance(memories, Mapping):
            return {str(key): value for key, value in memories.items()}
        return {}

    @staticmethod
    def _note_content(note: Any) -> str:
        return str(getattr(note, "content", note))

    @staticmethod
    def _note_links(note: Any) -> list[str]:
        return _as_list(getattr(note, "links", []))

    def _note_records(self) -> dict[str, SourceRecord]:
        records: dict[str, SourceRecord] = {}
        for note_id, note in self._memories().items():
            support = sorted(self._object_support.get(note_id, set()))
            metadata = {
                "note_id": note_id,
                "keywords": _as_list(getattr(note, "keywords", [])),
                "tags": _as_list(getattr(note, "tags", [])),
                "context": str(getattr(note, "context", "")),
                "importance_score": getattr(note, "importance_score", None),
                "retrieval_count": getattr(note, "retrieval_count", None),
                "evolution_history_length": len(getattr(note, "evolution_history", []) or []),
            }
            records[f"note:{note_id}"] = SourceRecord(
                source_id=f"note:{note_id}",
                content=self._note_content(note),
                kind="note",
                timestamp=str(getattr(note, "timestamp", "")),
                version=max(1, len(getattr(note, "evolution_history", []) or []) + 1),
                support_set=support,
                derived_from=self._note_links(note),
                metadata=metadata,
            )
        return records

    def _note_ids_for_indices(self, indices: Any) -> list[str]:
        note_ids = list(self._memories())
        output: list[str] = []
        for index in _as_sequence(indices):
            try:
                position = int(index)
            except (TypeError, ValueError):
                continue
            if 0 <= position < len(note_ids):
                output.append(note_ids[position])
        return output

    @staticmethod
    def _changed_fields(before: SourceRecord, after: SourceRecord) -> list[str]:
        left = before.model_dump(mode="json")
        right = after.model_dump(mode="json")
        return sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))

    @staticmethod
    def _index_summary(system: Any) -> dict[str, Any]:
        retriever = getattr(system, "retriever", None)
        if retriever is None:
            return {"available": False}
        corpus = getattr(retriever, "corpus", None)
        embeddings = getattr(retriever, "embeddings", None)
        shape = getattr(embeddings, "shape", None)
        document_ids = getattr(retriever, "document_ids", None)
        embedding_hash = None
        embedding_dtype = None
        if embeddings is not None:
            try:
                import hashlib
                import numpy as np

                array = np.asarray(embeddings)
                embedding_hash = hashlib.sha256(array.tobytes(order="C")).hexdigest()
                embedding_dtype = str(array.dtype)
            except Exception:
                # Optional diagnostic only; absence must remain explicit.
                embedding_hash = _hash_text(repr(embeddings))
        document_id_map = None
        if isinstance(document_ids, Mapping):
            # Retriever document keys in the stock A-MEM implementation are
            # rendered memory payloads (content, context, keywords and tags),
            # rather than opaque IDs.  Persisting those keys in a provenance
            # trace would disclose source text and can make the trace grow
            # quadratically as the index expands.  Keep the index shape and
            # positions useful for diagnostics while replacing each key with
            # a deterministic hash that cannot recover the payload.
            document_id_map = {
                f"sha256:{_hash_text(str(key))}": (
                    int(value)
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    else f"sha256:{_hash_text(str(value))}"
                )
                for key, value in sorted(document_ids.items(), key=lambda item: str(item[0]))
            }
        return {
            "available": True,
            "class": type(retriever).__name__,
            "corpus_size": len(corpus) if corpus is not None else None,
            "embedding_shape": list(shape) if shape is not None else None,
            "embedding_dtype": embedding_dtype,
            "embedding_hash": embedding_hash,
            "document_count": len(document_ids) if isinstance(document_ids, Mapping) else None,
            "document_id_map": document_id_map,
        }

    @staticmethod
    def _runtime_summary(system: Any) -> dict[str, Any]:
        """Expose optional queue/cache hooks without inventing absent state."""

        queue_names = ("queue", "pending_tasks", "task_queue", "async_queue")
        cache_names = ("cache", "retrieval_cache", "embedding_cache")
        queue = next((getattr(system, name) for name in queue_names if hasattr(system, name)), None)
        cache = next((getattr(system, name) for name in cache_names if hasattr(system, name)), None)
        counters: dict[str, Any] = {}
        for name in ("evo_cnt", "consolidation_count", "consolidation_counter", "step", "counter"):
            if hasattr(system, name):
                value = getattr(system, name)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    counters[name] = value
        def value_hash(value: Any) -> str | None:
            if value is None:
                return None
            try:
                return _hash_text(repr(value))
            except Exception:
                return _hash_text(type(value).__name__)
        return {
            "queue_available": queue is not None,
            "queue_size": len(queue) if hasattr(queue, "__len__") else None,
            "queue_hash": value_hash(queue),
            "cache_available": cache is not None,
            "cache_size": len(cache) if hasattr(cache, "__len__") else None,
            "cache_hash": value_hash(cache),
            "counters": counters,
        }

    @staticmethod
    def _branch_label(response: str) -> str | None:
        upper = str(response).upper()
        for label in (
            "STRENGTHEN_AND_UPDATE",
            "UPDATE_NEIGHBOR",
            "STRENGTHEN",
            "NO_EVOLUTION",
        ):
            if label in upper:
                return label
        return None

    @staticmethod
    def _call_kind(prompt: str) -> str:
        upper = str(prompt).upper()
        if "EVOLUTION" in upper or "DECISION" in upper:
            return "evolution_decision"
        if "STRENGTHEN" in upper:
            return "strengthen"
        if "UPDATE" in upper or "NEIGHBOR" in upper:
            return "neighbor_update"
        if "ANALYZE" in upper or "KEYWORDS" in upper:
            return "content_analysis"
        return "completion"

    @contextmanager
    def _llm_hook(self):
        """Capture prompt/response hashes and parsed branch labels when exposed.

        A-MEM's public adapter does not expose a call-level trace.  The stock
        adapter does expose ``_system.llm_controller.llm`` internally, so this
        hook records only audit hashes and a conservative branch classification
        while preserving the original callable.
        """
        system = self._system()
        controller = getattr(system, "llm_controller", None)
        llm = getattr(controller, "llm", None) if controller is not None else None
        completion = getattr(llm, "get_completion", None)
        adapter_llm = getattr(self.adapter, "llm", None)
        adapter_chat = getattr(adapter_llm, "chat", None) if adapter_llm is not None else None
        if not callable(completion) and not callable(adapter_chat):
            yield []
            return
        calls: list[dict[str, Any]] = []

        def usage_value(usage: Mapping[str, Any], *names: str) -> int | None:
            for name in names:
                if name in usage and usage[name] is not None:
                    try:
                        return int(usage[name])
                    except (TypeError, ValueError):
                        return None
            return None

        def record_provider_call(
            *,
            response: Any,
            prompt_hash: str,
            usage: Mapping[str, Any],
            call_kind: str,
            temperature: Any = None,
            task: str | None = None,
            cache_hit: bool = False,
        ) -> dict[str, Any]:
            response_text = str(response)
            entry: dict[str, Any] = {
                "call_kind": call_kind,
                "task": task or call_kind,
                "prompt_hash": prompt_hash,
                "response_hash": _hash_text(response_text),
                "temperature": temperature,
                "decision": self._branch_label(response_text),
                # ``None`` means provider usage was not exposed.  It is kept
                # distinct from a measured zero in both trace and cost code.
                "input_tokens": usage_value(usage, "prompt_tokens", "input_tokens"),
                "output_tokens": usage_value(usage, "completion_tokens", "output_tokens"),
                "usage_observed": bool(usage),
                "cache_hit": cache_hit,
            }
            if usage:
                entry["usage"] = dict(usage)
            calls.append(entry)
            self._llm_trace.append(entry)
            self._next_event(
                ProvenanceEventType.LLM_CALL,
                decision=entry["decision"],
                input_hashes=[prompt_hash],
                output_hashes=[entry["response_hash"]],
                metadata={key: value for key, value in entry.items() if key not in {"prompt_hash", "response_hash"}},
            )
            return entry

        def wrapped(prompt: str, temperature: float = 0.7, *args: Any, **kwargs: Any) -> str:
            prompt_hash = _hash_text(str(prompt))
            call_kind = self._call_kind(str(prompt))
            cache_key = _hash_text(
                json.dumps(
                    {"prompt_hash": prompt_hash, "temperature": temperature, "call_kind": call_kind},
                    sort_keys=True,
                )
            )
            cache_hit = cache_key in self._response_cache
            if cache_hit:
                response = self._response_cache[cache_key]
                usage = {}
            else:
                response = completion(prompt, *args, temperature=temperature, **kwargs)
                response = str(response)
                self._response_cache[cache_key] = response
                usage = dict(getattr(llm, "last_usage", {}) or {})
            # This fallback is used only when no provider-level ``chat`` hook
            # exists.  When both surfaces exist, wrapping both would count the
            # same provider request twice because get_completion delegates to
            # chat in the production A-MEM bridge.
            record_provider_call(
                response=response,
                prompt_hash=prompt_hash,
                usage=usage,
                call_kind=call_kind,
                temperature=temperature,
                task="amem_internal",
                cache_hit=cache_hit,
            )
            return str(response)

        def wrapped_chat(messages: Any, *args: Any, **kwargs: Any) -> Any:
            # Record once at the provider boundary.  The same hook sees both
            # A-MEM internal calls (task=amem_internal) and answer calls; the
            # nested get_completion wrapper is deliberately not installed.
            prompt_hash = _hash_text(repr(messages))
            task = str(kwargs.get("task", ""))
            call_kind = "query_answer" if task in {"", "amem_answer"} else (
                "amem_internal" if task == "amem_internal" else self._call_kind(repr(messages))
            )
            temperature = kwargs.get("temperature")
            cache_key = _hash_text(
                json.dumps(
                    {"prompt_hash": prompt_hash, "temperature": temperature, "call_kind": call_kind},
                    sort_keys=True,
                )
            )
            cache_hit = cache_key in self._response_cache
            if cache_hit:
                response = self._response_cache[cache_key]
                usage = {}
            else:
                response = adapter_chat(messages, *args, **kwargs)
                response = str(response)
                self._response_cache[cache_key] = response
                usage = dict(getattr(adapter_llm, "last_usage", {}) or {})
            record_provider_call(
                response=response,
                prompt_hash=prompt_hash,
                usage=usage,
                call_kind=call_kind,
                temperature=temperature,
                task=task or call_kind,
                cache_hit=cache_hit,
            )
            return response

        # Prefer the real provider boundary.  Only install the completion hook
        # when no lower-level chat surface is available.
        if callable(adapter_chat):
            setattr(adapter_llm, "chat", wrapped_chat)
        elif callable(completion):
            setattr(llm, "get_completion", wrapped)
        try:
            yield calls
        finally:
            if callable(completion) and not callable(adapter_chat):
                setattr(llm, "get_completion", completion)
            if callable(adapter_chat):
                setattr(adapter_llm, "chat", adapter_chat)

    def _retrieval_hook(self, question: str, call: Callable[..., str]) -> tuple[str, list[str], list[str], dict[str, Any]]:
        """Run adapter.query while capturing the actual retriever selection."""
        system = self._system()
        retriever = getattr(system, "retriever", None)
        search = getattr(retriever, "search", None)
        captured: dict[str, Any] = {
            "candidate_ids": [],
            "indices": [],
            "scores": {},
            "scores_available": False,
            "raw_context_hash": None,
            "raw_context_chars": None,
            "context_object_ids": [],
            "context_source_ids": [],
            "context_mapping_complete": False,
            "context_coverage": False,
            "context_coverage_ratio": 0.0,
            "lineage_coverage": False,
            "lineage_ambiguous": False,
            "retrieval_score_coverage": False,
            "unmapped_context_hash": None,
            # ``score_indices`` is an audit hook rather than part of the
            # adapter's retrieval API.  Keep its extra work explicit so a
            # score-complete trace cannot be mistaken for zero-overhead
            # instrumentation.
            "instrumentation_cost": {
                "score_encode_calls": 0,
                "score_matrix_rows": 0,
                "score_matrix_values": 0,
            },
        }

        def map_rendered_context(rendered: str) -> None:
            """Map raw context spans conservatively and expose coverage layers.

            A text-only upstream renderer does not expose its fragment
            boundaries.  We therefore mark every non-whitespace character not
            covered by a known note payload as unmapped.  This deliberately
            produces ``incomplete`` for templates, tool/cache tails, or other
            renderer text until a structured segment hook is available.
            """

            memories = self._memories()
            matches: list[tuple[int, int, str]] = []
            for note_id, note in memories.items():
                content = self._note_content(note)
                if not content:
                    continue
                start = 0
                while True:
                    position = rendered.find(content, start)
                    if position < 0:
                        break
                    matches.append((position, position + len(content), note_id))
                    start = position + max(1, len(content))
            matches.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
            ambiguous_note_ids: set[str] = set()
            for index, (start, end, note_id) in enumerate(matches):
                same_span = {
                    candidate_note
                    for other_start, other_end, candidate_note in matches[index:]
                    if other_start == start and other_end == end
                }
                if len(same_span) > 1:
                    ambiguous_note_ids.update(same_span)
            selected_matches: list[tuple[int, int, str]] = []
            covered_until = -1
            for start, end, note_id in matches:
                if start < covered_until:
                    continue
                selected_matches.append((start, end, note_id))
                covered_until = end
            context_ids = sorted(
                {note_id for _, _, note_id in selected_matches} | ambiguous_note_ids
            )
            intervals = [(start, end) for start, end, _ in selected_matches]
            unknown_parts: list[str] = []
            cursor = 0
            for start, end in intervals:
                if start > cursor:
                    unknown_parts.append(rendered[cursor:start])
                cursor = max(cursor, end)
            if cursor < len(rendered):
                unknown_parts.append(rendered[cursor:])
            unknown_text = "".join(unknown_parts)
            unknown_non_ws = "".join(character for character in unknown_text if not character.isspace())
            mapped_chars = sum(
                1
                for start, end in intervals
                for character in rendered[start:end]
                if not character.isspace()
            )
            non_ws_chars = sum(1 for character in rendered if not character.isspace())
            context_coverage = not unknown_non_ws
            lineage_coverage = context_coverage and not ambiguous_note_ids and all(
                bool(self._object_support.get(note_id)) for note_id in context_ids
            )
            captured["context_object_ids"] = context_ids
            captured["context_source_ids"] = sorted({
                source
                for note_id in context_ids
                for source in self._object_support.get(note_id, set())
            })
            captured["context_mapping_complete"] = context_coverage and lineage_coverage
            captured["context_coverage"] = context_coverage
            captured["context_coverage_ratio"] = (
                float(mapped_chars) / float(non_ws_chars) if non_ws_chars else 1.0
            )
            captured["lineage_coverage"] = lineage_coverage
            captured["lineage_ambiguous"] = bool(ambiguous_note_ids)
            captured["unmapped_context_hash"] = _hash_text(unknown_text) if unknown_text else None

        def score_indices(query: str, indices: Any) -> None:
            try:
                model = getattr(retriever, "model", None)
                embeddings = getattr(retriever, "embeddings", None)
                if model is None or embeddings is None:
                    return
                import numpy as np

                query_embedding = np.asarray(model.encode([query]))[0]
                instrumentation = captured["instrumentation_cost"]
                instrumentation["score_encode_calls"] += 1
                matrix = np.asarray(embeddings)
                if matrix.ndim != 2:
                    return
                instrumentation["score_matrix_rows"] += int(matrix.shape[0])
                instrumentation["score_matrix_values"] += int(matrix.size)
                norm = float(np.linalg.norm(query_embedding))
                if norm == 0:
                    return
                values = matrix @ query_embedding / (np.linalg.norm(matrix, axis=1) * norm + 1e-12)
                note_ids = self._note_ids_for_indices(indices)
                captured["scores"] = {
                    note_id: float(values[int(index)])
                    for note_id, index in zip(note_ids, _as_sequence(indices))
                    if 0 <= int(index) < len(values)
                }
                captured["scores_available"] = bool(captured["scores"])
            except Exception:
                return

        raw_finder = getattr(system, "find_related_memories_raw", None)

        def wrapped_raw(query: str, k: int = self.retrieve_k, *args: Any, **kwargs: Any) -> Any:
            result = raw_finder(query, k, *args, **kwargs)
            rendered = str(result)
            captured["raw_context_hash"] = _hash_text(rendered)
            captured["raw_context_chars"] = len(rendered)
            map_rendered_context(rendered)
            return result

        if not callable(search):
            finder = getattr(system, "find_related_memories", None)
            if callable(finder):
                try:
                    result = finder(question, k=self.retrieve_k)
                    indices = result[1] if isinstance(result, tuple) and len(result) > 1 else []
                    selected = self._note_ids_for_indices(indices)
                    return call(question), list(self._memories()), selected, captured
                except Exception:
                    pass
            return call(question), list(self._memories()), list(self._memories())[: self.retrieve_k], captured

        def wrapped(query: str, k: int = self.retrieve_k, *args: Any, **kwargs: Any) -> Any:
            indices = search(query, k, *args, **kwargs)
            captured["indices"] = _as_sequence(indices)
            captured["candidate_ids"] = self._note_ids_for_indices(indices)
            score_indices(query, indices)
            return indices

        setattr(retriever, "search", wrapped)
        if callable(raw_finder):
            setattr(system, "find_related_memories_raw", wrapped_raw)
        try:
            answer = call(question)
        finally:
            setattr(retriever, "search", search)
            if callable(raw_finder):
                setattr(system, "find_related_memories_raw", raw_finder)
        selected = captured["candidate_ids"][: self.retrieve_k]
        captured["retrieval_score_coverage"] = bool(captured["scores_available"])
        return answer, captured["candidate_ids"], selected, captured

    def _record(self, event: ProvenanceEvent) -> None:
        self._events.append(event)
        if self.trace is not None:
            callback = getattr(self.trace, "record_event", None)
            if callback is not None:
                payload = event.model_dump(mode="json")
                payload.update({key: value for key, value in self._trace_context.items() if value is not None})
                callback(payload)

    def _next_event(
        self,
        event_type: ProvenanceEventType,
        *,
        source_ids: Sequence[str] = (),
        object_ids: Sequence[str] = (),
        candidate_ids: Sequence[str] = (),
        selected_ids: Sequence[str] = (),
        control_dependencies: Sequence[str] = (),
        decision: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        query_id: str | None = None,
        input_hashes: Sequence[str] = (),
        output_hashes: Sequence[str] = (),
    ) -> ProvenanceEvent:
        self._step += 1
        context = dict(self._trace_context)
        operation_id = str(
            context.get("operation_id")
            or f"{context.get('instance_id', 'unknown')}:{context.get('arm', 'unknown')}:{context.get('phase', 'unknown')}:{self._step:06d}"
        )
        event_metadata = {**context, "operation_id": operation_id, **dict(metadata or {})}
        event = ProvenanceEvent(
            event_id=f"{operation_id}:{event_type.value}",
            event_type=event_type,
            step=self._step,
            source_ids=list(source_ids),
            object_ids=list(object_ids),
            candidate_ids=list(candidate_ids),
            selected_ids=list(selected_ids),
            control_dependencies=list(control_dependencies),
            decision=decision,
            query_id=query_id,
            input_hashes=list(input_hashes),
            output_hashes=list(output_hashes),
            metadata=event_metadata,
        )
        self._record(event)
        return event

    def observe(self, turn: Turn, *, source_id: str | None = None) -> SourceRecord:
        sid = source_id or f"turn:{self._step + 1:06d}"
        before = self._note_records()
        self._next_event(
            ProvenanceEventType.INGEST,
            source_ids=[sid],
            decision="observe",
            input_hashes=[_hash_text(turn.text)],
            metadata={"speaker": turn.speaker, "timestamp": turn.timestamp},
        )
        # The official robust A-MEM implementation asks the retriever for
        # neighbours inside ``process_memory``.  Capture those integer
        # positions at the system boundary while keeping the adapter itself
        # unchanged.  This is a runtime dependency trace, not a formal causal
        # proof: the upstream implementation does not expose its parsed LLM
        # branch decision.
        captured_candidates: list[str] = []
        system = self._system()
        index_before = self._index_summary(system)
        finder = getattr(system, "find_related_memories", None)
        def run_observe() -> None:
            if callable(finder):
                def wrapped_finder(query: str, k: int = 5, *args: Any, **kwargs: Any) -> Any:
                    result = finder(query, k, *args, **kwargs)
                    indices = result[1] if isinstance(result, tuple) and len(result) > 1 else []
                    captured_candidates.extend(self._note_ids_for_indices(indices))
                    return result

                setattr(system, "find_related_memories", wrapped_finder)
                try:
                    self.adapter.observe(turn)
                finally:
                    setattr(system, "find_related_memories", finder)
            else:
                self.adapter.observe(turn)

        with self._llm_hook() as llm_calls:
            run_observe()
        index_after = self._index_summary(system)
        after = self._note_records()
        before_ids, after_ids = set(before), set(after)
        new_ids = sorted(after_ids - before_ids)
        changed_ids = sorted(
            note_id for note_id in before_ids & after_ids
            if before[note_id].model_dump(mode="json") != after[note_id].model_dump(mode="json")
        )
        object_ids = [item.removeprefix("note:") for item in new_ids + changed_ids]
        # A NOOP is an execution decision and must not fabricate a materialized
        # object by attaching the input to an arbitrary existing note.
        # Downstream maintenance can still use the explicit candidate event.
        support_sources = {sid}
        support_sources.update(
            source
            for note_id in captured_candidates
            for source in self._object_support.get(note_id, set())
        )
        candidate_versions = {
            note_id: self._object_versions[note_id]
            for note_id in captured_candidates
            if note_id in self._object_versions
        }
        for object_id in object_ids:
            self._object_support.setdefault(object_id, set()).update(support_sources)
        # Establish physical-ID-independent identity only after the current
        # source support is known. Deriving identity before this update made
        # same-content notes from different source events collapse to one
        # canonical ID, which in turn hid retargeted links.
        for output_slot, object_id in enumerate(new_ids):
            record = after.get(object_id)
            if record is not None:
                self._logical_origins.setdefault(
                    object_id,
                    f"source:{sid}:output:{output_slot}",
                )
                self._ensure_logical_id(object_id.removeprefix("note:"), record)
        if captured_candidates:
            candidate_support = sorted({
                source
                for note_id in captured_candidates
                for source in self._object_support.get(note_id, set())
            })
            self._next_event(
                ProvenanceEventType.CANDIDATE,
                source_ids=candidate_support,
                candidate_ids=list(captured_candidates),
                control_dependencies=candidate_support,
                decision="process_memory_neighbours",
                input_hashes=[_hash_text(turn.text)],
                metadata={
                    "capture_mode": "find_related_memories_hook",
                    "candidate_order": list(captured_candidates),
                    "candidate_versions": candidate_versions,
                    "scores_available": False,
                    "index_before": index_before,
                    "index_after": index_after,
                },
            )
        decision = "write" if new_ids else "update" if changed_ids else "noop"
        event_type = {
            "write": ProvenanceEventType.WRITE,
            "update": ProvenanceEventType.UPDATE,
            "noop": ProvenanceEventType.NOOP,
        }[decision]
        object_versions = {
            object_id: f"{_hash_text(after[f'note:{object_id}'].content)[:16]}@{self._step + 1}"
            for object_id in object_ids
            if f"note:{object_id}" in after
        }
        self._next_event(
            event_type,
            source_ids=sorted(support_sources),
            object_ids=object_ids,
            selected_ids=object_ids,
            # The write decision is conditioned on the objects observed by
            # the retriever during this observe call.  Keeping these as
            # control dependencies connects read/candidate nodes to the
            # newly materialized object in the dependency graph.
            control_dependencies=captured_candidates,
            decision=decision,
            input_hashes=[_hash_text(turn.text)],
            metadata={
                "speaker": turn.speaker,
                "timestamp": turn.timestamp,
                "current_source_id": sid,
                "content_support_ids": sorted(support_sources),
                "object_versions": object_versions,
                "candidate_versions": candidate_versions,
                "llm_calls": list(llm_calls),
                "index_before": index_before,
                "index_after": index_after,
                "changed_fields": {
                    note_id: self._changed_fields(before[note_id], after[note_id])
                    for note_id in changed_ids
                },
            },
        )
        for support_source in support_sources:
            self._source_to_objects.setdefault(support_source, set()).update(object_ids)
        self._object_versions.update(object_versions)
        source = SourceRecord(
            source_id=sid,
            content=turn.text,
            kind="source",
            speaker=turn.speaker,
            timestamp=turn.timestamp,
            derived_from=object_ids,
            metadata={"object_ids": object_ids},
        )
        self._source_records[sid] = source
        return source

    def query(self, question: str) -> QueryResult:
        with self._llm_hook() as llm_calls:
            answer, candidate_ids, selected_ids, retrieval_details = self._retrieval_hook(question, self.adapter.query)
        support_ids = sorted({
            source_id
            for note_id in selected_ids
            for source_id in self._object_support.get(note_id, set())
        })
        context_object_ids = list(retrieval_details.get("context_object_ids", []))
        context_support_ids = {
            source_id
            for note_id in context_object_ids
            for source_id in self._object_support.get(note_id, set())
        }
        support_ids = sorted(set(support_ids) | context_support_ids)
        read_object_ids = sorted(set(selected_ids) | set(context_object_ids))
        context_mapping_complete = bool(retrieval_details.get("context_mapping_complete"))
        context_coverage = bool(retrieval_details.get("context_coverage"))
        lineage_coverage = bool(retrieval_details.get("lineage_coverage"))
        evidence_complete = bool(
            retrieval_details.get("raw_context_hash")
            and context_coverage
            and lineage_coverage
            and context_mapping_complete
        )
        query_id = f"query:{_hash_text(question)[:16]}"
        self._next_event(
            ProvenanceEventType.READ,
            source_ids=support_ids,
            object_ids=read_object_ids,
            candidate_ids=candidate_ids,
            selected_ids=selected_ids,
            control_dependencies=support_ids,
            decision="retrieve",
            query_id=query_id,
            input_hashes=[_hash_text(question)],
            metadata={
                "llm_calls": list(llm_calls),
                "evidence_complete": evidence_complete,
                "context_coverage": context_coverage,
                "lineage_coverage": lineage_coverage,
                "retrieval_score_coverage": bool(retrieval_details.get("scores_available")),
                **retrieval_details,
            },
        )
        self._next_event(
            ProvenanceEventType.SELECT,
            source_ids=support_ids,
            object_ids=read_object_ids,
            candidate_ids=candidate_ids,
            selected_ids=selected_ids,
            control_dependencies=support_ids,
            decision="query",
            query_id=query_id,
            input_hashes=[_hash_text(question)],
            metadata={
                "llm_calls": list(llm_calls),
                "evidence_complete": evidence_complete,
                "context_coverage": context_coverage,
                "lineage_coverage": lineage_coverage,
                "retrieval_score_coverage": bool(retrieval_details.get("scores_available")),
                **retrieval_details,
            },
        )
        self._query_records.append(
            CanonicalQuery(
                query_id=query_id,
                query_text_hash=_hash_text(question),
                query_kind="retrieval",
                visible_source_ids=support_ids,
                candidate_ids=candidate_ids,
                selected_ids=selected_ids,
                scores=dict(retrieval_details.get("scores", {})),
                top_k=self.retrieve_k,
                tie_break="adapter_order",
                control_dependencies=support_ids,
                metadata={
                    "provenance_quality": "wrapper_delta",
                    "answer_retrieval_is_best_effort": True,
                    "read_object_ids": read_object_ids,
                    "evidence_complete": evidence_complete,
                    **retrieval_details,
                },
            )
        )
        return QueryResult(
            answer=str(answer),
            evidence=QueryEvidence(
                source_ids=support_ids,
                read_set=support_ids,
                details={
                    "candidate_ids": candidate_ids,
                    "selected_ids": selected_ids,
                    "read_object_ids": read_object_ids,
                    "provenance_quality": "wrapper_delta",
                    "answer_retrieval_is_best_effort": True,
                    "evidence_complete": evidence_complete,
                    "context_coverage": context_coverage,
                    "lineage_coverage": lineage_coverage,
                    "retrieval_score_coverage": bool(retrieval_details.get("scores_available")),
                    "llm_calls": list(llm_calls),
                    **retrieval_details,
                },
            ),
        )

    def snapshot(self) -> MemorySnapshot:
        records = list(self._source_records.values()) + list(self._note_records().values())
        records.sort(key=lambda record: record.source_id)
        return MemorySnapshot(
            state_id=f"amem-state:{self._step:06d}",
            records=records,
            tombstones=sorted(self._tombstones),
            metadata={
                "backend": self.name,
                "note_count": len(records),
                "event_count": len(self._events),
            },
        )

    def _logical_note_id(self, note_id: str, record: SourceRecord) -> str:
        existing = self._logical_ids.get(note_id)
        if existing:
            return existing
        origin = self._logical_origins.get(note_id)
        seed = "|".join(
            [
                origin or "",
                _hash_text(record.content),
                ",".join(sorted(record.support_set)),
                str(record.timestamp or ""),
                str(record.kind or ""),
            ]
        )
        logical = f"note-sem:{_hash_text(seed)[:24]}"
        self._logical_ids[note_id] = logical
        return logical

    def _ensure_logical_id(self, note_id: str, record: SourceRecord) -> str:
        """Assign identity once, at object creation, rather than at export."""

        return self._logical_note_id(note_id, record)

    def _logical_links(self, note_id: str, record: SourceRecord) -> list[str]:
        """Normalize A-MEM integer neighbor positions to logical note IDs."""
        note_ids = list(self._memories())
        links: list[str] = []
        for raw in record.derived_from:
            target = str(raw)
            if target.isdigit() and int(target) < len(note_ids):
                target = note_ids[int(target)]
            if target in self._memories():
                target_record = self._note_records().get(f"note:{target}")
                links.append(
                    self._logical_note_id(target, target_record)
                    if target_record is not None else target
                )
            else:
                links.append(target)
        return links

    def canonical_snapshot(
        self,
        *,
        dataset: str = "unknown",
        instance_id: str = "unknown",
    ) -> CanonicalState:
        """Return a semantic state with physical note IDs normalized.

        The raw :meth:`snapshot` is useful for audit and replay.  This method
        is the state-oracle input: source records remain keyed by stable
        source IDs, while note UUIDs are replaced by a logical ID derived from
        content and source support.  Physical note IDs and the run-local state
        counter are kept only in metadata for debugging.
        """
        note_records = self._note_records()
        artifacts: list[CanonicalArtifact] = []
        for source_id, record in note_records.items():
            note_id = source_id.removeprefix("note:")
            metadata = dict(record.metadata)
            metadata.pop("note_id", None)
            artifacts.append(
                CanonicalArtifact(
                    object_id=note_id,
                    object_type="note",
                    content=record.content,
                    # SourceRecord.source_hash includes wrapper metadata such
                    # as physical object IDs.  Canonical state hashes content
                    # independently of those run-local fields.
                    content_hash=_hash_text(record.content),
                    canonical_id=self._logical_note_id(note_id, record),
                    version=record.version,
                    timestamp=record.timestamp,
                    support_set=list(record.support_set),
                    parent_ids=self._logical_links(note_id, record),
                    retrieval_parent_ids=self._logical_links(note_id, record),
                    visible=True,
                    status="active",
                    metadata=metadata,
                )
            )
        sources: list[SourceProvenance] = []
        for ordinal, record in enumerate(sorted(self._source_records.values(), key=lambda item: item.source_id)):
            metadata = {
                key: value for key, value in record.metadata.items() if key != "object_ids"
            }
            logical_derived = [
                self._logical_note_id(note_id, note_records[f"note:{note_id}"])
                if f"note:{note_id}" in note_records else note_id
                for note_id in record.derived_from
            ]
            sources.append(
                SourceProvenance(
                    source_id=record.source_id,
                    dataset=dataset,
                    instance_id=instance_id,
                    source_type=record.kind,
                    content_hash=_hash_text(record.content),
                    ordinal=ordinal,
                    timestamp=record.timestamp,
                    parent_source_ids=list(record.support_set),
                    derived_from=logical_derived,
                    version=record.version,
                    metadata=metadata,
                )
            )
        return CanonicalState(
            schema_version=1,
            state_id=f"amem-canonical:{self._step:06d}",
            sources=sources,
            artifacts=artifacts,
            queries=list(self._query_records),
            provenance_events=list(self._events),
            tombstones=sorted(self._tombstones),
            metadata={
                "backend": self.name,
                "trace_quality": "wrapper_delta",
                "index": self._index_summary(self._system()),
                "runtime": self._runtime_summary(self._system()),
            },
        )

    def canonical_hash(self, *, dataset: str = "unknown", instance_id: str = "unknown") -> str:
        """Return the semantic hash used by the Q01 state oracle."""
        return self.canonical_snapshot(dataset=dataset, instance_id=instance_id).canonical_hash

    def save_checkpoint(
        self,
        store: Any,
        *,
        state_id: str | None = None,
        dataset: str = "unknown",
        instance_id: str = "unknown",
    ) -> dict[str, Any]:
        """Persist a canonical audit checkpoint through ``CheckpointStore``.

        The stock A-MEM adapter cannot rehydrate a live index from this method;
        the checkpoint is the durable semantic state used to validate replay
        and recovery. A builder-specific restore routine must rebuild the
        physical adapter from the source ledger.
        """

        state = self.canonical_snapshot(dataset=dataset, instance_id=instance_id)
        return store.commit_state(state_id or state.state_id, state, metadata={"backend": self.name})

    @staticmethod
    def load_checkpoint(store: Any, state_id: str) -> CanonicalState:
        envelope = store.load_state(state_id)
        return CanonicalState.model_validate(envelope["state"])

    def introspect(self) -> dict[str, Any]:
        observed_graph = self.dependency_graph()
        branch_observed = any(call.get("decision") for call in self._llm_trace)
        score_observed = any(bool(query.scores) for query in self._query_records)
        raw_context_observed = any(
            query.metadata.get("raw_context_hash") for query in self._query_records
        )
        return {
            "backend": self.name,
            "adapter": getattr(self.adapter, "name", type(self.adapter).__name__),
            "active_notes": sorted(self._memories()),
            "source_to_objects": {key: sorted(value) for key, value in sorted(self._source_to_objects.items())},
            "object_support": {key: sorted(value) for key, value in sorted(self._object_support.items())},
            "dependency_graph": {
                "source_to_objects": {
                    key: sorted(value) for key, value in sorted(self._source_to_objects.items())
                },
                "object_to_sources": {
                    key: sorted(value) for key, value in sorted(self._object_support.items())
                },
                "object_to_links": {
                    key: sorted(record.derived_from)
                    for key, record in sorted(self._note_records().items())
                },
            },
            "source_records": sorted(self._source_records),
            "llm_trace": list(self._llm_trace),
            "query_records": [query.model_dump(mode="json") for query in self._query_records],
            "events": [event.model_dump(mode="json") for event in self._events],
            "observed_dependency_graph": observed_graph.to_dict(),
            "index": self._index_summary(self._system()),
            "runtime": self._runtime_summary(self._system()),
            "provider_fork_mode": self._provider_fork_mode,
            "provider_replay_deterministic": self._provider_fork_mode == "request_key_replay",
            "supports": {
                "snapshot": True,
                "introspect": True,
                "provenance": True,
                "decision_trace": True,
                "explicit_forget": callable(getattr(self.adapter, "forget", None)),
                "dependency_introspection": True,
                "llm_branch_decision": branch_observed,
                "stable_snapshot_ids": False,
                "canonical_snapshot": True,
                "llm_trace": True,
                "retriever_scores": score_observed,
                "raw_context_trace": raw_context_observed,
                "queue_trace": bool(self._runtime_summary(self._system())["queue_available"]),
                "cache_trace": bool(self._runtime_summary(self._system())["cache_available"]),
                # Stable names retained for older ResearchProtocol clients.
                "forget": callable(getattr(self.adapter, "forget", None)),
                "delete": callable(getattr(self.adapter, "delete", None)),
                "exact_query_evidence": bool(self._query_records)
                and all(bool(query.metadata.get("evidence_complete")) for query in self._query_records),
            },
            "trace_quality": "wrapper_delta",
            "limitations": [
                "The upstream adapter does not expose its parsed LLM branch decision.",
                "Candidate/control dependencies are observed at the find_related_memories boundary.",
                "Stock A-MEM has no source-level delete; M0 uses independent rebuilds and M1-M3 use isolated runtime forks/checkpoints.",
                "Stock note UUIDs are physical IDs; canonical cross-run equality requires the Q01 state normalizer.",
                "Official robust A-MEM exposes no queue/cache hooks in the stock runtime.",
            ],
        }

    @property
    def decision_events(self) -> list[ProvenanceEvent]:
        return list(self._events)

    @property
    def provenance_events(self) -> list[ProvenanceEvent]:
        """Compatibility alias for the research trace."""
        return list(self._events)

    def dependency_graph(self, *, complete: bool = False) -> DependencyGraph:
        """Build a forward/reverse graph from the observed decision trace."""

        return DependencyGraph.from_events(self._events, complete=complete)

    def forget(self, source_ids: Sequence[str], *, reason: str | None = None) -> ForgetResult:
        forget = getattr(self.adapter, "forget", None)
        if not callable(forget):
            raise NotImplementedError(
                "A-MEM stock adapter has no source-level forget/delete API; use the paired M0-M3 maintenance runner."
            )
        result = forget(list(source_ids))
        if isinstance(result, ForgetResult):
            return result
        raise TypeError("wrapped adapter forget() must return ForgetResult")

    def delete(self, source_ids: Sequence[str], *, reason: str | None = None) -> ForgetResult:
        return self.forget(source_ids, reason=reason)


@dataclass
class AmemHistoryResult:
    arm: MaintenanceArm
    snapshot: MemorySnapshot
    trace: list[BuildStep]
    cost: BuildCost
    wrapper: AmemResearchWrapper
    metadata: dict[str, Any]
    # M0 carries an independently rebuilt runtime so query continuation can
    # use the same oracle discipline as the state comparison. This is an
    # in-memory handle and is deliberately excluded from serialized metadata.
    reference_repeat_wrapper: AmemResearchWrapper | None = None

    @property
    def canonical_state(self) -> CanonicalState:
        return self.wrapper.canonical_snapshot(dataset="amem", instance_id="fixture")


class AmemHistoryBuilder:
    """Factory-based single-history builder used by offline M0-M3 tests."""

    name = "amem-research-builder"

    def __init__(
        self,
        factory: AmemAdapterFactory,
        *,
        retrieve_k: int = 10,
        trace: Any | None = None,
        trace_context: Mapping[str, Any] | None = None,
        m3_mode: str = "materialized",
    ):
        self.factory = factory
        self.retrieve_k = retrieve_k
        self.trace = trace
        self.trace_context = dict(trace_context or {})
        if m3_mode not in {"materialized", "conservative_full", "recheck"}:
            raise ValueError(f"unknown m3_mode: {m3_mode}")
        self.m3_mode = m3_mode

    def _new(self) -> AmemResearchWrapper:
        wrapper = AmemResearchWrapper(self.factory(), retrieve_k=self.retrieve_k, trace=self.trace)
        wrapper.set_trace_context(**self.trace_context)
        return wrapper

    @staticmethod
    def _embedding_counters(wrapper: AmemResearchWrapper) -> tuple[int | None, int | None]:
        """Read optional embedder counters without requiring a provider API.

        The controlled preflight embedder exposes these counters, and a real
        provider may expose the same optional attributes.  Missing counters
        remain explicitly unknown rather than being inferred from index size.
        """

        embedder = getattr(wrapper.adapter, "embedder", None)
        if embedder is None or not hasattr(embedder, "calls") or not hasattr(embedder, "items"):
            return None, None
        calls = getattr(embedder, "calls", None)
        items = getattr(embedder, "items", None)
        try:
            if calls is None or items is None:
                return None, None
            return int(calls), int(items)
        except (TypeError, ValueError):
            return None, None

    @staticmethod
    def _usage_total(
        wrapper: AmemResearchWrapper,
        field: str,
        events: Sequence[ProvenanceEvent] | None = None,
    ) -> int | None:
        observed_events = list(events) if events is not None else wrapper.provenance_events
        values = [
            event.metadata.get(field)
            for event in observed_events
            if event.event_type is ProvenanceEventType.LLM_CALL
        ]
        if not values or any(value is None for value in values):
            return None
        try:
            return sum(int(value) for value in values)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _counter_delta(after: int | None, before: int | None) -> int | None:
        if after is None or before is None:
            return None
        return max(0, after - before)

    @staticmethod
    def _turn(event: SourceEvent) -> Turn:
        metadata = dict(event.metadata)
        return Turn(
            speaker=str(event.speaker or metadata.get("speaker", "user")),
            text=event.content,
            timestamp=event.timestamp,
            linked_fact_ids=list(metadata.get("linked_fact_ids", [])),
        )

    def _ingest(
        self,
        wrapper: AmemResearchWrapper,
        events: Sequence[SourceEvent],
        deleted: set[str],
        *,
        start_index: int = 0,
    ) -> tuple[list[BuildStep], int]:
        steps: list[BuildStep] = []
        replayed = 0
        for index, event in enumerate(events[start_index:], start=start_index):
            if event.source_id in deleted:
                wrapper._tombstones.add(event.source_id)
                continue
            before = len(wrapper.decision_events)
            wrapper.observe(self._turn(event), source_id=event.source_id)
            wrapper._save_checkpoint(index + 1)
            replayed += 1
            new_events = wrapper.decision_events[before:]
            decision = next(
                (
                    item
                    for item in reversed(new_events)
                    if item.event_type
                    in {
                        ProvenanceEventType.WRITE,
                        ProvenanceEventType.UPDATE,
                        ProvenanceEventType.NOOP,
                    }
                ),
                new_events[-1] if new_events else None,
            )
            candidate_ids = tuple(
                dict.fromkeys(item for trace_event in new_events for item in trace_event.candidate_ids)
            )
            read_set = tuple(
                dict.fromkeys(item for trace_event in new_events for item in trace_event.control_dependencies)
            )
            support_set = tuple(
                dict.fromkeys(item for trace_event in new_events for item in trace_event.source_ids)
            )
            output_ids = tuple(
                dict.fromkeys(item for trace_event in new_events for item in trace_event.object_ids)
            )
            steps.append(
                BuildStep(
                    index=index,
                    event_id=event.event_id,
                    source_id=event.source_id,
                    operation=decision.decision if decision else "observe",
                    read_set=read_set,
                    candidate_ids=candidate_ids,
                    output_ids=output_ids,
                    support_set=support_set,
                    control_set=tuple(dict.fromkeys((*candidate_ids, *read_set))),
                )
            )
        return steps, replayed

    def build(
        self,
        events: Sequence[SourceEvent],
        *,
        arm: MaintenanceArm,
        deleted_ids: Sequence[str] = (),
        start_index: int = 0,
        prefix_wrapper: AmemResearchWrapper | None = None,
        replayed_events: int | None = None,
        trace_phase: str = "build",
    ) -> AmemHistoryResult:
        deleted = set(deleted_ids)
        wrapper = prefix_wrapper or self._new()
        wrapper.set_trace_context(arm=arm.value, phase=trace_phase)
        if start_index not in wrapper._checkpoints:
            wrapper._checkpoints[start_index] = wrapper.fork(trace=None)
        event_start = len(wrapper.provenance_events)
        embedding_calls_before, embedding_items_before = self._embedding_counters(wrapper)
        steps, replayed = self._ingest(wrapper, events, deleted, start_index=start_index)
        if replayed_events is not None:
            replayed = replayed_events
        snapshot = wrapper.snapshot()
        new_events = wrapper.provenance_events[event_start:]
        processed_events = events[start_index:]
        cost = BuildCost(
            input_events=len(events),
            replayed_events=replayed,
            retrieved_candidates=sum(len(step.candidate_ids) for step in steps),
            generated_objects=sum(len(step.output_ids) for step in steps),
            state_records=len(snapshot.records),
            llm_calls=sum(
                1 for event in new_events if event.event_type is ProvenanceEventType.LLM_CALL
            ),
            llm_cache_hits=sum(
                bool(event.metadata.get("cache_hit"))
                for event in new_events
                if event.event_type is ProvenanceEventType.LLM_CALL
            ),
            input_tokens=self._usage_total(wrapper, "input_tokens", new_events),
            output_tokens=self._usage_total(wrapper, "output_tokens", new_events),
            memory_reads=sum(
                1 for event in new_events
                if event.event_type in {ProvenanceEventType.CANDIDATE, ProvenanceEventType.READ}
            ),
            index_updates=sum(
                1 for event in new_events
                if event.event_type in {ProvenanceEventType.WRITE, ProvenanceEventType.UPDATE}
            ),
            bytes_read=sum(
                len(event.content.encode("utf-8"))
                for event in processed_events
                if event.source_id not in deleted
            ),
            embedding_calls=self._counter_delta(
                self._embedding_counters(wrapper)[0], embedding_calls_before
            ),
            embedding_items=self._counter_delta(
                self._embedding_counters(wrapper)[1], embedding_items_before
            ),
        )
        return AmemHistoryResult(
            arm=arm,
            snapshot=snapshot,
            trace=steps,
            cost=cost,
            wrapper=wrapper,
            metadata={"builder": self.name, "replay_start": start_index},
        )

    def run_arm(
        self,
        events: Sequence[SourceEvent],
        deleted_ids: Sequence[str],
        arm: MaintenanceArm,
        *,
        baseline: AmemHistoryResult | None = None,
        trace_phase: str | None = None,
    ) -> AmemHistoryResult:
        deleted = set(deleted_ids)
        if arm is MaintenanceArm.M0_FULL_REBUILD:
            kept = [event for event in events if event.source_id not in deleted]
            result = self.build(
                kept,
                arm=arm,
                deleted_ids=(),
                replayed_events=len(kept),
                trace_phase=trace_phase or "reference_rebuild",
            )
            # The reference rebuild is evaluated for the same deletion
            # request.  Preserve the audit tombstone even though the removed
            # event is absent from the replay input.
            result.wrapper._tombstones.update(deleted)
            result.snapshot = result.wrapper.snapshot()
            return result
        if baseline is None:
            baseline = self.build(events, arm=MaintenanceArm.M0_FULL_REBUILD)
        if arm is MaintenanceArm.M1_TOMBSTONE:
            # M1 is a live tombstone/filter arm.  Filtering only the exported
            # snapshot leaves the baseline adapter serving deleted notes.  Apply
            # the same conservative source/support rule to the wrapper's live
            # state, then derive the snapshot from that state.
            live_wrapper = baseline.wrapper.fork(trace=self.trace)
            live_wrapper.set_trace_context(arm=arm.value, phase="maintenance")
            embedding_calls_before, embedding_items_before = self._embedding_counters(live_wrapper)
            index_updates = self._apply_live_tombstones(live_wrapper, deleted)
            embedding_calls_after, embedding_items_after = self._embedding_counters(live_wrapper)
            snapshot = live_wrapper.snapshot()
            return AmemHistoryResult(
                arm=arm,
                snapshot=snapshot,
                trace=list(baseline.trace),
                cost=BuildCost(
                    input_events=len(events),
                    state_records=len(snapshot.records),
                    index_updates=index_updates,
                    embedding_calls=self._counter_delta(embedding_calls_after, embedding_calls_before),
                    embedding_items=self._counter_delta(embedding_items_after, embedding_items_before),
                    cost_scope="maintenance_live_tombstone",
                ),
                wrapper=live_wrapper,
                metadata={
                    "builder": self.name,
                    "replay_start": 0,
                    # M1 is intentionally the conservative cascade arm.  A
                    # future M1a can provide source-only tombstones while this
                    # arm remains an explicit over-delete baseline.
                    "tombstone_policy": "conservative_cascade",
                    "index_rebuilt": bool(index_updates),
                    "cost_scope": "maintenance_live_tombstone",
                    "shared_baseline": True,
                },
            )
        # M2 is the conservative suffix baseline: it invalidates from the
        # first deleted source regardless of whether later steps actually
        # read that source.  M3 is the stronger materialized read/control
        # baseline and may start earlier when a later decision consumed a
        # derived object supported by the deletion.  Keeping these start
        # policies separate is essential; otherwise M3 is only a relabelled
        # copy of M2.
        materialized_start = self._earliest_affected_index(events, deleted, baseline)
        if arm is MaintenanceArm.M2_REPLAY:
            start = self._earliest_deleted_source_index(events, deleted)
            replay_mode = "conservative_suffix"
        else:
            if self.m3_mode == "conservative_full":
                # A complete generic dependency baseline must remain correct
                # when a query/NOOP/control dependency was not materialized.
                # The conservative barrier is event zero: re-execute every
                # retained event from the empty runtime checkpoint.
                start = 0
                replay_mode = "conservative_full_history"
            elif self.m3_mode == "recheck":
                # Restore the materialized invalidation boundary, re-run the
                # current retrieval/control/write logic, and reuse only exact
                # request-key responses inherited from the shared baseline.
                start = materialized_start
                replay_mode = "generic_recheck_safe_reuse"
            else:
                start = materialized_start
                replay_mode = "read_control_closure"
        wrapper, checkpoint_bytes = baseline.wrapper.fork_checkpoint(start, trace=self.trace)
        wrapper.set_trace_context(arm=arm.value, phase="maintenance")
        suffix_result = self.build(
            events,
            arm=arm,
            deleted_ids=deleted,
            start_index=start,
            prefix_wrapper=wrapper,
            trace_phase="maintenance_checkpoint_replay",
        )
        suffix_result.metadata.update(
            {
                "replay_start": start,
                "affected_steps": [step.index for step in baseline.trace if step.index >= start],
                "replay_mode": replay_mode,
                "m3_status": (
                    (
                        "strong_conservative_full_history_checkpoint_replay"
                        if self.m3_mode == "conservative_full"
                        else "generic_recheck_request_key_cache"
                        if self.m3_mode == "recheck"
                        else "limited_materialized_read_control_checkpoint_replay"
                    )
                    if arm is MaintenanceArm.M3_READ_SET
                    else "conservative_suffix_checkpoint_replay"
                ),
                "m3_mode": self.m3_mode if arm is MaintenanceArm.M3_READ_SET else None,
                "materialized_candidate_start": materialized_start if arm is MaintenanceArm.M3_READ_SET else None,
                "dependency_baseline": (
                    "complete_conservative_barrier"
                    if arm is MaintenanceArm.M3_READ_SET and self.m3_mode == "conservative_full"
                    else "generic_recheck_request_key_cache"
                    if arm is MaintenanceArm.M3_READ_SET and self.m3_mode == "recheck"
                    else "materialized_read_control_set"
                    if arm is MaintenanceArm.M3_READ_SET
                    else None
                ),
                "checkpoint_restored": True,
                "checkpoint_source": "shared_baseline_runtime",
                "checkpoint_event": start,
                "checkpoint_restore_size_kind": "semantic_snapshot_utf8_bytes",
                "checkpoint_restore_scope": "canonical_state_estimate",
            }
        )
        suffix_replayed = suffix_result.cost.replayed_events
        suffix_result.cost = replace(
            suffix_result.cost,
            input_events=len(events),
            replayed_events=suffix_replayed,
            prefix_replayed_events=0,
            suffix_replayed_events=suffix_replayed,
            checkpoint_restore_events=1,
            checkpoint_restore_bytes=checkpoint_bytes,
            cost_scope="maintenance_checkpoint_replay",
        )
        suffix_result.metadata.update(
            {
                "prefix_events": start,
                "suffix_events": suffix_replayed,
                "cost_scope": "maintenance_checkpoint_replay",
            }
        )
        return suffix_result

    @staticmethod
    def _apply_live_tombstones(wrapper: AmemResearchWrapper, deleted: set[str]) -> int:
        """Apply a conservative source tombstone to the serving wrapper.

        The stock adapter has no source-level delete API.  For the research
        M1 arm we therefore remove every note whose observed support intersects
        the deleted source set.  This is an explicit conservative-cascade
        baseline; it keeps the exported snapshot and the actual query path
        aligned without claiming exact source-only deletion.
        """
        if not deleted:
            return 0
        wrapper._tombstones.update(deleted)
        for source_id in deleted:
            wrapper._source_records.pop(source_id, None)
            wrapper._source_to_objects.pop(source_id, None)

        memories = getattr(wrapper._system(), "memories", None)
        old_memory_order = list(memories) if isinstance(memories, Mapping) else []
        remove_ids: set[str] = set()
        for object_id, support in list(wrapper._object_support.items()):
            # Preserve the policy boundary: any materialized object that was
            # supported by a deleted source is removed, even if another source
            # also supported it.  This is a conservative cascade baseline,
            # not a claim that source-level deletion has been solved.
            if deleted.intersection(support):
                remove_ids.add(object_id)
                wrapper._object_support.pop(object_id, None)
        if isinstance(memories, Mapping):
            for object_id in remove_ids:
                memories.pop(object_id, None)
            survivor_order = list(memories)
            old_to_new = {
                old_index: new_index
                for new_index, object_id in enumerate(survivor_order)
                for old_index, old_id in enumerate(old_memory_order)
                if old_id == object_id
            }
            # A-MEM stores links as integer positions.  Removing an earlier
            # note must compact those positions without retargeting a link to
            # the new occupant of the old integer slot.
            for note in memories.values():
                remapped: list[Any] = []
                for raw_link in wrapper._note_links(note):
                    try:
                        old_index = int(raw_link)
                    except (TypeError, ValueError):
                        if str(raw_link) in memories:
                            remapped.append(raw_link)
                        continue
                    new_index = old_to_new.get(old_index)
                    if new_index is not None:
                        remapped.append(new_index)
                try:
                    setattr(note, "links", remapped)
                except Exception:
                    pass
        for source_id, object_ids in list(wrapper._source_to_objects.items()):
            object_ids.difference_update(remove_ids)
            if not object_ids:
                wrapper._source_to_objects.pop(source_id, None)

        # The official A-MEM retriever indexes positions in ``memories``.  Once
        # notes are removed, leaving the old index in place can return stale
        # positions or index into the wrong surviving note.  Rebuild the index
        # from the live survivor set when the adapter exposes its factory; this
        # cost is reported as M1 maintenance work.  If a third-party adapter
        # cannot rebuild, keep the tombstone policy explicit and report zero
        # rather than pretending the index is synchronized.
        system = wrapper._system()
        factory = getattr(wrapper.adapter, "_new_retriever", None)
        retriever = factory() if callable(factory) else None
        if retriever is None:
            return 0
        add_documents = getattr(retriever, "add_documents", None)
        if not callable(add_documents):
            return 0
        system.retriever = retriever
        updates = 0
        for note in getattr(system, "memories", {}).values():
            content = wrapper._note_content(note)
            if not content:
                continue
            metadata_text = (
                f"{getattr(note, 'context', '')} "
                f"{' '.join(_as_list(getattr(note, 'keywords', [])))} "
                f"{' '.join(_as_list(getattr(note, 'tags', [])))}"
            )
            add_documents([f"{content} , {metadata_text}"])
            updates += 1
        return updates

    @staticmethod
    def _earliest_affected_index(
        events: Sequence[SourceEvent], deleted: set[str], baseline: AmemHistoryResult
    ) -> int:
        # Read/candidate traces often refer to materialized note IDs while the
        # deletion protocol names source IDs.  Expand the roots through the
        # baseline source-to-object map before scanning the trace.
        object_roots = {
            object_id
            for source_id in deleted
            for object_id in baseline.wrapper._source_to_objects.get(source_id, set())
        }
        roots = set(deleted) | object_roots | {f"note:{item}" for item in object_roots}
        affected = [
            step.index for step in baseline.trace
            if roots.intersection(
                {step.source_id}
                | set(step.read_set)
                | set(step.candidate_ids)
                | set(step.support_set)
                | set(getattr(step, "control_set", ()))
            )
        ]
        if affected:
            return min(affected)
        for index, event in enumerate(events):
            if event.source_id in deleted:
                return index
        return len(events)

    @staticmethod
    def _earliest_deleted_source_index(events: Sequence[SourceEvent], deleted: set[str]) -> int:
        """Return the first position whose own source is deleted.

        This intentionally ignores read/support edges.  It is the M2
        repair-all baseline against which M3's materialized dependency
        invalidation is compared.
        """

        for index, event in enumerate(events):
            if event.source_id in deleted:
                return index
        return len(events)


def run_amem_all_arms(
    events: Sequence[SourceEvent],
    deleted_ids: Sequence[str],
    factory: AmemAdapterFactory,
    *,
    retrieve_k: int = 10,
    trace: Any | None = None,
    trace_context: Mapping[str, Any] | None = None,
    m3_mode: str = "materialized",
) -> dict[MaintenanceArm, tuple[AmemHistoryResult, StateComparison]]:
    """Run the four fixed arms through a source-aware A-MEM wrapper."""
    builder = AmemHistoryBuilder(
        factory,
        retrieve_k=retrieve_k,
        trace=trace,
        trace_context=trace_context,
        m3_mode=m3_mode,
    )
    deleted = tuple(dict.fromkeys(deleted_ids))
    reference_started = time.perf_counter()
    reference = builder.run_arm(
        events,
        deleted,
        MaintenanceArm.M0_FULL_REBUILD,
        trace_phase="reference_rebuild",
    )
    reference.cost = replace(
        reference.cost,
        wall_time_ms=(time.perf_counter() - reference_started) * 1000.0,
    )
    repeat_started = time.perf_counter()
    reference_repeat = builder.run_arm(
        events,
        deleted,
        MaintenanceArm.M0_FULL_REBUILD,
        trace_phase="reference_repeat",
    )
    reference_repeat.cost = replace(
        reference_repeat.cost,
        wall_time_ms=(time.perf_counter() - repeat_started) * 1000.0,
    )
    reference_diff = CanonicalStateComparator().compare(
        reference.wrapper.canonical_snapshot(dataset="amem", instance_id="fixture"),
        reference_repeat.wrapper.canonical_snapshot(dataset="amem", instance_id="fixture"),
    )
    common_started = time.perf_counter()
    common_baseline = builder.build(
        events,
        arm=MaintenanceArm.M0_FULL_REBUILD,
        trace_phase="common_baseline",
    )
    common_baseline.cost = replace(
        common_baseline.cost,
        wall_time_ms=(time.perf_counter() - common_started) * 1000.0,
    )
    output: dict[MaintenanceArm, tuple[AmemHistoryResult, StateComparison]] = {}

    for arm in MaintenanceArm:
        if arm is MaintenanceArm.M0_FULL_REBUILD:
            result = reference
            result.metadata.update(
                {
                    "reference_repeat": True,
                    "reference_repeat_equal": reference_diff.equivalent,
                    "reference_repeat_visible_equal": reference_diff.visible_equivalent,
                    "reference_repeat_runtime_equal": reference_diff.runtime_equivalent,
                    "reference_repeat_hash": reference_repeat.wrapper.canonical_hash(
                        dataset="amem", instance_id="fixture"
                    ),
                    "reference_repeat_cost": dict(reference_repeat.cost.__dict__),
                    "cost_scope": "reference_rebuild",
                }
            )
            # M0 must validate independent repeatability.  Comparing the
            # reference object with itself would make the reference arm
            # self-certifying even when the builder is nondeterministic.
            comparison_reference = reference_repeat
        else:
            maintenance_started = time.perf_counter()
            result = builder.run_arm(events, deleted, arm, baseline=common_baseline)
            result.metadata.update(
                {
                    "shared_baseline": True,
                    "shared_baseline_cost": dict(common_baseline.cost.__dict__),
                    "shared_baseline_wall_time_ms": common_baseline.cost.wall_time_ms,
                }
            )
            result.cost = replace(
                result.cost,
                wall_time_ms=(time.perf_counter() - maintenance_started) * 1000.0,
            )
            comparison_reference = reference
        # Every arm must be compared through the same builder-specific
        # exporter.  M1 now filters the live wrapper, so comparing a generic
        # snapshot conversion against the A-MEM exporter would create a false
        # mismatch even for an empty deletion.
        left = result.wrapper.canonical_snapshot(dataset="amem", instance_id="fixture")
        if arm is MaintenanceArm.M0_FULL_REBUILD:
            right = reference_repeat.wrapper.canonical_snapshot(dataset="amem", instance_id="fixture")
            diff = reference_diff
        else:
            right = comparison_reference.wrapper.canonical_snapshot(dataset="amem", instance_id="fixture")
            diff = CanonicalStateComparator().compare(left, right)
        output[arm] = (
            result,
            StateComparison(
                equal=diff.equivalent,
                left_hash=diff.left_hash,
                right_hash=diff.right_hash,
                visible_equivalent=diff.visible_equivalent,
                runtime_equivalent=diff.runtime_equivalent,
                continuation_checks_passed=diff.continuation_checks_passed,
                changed_ids=tuple(mismatch.key for mismatch in diff.mismatches),
                reason="equal" if diff.equivalent else "canonical state differs",
            ),
        )
        if arm is MaintenanceArm.M0_FULL_REBUILD:
            result.reference_repeat_wrapper = reference_repeat.wrapper
    return output


AmemBackend = AmemResearchWrapper

__all__ = [
    "AmemAdapterFactory",
    "AmemResearchWrapper",
    "AmemHistoryResult",
    "AmemHistoryBuilder",
    "run_amem_all_arms",
    "AmemBackend",
]
