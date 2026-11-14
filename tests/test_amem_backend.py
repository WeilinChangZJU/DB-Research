from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from bemula.baselines.base import MemoryAdapter
from bemula.research import (
    AmemBackend,
    AmemResearchWrapper,
    MaintenanceArm,
    SourceEvent,
    run_amem_all_arms,
    run_toy_experiment,
)
from bemula.schemas import Turn


class _Note:
    def __init__(self, note_id: str, content: str, timestamp: str):
        self.id = note_id
        self.content = content
        self.timestamp = timestamp
        self.context = "fixture"
        self.keywords = content.lower().split()[:3]
        self.tags = ["fixture"]
        self.links = []


class _System:
    def __init__(self):
        self.memories = OrderedDict()
        self.counter = 0

    def add_note(self, content: str, time: str = "") -> str:
        note_id = f"note-{self.counter}"
        self.counter += 1
        self.memories[note_id] = _Note(note_id, content, time)
        return note_id

    def find_related_memories(self, query: str, k: int = 5):
        tokens = set(query.lower().split())
        ranked = []
        for index, note in enumerate(self.memories.values()):
            score = len(tokens & set(note.content.lower().split()))
            ranked.append((-score, index))
        ranked.sort()
        return "", [index for score, index in ranked[:k] if score < 0]


class _StubAmemAdapter(MemoryAdapter):
    name = "stub-amem"

    def __init__(self):
        self.reset()

    def reset(self):
        self._system = _System()

    def observe(self, turn: Turn) -> None:
        # Mirror the official process_memory read boundary so the wrapper can
        # verify a candidate/dependency event without an LLM.
        self._system.find_related_memories(turn.text, k=5)
        self._system.add_note(turn.text, time=turn.timestamp)

    def query(self, question: str) -> str:
        _, indices = self._system.find_related_memories(question, k=5)
        notes = list(self._system.memories.values())
        return "\n".join(notes[index].content for index in indices)


class _ScriptedAmemAdapter(_StubAmemAdapter):
    class _LLM:
        def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
            self.last_usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            return "NO_EVOLUTION"

    def reset(self):
        super().reset()
        self._system.llm_controller = SimpleNamespace(llm=self._LLM())

    def observe(self, turn: Turn) -> None:
        self._system.llm_controller.llm.get_completion("EVOLUTION DECISION")
        super().observe(turn)


class _QueryUsageAmemAdapter(_StubAmemAdapter):
    class _LLM:
        def __init__(self):
            self.last_usage = {}

        def chat(self, messages, **kwargs):
            self.last_usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
            return "answer"

    def __init__(self):
        self.llm = self._LLM()
        super().__init__()

    def query(self, question: str) -> str:
        super().query(question)
        return self.llm.chat([{"role": "user", "content": question}], temperature=0.0)


def test_amem_backend_captures_snapshot_provenance_and_best_effort_evidence():
    backend = AmemBackend(_StubAmemAdapter(), retrieve_k=2)
    backend.observe(Turn(speaker="user", text="approval limit is ten thousand", timestamp="t0"), source_id="src-a")
    backend.observe(Turn(speaker="user", text="office is in Shanghai", timestamp="t1"), source_id="src-b")

    snapshot = backend.snapshot()
    result = backend.query("office")
    kinds = [event.event_type.value for event in backend.provenance_events]
    info = backend.introspect()

    assert {record.source_id for record in snapshot.records} >= {"src-a", "src-b"}
    assert any(record.kind == "note" for record in snapshot.records)
    assert "ingest" in kinds and "write" in kinds and "read" in kinds
    assert result.answer == "office is in Shanghai"
    assert result.evidence.source_ids
    assert result.evidence.details["provenance_quality"] == "wrapper_delta"
    assert info["supports"]["snapshot"] is True
    assert info["supports"]["forget"] is False
    assert info["supports"]["decision_trace"] is True

    with pytest.raises(NotImplementedError, match="no source-level forget"):
        backend.forget(["src-a"])


def test_amem_wrapper_and_q01_arms_are_separate_offline_contracts():
    events = [
        SourceEvent("e0", "src-a", "approval limit is ten thousand"),
        SourceEvent("e1", "src-b", "approval limit is ten thousand"),
        SourceEvent("e2", "src-c", "office is in Shanghai"),
    ]
    q01 = run_toy_experiment(events, ["src-a"])

    # The deterministic M0-M3 harness is the Q01 maintenance contract; the
    # A-MEM wrapper supplies observation/provenance only and does not silently
    # turn A-MEM's missing delete API into a correctness claim.
    assert q01.arms[MaintenanceArm.M0_FULL_REBUILD].exact_to_reference
    assert q01.arms[MaintenanceArm.M2_REPLAY].exact_to_reference
    assert q01.arms[MaintenanceArm.M3_READ_SET].exact_to_reference
    assert not q01.arms[MaintenanceArm.M1_TOMBSTONE].exact_to_reference


def test_amem_wrapper_runs_source_aware_m0_m3_offline():
    events = [
        SourceEvent("e0", "src-a", "approval limit is ten thousand", timestamp="t0"),
        SourceEvent("e1", "src-b", "approval limit is ten thousand", timestamp="t1"),
        SourceEvent("e2", "src-c", "office is in Shanghai", timestamp="t2"),
    ]
    results = run_amem_all_arms(events, ["src-a"], _StubAmemAdapter, retrieve_k=2)

    assert results[MaintenanceArm.M0_FULL_REBUILD][1].equal
    assert not results[MaintenanceArm.M1_TOMBSTONE][1].equal
    assert results[MaintenanceArm.M2_REPLAY][1].equal
    assert results[MaintenanceArm.M3_READ_SET][1].equal

    wrapper = results[MaintenanceArm.M0_FULL_REBUILD][0].wrapper
    info = wrapper.introspect()
    assert info["supports"]["dependency_introspection"] is True
    assert info["supports"]["llm_branch_decision"] is False
    assert info["supports"]["stable_snapshot_ids"] is False
    assert info["trace_quality"] == "wrapper_delta"
    assert "src-b" in info["source_records"]
    assert any(event.event_type.value == "ingest" for event in wrapper.decision_events)


def test_amem_canonical_snapshot_ignores_physical_note_ids():
    def build_state():
        backend = AmemResearchWrapper(_StubAmemAdapter(), retrieve_k=2)
        backend.observe(Turn(speaker="user", text="office is in Shanghai", timestamp="t0"), source_id="src-a")
        backend.query("office")
        return backend.canonical_snapshot(dataset="fixture", instance_id="one")

    left = build_state()
    right = build_state()
    assert left.canonical_hash == right.canonical_hash
    assert left.queries and left.queries[0].selected_ids


def test_amem_llm_hook_records_branch_without_raw_prompt():
    backend = AmemResearchWrapper(_ScriptedAmemAdapter())
    backend.observe(Turn(speaker="user", text="a private fact", timestamp="t0"), source_id="src-a")
    calls = backend.introspect()["llm_trace"]
    assert calls[0]["call_kind"] == "evolution_decision"
    assert calls[0]["decision"] == "NO_EVOLUTION"
    assert "prompt" not in calls[0] and "response" not in calls[0]
    assert any(event.event_type.value == "llm_call" for event in backend.decision_events)


def test_amem_llm_hook_records_query_chat_usage():
    backend = AmemResearchWrapper(_QueryUsageAmemAdapter())
    backend.observe(Turn(speaker="user", text="a fact", timestamp="t0"), source_id="src-a")
    result = backend.query("fact")
    assert result.answer == "answer"
    query_calls = [call for call in backend.introspect()["llm_trace"] if call["call_kind"] == "query_answer"]
    assert query_calls and query_calls[-1]["input_tokens"] == 11 and query_calls[-1]["output_tokens"] == 7
    assert any(event.event_type.value == "llm_call" for event in backend.decision_events)


def test_amem_index_summary_hashes_rendered_document_keys():
    private_payload = "content: a private fact, context: personal, keywords: secret"
    retriever = SimpleNamespace(
        corpus=[private_payload, "second"],
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        document_ids={private_payload: 0, "second": private_payload},
    )
    summary = AmemResearchWrapper._index_summary(SimpleNamespace(retriever=retriever))

    encoded = repr(summary)
    assert private_payload not in encoded
    key_hash = "sha256:287b592bbe322cced7ed0630791cb01fc7d7db66ed2c13627cec042ab2c9241a"
    assert summary["document_id_map"][key_hash] == 0
    assert all(value != private_payload for value in summary["document_id_map"].values())


def test_amem_maintenance_cost_keeps_trace_counters_across_replay():
    events = [
        SourceEvent("e0", "src-a", "a private fact", timestamp="t0"),
        SourceEvent("e1", "src-b", "a second fact", timestamp="t1"),
    ]
    results = run_amem_all_arms(events, ["src-a"], _ScriptedAmemAdapter)
    m0 = results[MaintenanceArm.M0_FULL_REBUILD][0].cost
    m2 = results[MaintenanceArm.M2_REPLAY][0].cost
    assert m0.llm_calls == 1
    assert m2.llm_calls == 1
    assert m0.input_tokens == 3 and m0.output_tokens == 2
    assert m2.input_tokens == 3 and m2.output_tokens == 2
    assert m0.bytes_read > 0
    assert m2.bytes_read > 0


def test_amem_m2_and_m3_record_distinct_invalidation_policies():
    events = [
        SourceEvent("e0", "src-a", "shared fact", timestamp="t0"),
        SourceEvent("e1", "src-b", "shared followup", timestamp="t1"),
    ]
    results = run_amem_all_arms(events, ["src-a"], _StubAmemAdapter)
    m2 = results[MaintenanceArm.M2_REPLAY][0]
    m3 = results[MaintenanceArm.M3_READ_SET][0]
    assert m2.metadata["replay_mode"] == "conservative_suffix"
    assert m3.metadata["replay_mode"] == "read_control_closure"
    assert m3.metadata["m3_status"].startswith("implemented_materialized")


def test_amem_paired_runner_can_receive_durable_trace_recorder():
    from bemula.research import TraceRecorder

    trace = TraceRecorder()
    events = [SourceEvent("e0", "src-a", "alpha"), SourceEvent("e1", "src-b", "beta")]
    run_amem_all_arms(events, ["src-a"], _StubAmemAdapter, trace=trace)
    assert trace.events
    assert any(event.get("event_type") == "ingest" for event in trace.events)


def test_amem_replay_cost_counts_prefix_reingest():
    events = [
        SourceEvent("e0", "src-a", "first fact", timestamp="t0"),
        SourceEvent("e1", "src-b", "second fact", timestamp="t1"),
        SourceEvent("e2", "src-c", "deleted last", timestamp="t2"),
    ]
    results = run_amem_all_arms(events, ["src-c"], _StubAmemAdapter)
    for arm in (MaintenanceArm.M2_REPLAY, MaintenanceArm.M3_READ_SET):
        result = results[arm][0]
        assert result.cost.replayed_events == 2
        assert result.cost.prefix_replayed_events == 2
        assert result.cost.suffix_replayed_events == 0
        assert result.cost.cost_scope == "maintenance_full_execution"
        assert result.metadata["checkpoint_restored"] is False


def test_amem_canonical_checkpoint_round_trip(tmp_path):
    from bemula.research import CheckpointStore

    backend = AmemResearchWrapper(_StubAmemAdapter())
    backend.observe(Turn(speaker="user", text="office is in Shanghai", timestamp="t0"), source_id="src-a")
    store = CheckpointStore(tmp_path / "run")
    entry = backend.save_checkpoint(store, state_id="amem-state-0", dataset="fixture", instance_id="i0")
    restored = AmemResearchWrapper.load_checkpoint(store, entry["state_id"])
    assert restored.canonical_hash == backend.canonical_snapshot(dataset="fixture", instance_id="i0").canonical_hash



