"""Provider-free round-four measurement and baseline contracts."""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from types import SimpleNamespace

from bemula.baselines.base import MemoryAdapter
from bemula.research.backend import QueryEvidence, QueryResult
from bemula.research.amem import AmemHistoryBuilder, AmemResearchWrapper, run_amem_all_arms
from bemula.research.dependency import DependencyGraph
from bemula.research.maintenance import MaintenanceArm, SourceEvent
from bemula.research.state import ProvenanceEvent, ProvenanceEventType
from bemula.research.trace import TraceRecorder
from bemula.schemas import Turn


class _Note:
    def __init__(self, note_id, content, timestamp=""):
        self.id, self.content, self.timestamp = note_id, content, timestamp
        self.context, self.keywords, self.tags, self.links = "fixture", [], [], []


class _System:
    def __init__(self):
        self.memories, self.counter = OrderedDict(), 0

    def add_note(self, content, time=""):
        note_id = f"n{self.counter}"
        self.counter += 1
        self.memories[note_id] = _Note(note_id, content, time)
        return note_id

    def find_related_memories(self, query, k=5):
        return "", []


class _SerialProvider:
    serial = 0

    def __init__(self):
        self.last_usage = {"prompt_tokens": 11, "completion_tokens": 7}

    def chat(self, messages, *args, **kwargs):
        type(self).serial += 1
        return f"{messages[-1]['content']}:response-{type(self).serial}"


class _ProviderAdapter(MemoryAdapter):
    def __init__(self):
        self.llm = _SerialProvider()
        self.reset()

    def reset(self):
        self._system = _System()

    def observe(self, turn: Turn):
        self.llm.chat([{"role": "user", "content": turn.text}], temperature=0.0, task="amem_internal")
        self._system.add_note(turn.text, time=turn.timestamp)

    def query(self, question: str) -> str:
        return "\n".join(note.content for note in self._system.memories.values())


def _events(n=3):
    return [SourceEvent(f"e{i}", f"s{i}", f"payload-{i}", timestamp=f"t{i}") for i in range(n)]


def test_empty_delete_uses_one_common_counterfactual_response_function():
    _SerialProvider.serial = 0
    results = run_amem_all_arms(_events(3), [], _ProviderAdapter, m3_mode="recheck")
    assert all(comparison.equal for _arm, (_result, comparison) in results.items())
    m0 = results[MaintenanceArm.M0_FULL_REBUILD][0]
    assert m0.metadata["reference_repeat_equal"]
    assert m0.metadata["reference_repeat_visible_equal"]


def test_recheck_imports_full_baseline_memo_for_unchanged_suffix():
    builder = AmemHistoryBuilder(_ProviderAdapter, m3_mode="recheck")
    history = _events(3)
    baseline = builder.build(history, arm=MaintenanceArm.M0_FULL_REBUILD)
    before = _SerialProvider.serial
    result = builder.run_arm(history, ["s1"], MaintenanceArm.M3_READ_SET, baseline=baseline)
    assert result.cost.llm_cache_hits == 1
    assert _SerialProvider.serial == before


def test_direct_wrapper_does_not_change_uncached_builder_semantics():
    _SerialProvider.serial = 0
    raw = _ProviderAdapter()
    raw.observe(Turn(speaker="user", text="same", timestamp="t0"))
    raw.observe(Turn(speaker="user", text="same", timestamp="t1"))
    raw_count = _SerialProvider.serial
    _SerialProvider.serial = 0
    wrapped = AmemResearchWrapper(_ProviderAdapter())
    wrapped.observe(Turn(speaker="user", text="same", timestamp="t0"), source_id="s0")
    wrapped.observe(Turn(speaker="user", text="same", timestamp="t1"), source_id="s1")
    assert _SerialProvider.serial == raw_count


def test_method_cache_usage_is_known_and_cache_key_contains_max_tokens():
    builder = AmemHistoryBuilder(_ProviderAdapter)
    result = builder.build(
        [SourceEvent("e0", "s0", "same", timestamp="t0"), SourceEvent("e1", "s1", "same", timestamp="t1")],
        arm=MaintenanceArm.M0_FULL_REBUILD,
    )
    assert result.cost.input_tokens == 11
    assert result.cost.output_tokens == 7
    wrapper = AmemResearchWrapper(_ProviderAdapter())
    provider = wrapper.adapter.llm
    with wrapper._llm_hook():
        first = provider.chat([{"role": "user", "content": "same"}], temperature=0.0, max_tokens=1)
        second = provider.chat([{"role": "user", "content": "same"}], temperature=0.0, max_tokens=9)
    assert first != second


def test_query_probe_shares_only_request_cassette_across_isolated_forks():
    """Query behavior uses one oracle cassette without sharing runtime state."""

    from scripts.run_q01_amem_paired import _probe_query

    class _Probe:
        def __init__(self):
            self._response_cache = {"build-key": "build-response"}
            self._response_usage_cache = {"build-key": {"prompt_tokens": 3}}
            self.provenance_events = []
            self.forked = None
            self.cache_mode = "record"

        def fork(self, trace=None):
            clone = _Probe()
            clone._response_cache = dict(self._response_cache)
            clone._response_usage_cache = dict(self._response_usage_cache)
            self.forked = clone
            return clone

        def set_trace_context(self, **_kwargs):
            return None

        def query(self, _question):
            return QueryResult(
                answer="ok",
                evidence=QueryEvidence(
                    source_ids=[],
                    read_set=[],
                    details={
                        "evidence_complete": True,
                        "context_segments": [],
                        "context_segments_complete": True,
                    },
                ),
            )

    first = _Probe()
    second = _Probe()
    cassette = {"build-key": "build-response"}
    usage = {"build-key": {"prompt_tokens": 3}}
    trace = TraceRecorder()
    _probe_query(
        SimpleNamespace(wrapper=first),
        arm=MaintenanceArm.M0_FULL_REBUILD,
        question="q",
        question_id="q0",
        deleted=set(),
        trace=trace,
        response_cache=cassette,
        response_usage_cache=usage,
    )
    _probe_query(
        SimpleNamespace(wrapper=second),
        arm=MaintenanceArm.M1_TOMBSTONE,
        question="q",
        question_id="q0",
        deleted=set(),
        trace=trace,
        response_cache=cassette,
        response_usage_cache=usage,
    )
    assert first.forked is not first
    assert second.forked is not second
    assert first.forked._response_cache is cassette
    assert second.forked._response_cache is cassette
    assert first.forked._response_usage_cache is usage
    assert second.forked._response_usage_cache is usage
    assert first.forked.cache_mode == second.forked.cache_mode == "replay"


def test_completion_fallback_keeps_explicit_zero_temperature_default():
    """The completion hook must not reintroduce the provider's old default."""

    class Completion:
        def __init__(self):
            self.received = []

        def get_completion(self, prompt, temperature=0.0):
            self.received.append(temperature)
            return "ok"

    class CompletionAdapter(MemoryAdapter):
        def reset(self):
            self._system = SimpleNamespace(
                llm_controller=SimpleNamespace(llm=Completion())
            )

        def observe(self, turn):
            return None

        def query(self, question):
            return ""

    wrapper = AmemResearchWrapper(CompletionAdapter())
    llm = wrapper._system().llm_controller.llm
    with wrapper._llm_hook():
        llm.get_completion("probe")
    assert llm.received == [0.0]


def test_controlled_loader_preserves_top_level_speaker_and_session(tmp_path):
    """History conversion must keep source metadata used by the frozen trace."""

    from scripts.run_q01_amem_controlled import _load_histories

    path = Path(tmp_path) / "history.jsonl"
    path.write_text(
        json.dumps(
            {
                "instance_id": "h0",
                "events": [
                    {
                        "event_id": "e0",
                        "source_id": "s0",
                        "content": "hello",
                        "speaker": "assistant",
                        "session_id": "session-1",
                        "timestamp": "t0",
                    }
                ],
                "deleted_source_ids": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    event = _load_histories(path)[0]["events"][0]
    assert event.speaker == "assistant"
    assert event.session_id == "session-1"


def test_versioned_dependency_closure_does_not_backpropagate_to_old_consumers():
    """Deleting a newer object version must not invalidate an old-version read."""

    events = [
        ProvenanceEvent(
            event_id="w1",
            event_type=ProvenanceEventType.WRITE,
            step=1,
            source_ids=["s0"],
            object_ids=["n"],
            metadata={"object_versions": {"n": "v1"}},
        ),
        ProvenanceEvent(
            event_id="w2",
            event_type=ProvenanceEventType.UPDATE,
            step=2,
            source_ids=["s1"],
            object_ids=["n"],
            candidate_ids=["n"],
            metadata={
                "object_versions": {"n": "v2"},
                "candidate_versions": {"n": "v1"},
            },
        ),
        ProvenanceEvent(
            event_id="use_old",
            event_type=ProvenanceEventType.WRITE,
            step=3,
            object_ids=["x"],
            candidate_ids=["n"],
            metadata={
                "object_versions": {"x": "v1"},
                "candidate_versions": {"n": "v1"},
            },
        ),
    ]
    closure = DependencyGraph.from_events(events).affected_closure(["s1"])
    assert "decision:1:write" not in closure
    assert "decision:3:write" not in closure


def test_amem_mdag_is_explicit_optional_materialized_repair_all_baseline():
    _SerialProvider.serial = 0
    results = run_amem_all_arms(
        _events(3), ["s0"], _ProviderAdapter, m3_mode="recheck", include_m_dag=True
    )
    assert MaintenanceArm.M_DAG_REPAIR_ALL in results
    result, comparison = results[MaintenanceArm.M_DAG_REPAIR_ALL]
    assert comparison.equal
    assert result.metadata["dependency_baseline"] == "materialized_dependency_dag"
    assert result.metadata["dag_status"] == "materialized_dependency_repair_all"
    assert result.metadata["dag_complete"] is True
    assert result.metadata["repair_policy"] == "repair_all_downstream_events"
    assert result.metadata["replay_mode"] == "materialized_dependency_repair_all"
    assert result.metadata["response_reuse_policy"] == "common_baseline_request_key_cassette"
    assert result.cost.provider_attempts == 0
    # Existing serial M0--M3 receipts are unchanged unless the extra arm is
    # requested explicitly.
    default_results = run_amem_all_arms(_events(3), ["s0"], _ProviderAdapter)
    assert MaintenanceArm.M_DAG_REPAIR_ALL not in default_results


def test_formal_reviewer_discovers_optional_mdag_without_changing_core_schema():
    from scripts.review_q01_amem_formal import ARMS, OPTIONAL_ARMS, _arms_for_rows

    assert _arms_for_rows([{"arms": {}}]) == ARMS
    assert _arms_for_rows(
        [{"arms": {OPTIONAL_ARMS[0]: {}}}]
    ) == ARMS + OPTIONAL_ARMS
