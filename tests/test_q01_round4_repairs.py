"""Provider-free round-four measurement and baseline contracts."""
from __future__ import annotations

from collections import OrderedDict

from bemula.baselines.base import MemoryAdapter
from bemula.research.amem import AmemHistoryBuilder, AmemResearchWrapper, run_amem_all_arms
from bemula.research.maintenance import MaintenanceArm, SourceEvent
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
