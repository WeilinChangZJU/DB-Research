from bemula.research import (
    InMemoryBackend,
    MemorySnapshot,
    ResearchInstance,
    ResearchItem,
    ResearchProtocol,
    ResearchTurn,
    TraceRecorder,
)


def _instance() -> ResearchInstance:
    turns = [
        ResearchTurn(
            dataset="fixture",
            instance_id="fixture-0",
            source_id="src-1",
            speaker="user",
            text="My favorite color is blue.",
            timestamp="2025-01-01T00:00:00Z",
        ),
        ResearchTurn(
            dataset="fixture",
            instance_id="fixture-0",
            source_id="src-2",
            speaker="user",
            text="I work in robotics.",
            timestamp="2025-01-01T00:01:00Z",
        ),
    ]
    items = [
        ResearchItem(
            dataset="fixture",
            instance_id="fixture-0",
            item_id="q-1",
            question="What is my favorite color?",
            gold_answer="blue",
            related_source_ids=["src-1"],
        )
    ]
    return ResearchInstance(
        dataset="fixture",
        instance_id="fixture-0",
        turns=turns,
        items=items,
        forget_source_ids=["src-1"],
    )


def test_source_hash_and_snapshot_are_deterministic():
    a = ResearchTurn(
        dataset="fixture",
        instance_id="i",
        source_id="s",
        text="hello",
    )
    b = ResearchTurn(
        dataset="fixture",
        instance_id="i",
        source_id="s",
        text="hello",
    )
    assert a.source_hash == b.source_hash
    backend = InMemoryBackend()
    backend.observe(a.to_turn(), source_id=a.source_id)
    snap = backend.snapshot()
    assert isinstance(snap, MemorySnapshot)
    assert snap.records[0].source_id == "s"
    assert snap.state_hash


def test_delete_and_diff_keep_tombstone_provenance():
    backend = InMemoryBackend()
    first = backend.write("alpha", source_id="a")
    before = backend.snapshot()
    result = backend.delete([first.source_id])
    after = backend.snapshot()
    diff = backend.diff(before, after)
    assert result.removed_ids == ["a"]
    assert diff.removed == ["a"]
    assert "a" in after.tombstones
    assert backend.introspect()["active_records"] == 0


def test_protocol_runs_control_and_explicit_forget_without_external_api():
    backend = InMemoryBackend()
    result = ResearchProtocol(backend).run_instance(
        _instance(), apply_explicit_forget=True
    )
    assert result.dataset == "fixture"
    assert result.control_queries[0].phase.value == "control"
    assert result.forget_result is not None
    assert result.forget_result.removed_ids == ["src-1"]
    assert result.state_diff is not None
    assert result.state_diff.removed == ["src-1"]
    assert result.forget_queries[0].answer == ""


def test_protocol_emits_trace_events_without_api():
    trace = TraceRecorder()
    result = ResearchProtocol(InMemoryBackend(), trace=trace).run_instance(_instance())
    assert result.events
    assert any(event.get("kind") == "phase_start" for event in trace.events)


def test_protocol_supports_relearn_phase():
    backend = InMemoryBackend()
    result = ResearchProtocol(backend).run_instance(
        _instance(),
        apply_explicit_forget=True,
        relearn_turns=[_instance().turns[0]],
        relearn_items=_instance().items,
    )
    assert result.relearn_queries[0].phase.value == "relearn"
    assert result.relearn_queries[0].answer

