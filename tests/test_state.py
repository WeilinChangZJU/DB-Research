from bemula.research import (
    CanonicalArtifact,
    CanonicalQuery,
    CanonicalRelation,
    CanonicalState,
    CanonicalStateComparator,
    ProvenanceEvent,
    ProvenanceEventType,
    SourceProvenance,
)


def _state(*, storage_id: str = "physical-a", content: str = "blue", source_id: str = "src-a") -> CanonicalState:
    source = SourceProvenance(
        source_id=source_id,
        dataset="fixture",
        instance_id="i0",
        content_hash="hash-src-a",
        timestamp="2025-01-01T00:00:00Z",
        metadata={"created_at": "different-wall-clock", "session_id": "s0"},
    )
    artifact = CanonicalArtifact(
        object_id=storage_id,
        canonical_id="note:blue",
        object_type="note",
        content=content,
        support_set=[source_id],
        metadata={"storage_key": storage_id, "created_at": "different-wall-clock"},
    )
    relation = CanonicalRelation(
        relation_id=f"relation-{storage_id}",
        relation_type="supports",
        subject_id="note:blue",
        object_id=source_id,
        support_set=[source_id],
    )
    query = CanonicalQuery(
        query_id="q0",
        query_text_hash="query-hash",
        candidate_ids=["note:blue", "note:other"],
        selected_ids=["note:blue"],
        control_dependencies=[source_id],
        metadata={"runtime_ns": 123},
    )
    event = ProvenanceEvent(
        event_id="e0",
        event_type=ProvenanceEventType.SELECT,
        step=1,
        source_ids=[source_id],
        object_ids=[storage_id],
        query_id="q0",
        candidate_ids=["note:blue", "note:other"],
        selected_ids=["note:blue"],
        decision="write",
    )
    return CanonicalState(
        state_id=f"state-{storage_id}",
        sources=[source],
        artifacts=[artifact],
        relations=[relation],
        queries=[query],
        provenance_events=[event],
        metadata={"created_at": "different-wall-clock", "physical_id": storage_id},
    )


def test_canonical_comparator_ignores_physical_ids_and_wall_clock_metadata():
    left = _state(storage_id="physical-a")
    right = _state(storage_id="physical-b")
    result = CanonicalStateComparator().compare(left, right)
    assert result.equivalent
    assert result.left_hash == result.right_hash


def test_canonical_comparator_preserves_content_and_support_semantics():
    content_change = _state(content="green")
    source_change = _state(source_id="src-b")
    base = _state()

    content_result = CanonicalStateComparator().compare(base, content_change)
    source_result = CanonicalStateComparator().compare(base, source_change)

    assert not content_result.equivalent
    assert "artifact" in content_result.mismatch_counts
    assert not source_result.equivalent
    assert "source" in source_result.mismatch_counts


def test_query_candidate_order_is_semantic():
    left = _state()
    right = _state()
    right.queries[0].candidate_ids = ["note:other", "note:blue"]
    result = CanonicalStateComparator().compare(left, right)
    assert not result.equivalent
    assert "query" in result.mismatch_counts


def test_source_provenance_from_snapshot_is_stable():
    from bemula.research import InMemoryBackend, ResearchTurn

    backend = InMemoryBackend()
    turn = ResearchTurn(
        dataset="fixture",
        instance_id="i0",
        source_id="src-a",
        speaker="user",
        text="My favorite color is blue.",
    )
    backend.observe(turn.to_turn(), source_id=turn.source_id)
    snapshot = backend.snapshot()
    state = CanonicalState.from_snapshot(snapshot, dataset="fixture", instance_id="i0")
    assert state.sources[0].source_id == "src-a"
    assert state.sources[0].content_hash == snapshot.records[0].source_hash
    assert state.artifacts[0].support_set == []
    assert state.canonical_hash


def test_state_hashes_separate_visible_runtime_and_audit_layers():
    left = _state()
    right = _state()
    right.metadata["runtime"] = {"queue_size": 1}
    right.provenance_events[0].decision = "different_valid_execution"
    assert left.visible_state_hash == right.visible_state_hash
    assert left.runtime_state_hash != right.runtime_state_hash
    assert left.audit_trace_hash != right.audit_trace_hash


def test_provenance_event_hash_is_ordered_and_serializable():
    event = ProvenanceEvent(
        event_id="e0",
        event_type=ProvenanceEventType.NOOP,
        step=2,
        source_ids=["src-a"],
        decision="suppressed_duplicate",
    )
    assert len(event.event_hash) == 64
    assert event.model_dump(mode="json")["event_type"] == "noop"



def test_comparator_treats_provenance_as_audit_by_default():
    left = _state()
    right = _state()
    right.provenance_events[0].decision = "different_valid_execution"
    default_result = CanonicalStateComparator().compare(left, right)
    strict_result = CanonicalStateComparator(include_provenance=True).compare(left, right)
    assert default_result.equivalent
    assert not strict_result.equivalent
    assert "provenance" in strict_result.mismatch_counts
