from bemula.research import DependencyGraph, ProvenanceEvent, ProvenanceEventType


def test_dependency_graph_builds_and_closes_observed_edges():
    events = [
        ProvenanceEvent(
            event_id="e1",
            event_type=ProvenanceEventType.SELECT,
            step=1,
            source_ids=["src-a"],
            candidate_ids=["note-a"],
            selected_ids=["note-a"],
            object_ids=["answer-a"],
            control_dependencies=["note-a"],
        )
    ]
    graph = DependencyGraph.from_events(events)
    closure = graph.affected_closure(["src-a"])
    assert "decision:1:select" in closure
    assert "answer-a" in closure
    assert graph.affected_steps(["src-a"]) == [1]
    assert graph.to_dict()["complete"] is False


def test_dependency_graph_versions_in_place_updates_without_self_cycle():
    events = [
        ProvenanceEvent(
            event_id="e1",
            event_type=ProvenanceEventType.UPDATE,
            step=1,
            source_ids=["src-b"],
            candidate_ids=["note-a"],
            object_ids=["note-a"],
            control_dependencies=["note-a"],
            metadata={
                "candidate_versions": {"note-a": "old@0"},
                "object_versions": {"note-a": "new@1"},
            },
        )
    ]
    graph = DependencyGraph.from_events(events)
    edges = graph.to_dict()["edges"]
    assert {edge["upstream"] for edge in edges} >= {"note-a@old@0", "decision:1:update"}
    assert {edge["downstream"] for edge in edges} >= {"decision:1:update", "note-a@new@1"}
    assert not any(
        edge["upstream"] == "note-a" and edge["downstream"] == "decision:1:update"
        for edge in edges
    )


def test_versioned_output_keeps_logical_id_in_affected_closure():
    event = ProvenanceEvent(
        event_id="e1",
        event_type=ProvenanceEventType.WRITE,
        step=1,
        source_ids=["src-a"],
        object_ids=["note-a"],
        metadata={"object_versions": {"note-a": "v1"}},
    )
    graph = DependencyGraph.from_events([event])
    closure = graph.affected_closure(["src-a"])
    assert "note-a@v1" in closure
    assert "note-a" in closure
