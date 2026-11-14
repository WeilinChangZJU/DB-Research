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
