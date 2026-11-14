from bemula.research import (
    BuildCost,
    CanonicalStateComparator,
    MaintenanceArm,
    SourceEvent,
    evaluate_maintenance_outputs,
    compare_query_results,
    old_source_exposure,
    run_all_arms,
    snapshot_to_canonical,
    survivor_loss,
)


def test_maintenance_outputs_use_semantic_state_not_physical_ids():
    events = [
        SourceEvent("e0", "src-a", "approval limit is ten thousand"),
        SourceEvent("e1", "src-b", "approval limit is ten thousand"),
        SourceEvent("e2", "src-c", "office is in Shanghai"),
    ]
    outputs = run_all_arms(events, ["src-a"])
    report = evaluate_maintenance_outputs(outputs, dataset="fixture", instance_id="i0")
    assert report.arms[MaintenanceArm.M0_FULL_REBUILD.value].exact_to_reference
    assert report.arms[MaintenanceArm.M2_REPLAY.value].exact_to_reference
    assert report.arms[MaintenanceArm.M3_READ_SET.value].exact_to_reference
    assert not report.arms[MaintenanceArm.M1_TOMBSTONE.value].exact_to_reference
    assert report.reference.canonical_hash


def test_exposure_and_survivor_helpers_are_explicit():
    events = [SourceEvent("e0", "src-a", "alpha")]
    result = run_all_arms(events, ["src-a"])[MaintenanceArm.M0_FULL_REBUILD][0]
    assert old_source_exposure(result.snapshot, ["src-a"]) == []
    assert survivor_loss(result.snapshot, ["src-a"]) == ["src-a"]


def test_cost_vector_has_lifecycle_fields():
    cost = BuildCost(llm_calls=2, embedding_calls=1, index_updates=3, wall_time_ms=4.5)
    assert cost.llm_calls == 2
    assert cost.index_updates == 3
    assert cost.wall_time_ms == 4.5


def test_query_comparison_detects_old_source_exposure():
    from bemula.research import QueryEvidence, QueryResult

    reference = QueryResult(answer="office", evidence=QueryEvidence(source_ids=["src-b"], read_set=["src-b"]))
    exposed = QueryResult(answer="office", evidence=QueryEvidence(source_ids=["src-a"], read_set=["src-a"]))
    comparison = compare_query_results(exposed, reference, deleted_source_ids=["src-a"])
    assert not comparison.equivalent
    assert comparison.old_source_exposure == ("src-a",)
