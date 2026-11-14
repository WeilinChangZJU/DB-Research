from bemula.research import CostRecorder, MaintenanceArm, SourceEvent, run_toy_experiment


def test_q01_runner_compares_all_arms_and_records_costs():
    events = [
        SourceEvent("e0", "src-a", "approval limit is ten thousand"),
        SourceEvent("e1", "src-b", "approval limit is ten thousand"),
        SourceEvent("e2", "src-c", "office is in Shanghai"),
    ]
    cost = CostRecorder()
    result = run_toy_experiment(events, ["src-a"], cost=cost)
    assert result.reference.arm is MaintenanceArm.M0_FULL_REBUILD
    assert result.arms[MaintenanceArm.M0_FULL_REBUILD].exact_to_reference
    assert result.arms[MaintenanceArm.M2_REPLAY].exact_to_reference
    assert result.arms[MaintenanceArm.M3_READ_SET].exact_to_reference
    assert not result.arms[MaintenanceArm.M1_TOMBSTONE].exact_to_reference
    summary = result.summary()
    assert summary["arms"][MaintenanceArm.M0_FULL_REBUILD.value]["exact_to_reference"]
    assert cost.summary()["totals"]["rebuild_units"] > 0
