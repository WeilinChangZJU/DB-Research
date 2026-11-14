from bemula.research.maintenance import (
    DeterministicMemoryBuilder,
    MaintenanceArm,
    SourceEvent,
    compare_snapshots,
    run_all_arms,
)


def _events():
    return [
        SourceEvent("e0", "src-a", "approval limit is ten thousand"),
        SourceEvent("e1", "src-b", "approval limit is ten thousand"),
        SourceEvent("e2", "src-c", "office is in Shanghai"),
        SourceEvent("e3", "src-d", "approval limit is ten thousand"),
    ]


def test_m0_reference_and_m2_m3_replay_match_after_delete():
    runs = run_all_arms(_events(), {"src-a"})
    assert runs[MaintenanceArm.M0_FULL_REBUILD][1].equal
    assert runs[MaintenanceArm.M2_REPLAY][1].equal
    assert runs[MaintenanceArm.M3_READ_SET][1].equal
    assert runs[MaintenanceArm.M2_REPLAY][0].cost.replayed_events < len(_events())
    assert runs[MaintenanceArm.M3_READ_SET][0].metadata["replay_start"] == 0


def test_m1_is_intentionally_weak_for_shared_support():
    runs = run_all_arms(_events(), {"src-a"})
    result, comparison = runs[MaintenanceArm.M1_TOMBSTONE]
    assert not comparison.equal
    assert any(item.startswith("fact:") for item in comparison.missing_ids)
    assert result.snapshot.tombstones == ["src-a"]


def test_m3_records_affected_steps_and_state_hash_is_stable():
    builder = DeterministicMemoryBuilder()
    events = _events()
    baseline = builder.build(events)
    result = builder.run_arm(events, {"src-d"}, MaintenanceArm.M3_READ_SET, baseline=baseline)
    reference = builder.run_arm(events, {"src-d"}, MaintenanceArm.M0_FULL_REBUILD)
    assert result.metadata["affected_steps"]
    assert compare_snapshots(result.snapshot, reference.snapshot).equal
    assert result.snapshot.state_hash == reference.snapshot.state_hash


def test_replay_does_not_skip_events_after_nonzero_earliest_delete():
    events = [
        SourceEvent("e0", "src-0", "alpha note"),
        SourceEvent("e1", "src-1", "remove me"),
        SourceEvent("e2", "src-2", "beta note"),
        SourceEvent("e3", "src-3", "gamma note"),
    ]
    runs = run_all_arms(events, {"src-1"})
    reference = runs[MaintenanceArm.M0_FULL_REBUILD][0].snapshot
    assert runs[MaintenanceArm.M2_REPLAY][0].snapshot.records == reference.records
    assert runs[MaintenanceArm.M3_READ_SET][0].snapshot.records == reference.records


def test_m2_and_m3_costs_include_rebuilt_prefix_and_report_scope():
    events = _events()
    runs = run_all_arms(events, {"src-d"})
    for arm in (MaintenanceArm.M2_REPLAY, MaintenanceArm.M3_READ_SET):
        result = runs[arm][0]
        assert result.cost.replayed_events == (
            result.cost.prefix_replayed_events + result.cost.suffix_replayed_events
        )
        assert result.cost.cost_scope == "maintenance_full_execution"
        assert result.metadata["checkpoint_restored"] is False
    assert runs[MaintenanceArm.M3_READ_SET][0].metadata["m3_status"].startswith("implemented_")
