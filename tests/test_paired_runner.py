from bemula.research.paired_runner import run_paired_histories, run_paired_instance
from bemula.research.datasets.common import ResearchInstance, SourceRecord


def _instance(instance_id: str = "h0") -> ResearchInstance:
    records = [
        SourceRecord("fixture", instance_id, "a", "approval is ten", "user", ordinal=0),
        SourceRecord("fixture", instance_id, "b", "approval is ten", "user", ordinal=1),
        SourceRecord("fixture", instance_id, "c", "office Shanghai", "user", ordinal=2),
    ]
    return ResearchInstance(
        dataset="fixture",
        instance_id=instance_id,
        sessions=[records],
        items=[],
        source_path="fixture",
        source_hash="fixture",
    )


def test_paired_instance_keeps_all_arms_and_history_identity():
    result = run_paired_instance(_instance(), max_scenarios=2)
    assert result.dataset == "fixture"
    assert result.instance_id == "h0"
    assert len(result.executed_scenarios) == 2
    assert len(result.manifest.scenarios) >= 2
    assert {item.arm for item in result.observations} == {
        "M0_full_rebuild",
        "M1_tombstone_filter",
        "M2_replay_repair_all",
        "M3_exact_read_set",
    }
    assert all(item.metadata["evidence_level"] == "controlled_replay" for item in result.observations)


def test_paired_histories_reports_clustered_unit_and_bootstrap():
    payload = run_paired_histories([_instance("h0"), _instance("h1")], max_scenarios=1)
    assert payload["independent_history_count"] == 2
    assert payload["aggregation_unit"] == "scenario_with_history_cluster"
    assert payload["scenario_count"] == 2
    assert "M3_exact_read_set" in payload["summaries"]
    assert payload["paired_bootstrap_replay_difference"]["M3_exact_read_set"]["n"] == 2
