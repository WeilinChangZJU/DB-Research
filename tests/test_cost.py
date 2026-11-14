import json
from pathlib import Path

import pytest

from bemula.research.cost import CostRecorder


def test_cost_recorder_aggregates_operations_and_counters(tmp_path: Path):
    path = tmp_path / "cost.jsonl"
    recorder = CostRecorder(path)
    recorder.record(
        "observe",
        duration_ms=2.5,
        counters={"memory_writes": 1, "retrieval_calls": 1, "retrieved_items": 3},
        instance_id="i0",
    )
    with recorder.span("rebuild", arm="M0"):
        pass
    recorder.record_llm(
        task="summary",
        model="fixture",
        prompt="secret prompt",
        input_tokens=10,
        output_tokens=4,
        duration_ms=1.0,
    )
    summary = recorder.summary()
    assert summary["events"] == 3
    assert summary["operations"]["observe"]["count"] == 1
    assert summary["operations"]["rebuild"]["count"] == 1
    assert summary["totals"]["memory_writes"] == 1
    assert summary["totals"]["llm_calls"] == 1
    assert summary["totals"]["input_tokens"] == 10
    assert summary["totals"]["output_tokens"] == 4
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert all("secret prompt" not in line for line in lines)
    assert json.loads(lines[-1])["prompt_sha256"]


def test_cost_recorder_rejects_non_numeric_counters():
    recorder = CostRecorder()
    with pytest.raises(TypeError):
        recorder.record("retrieve", counters={"retrieved_items": "three"})


def test_cost_recorder_records_failures_and_writes_summary(tmp_path: Path):
    recorder = CostRecorder()
    with pytest.raises(RuntimeError):
        with recorder.span("decision", phase="M3"):
            raise RuntimeError("fixture failure")
    summary_path = tmp_path / "summary.json"
    recorder.write_summary(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["errors"] == 1
    assert summary["operations"]["decision"]["errors"] == 1


def test_cost_recorders_merge_without_copying_raw_sensitive_fields():
    left = CostRecorder()
    right = CostRecorder()
    right.record_llm(task="fixture", prompt="private prompt", input_tokens=2)
    assert left.merge(right, arm="M2") == 1
    assert left.summary()["totals"]["llm_calls"] == 1
    assert "private prompt" not in json.dumps(left.events)


def test_cost_recorder_loads_existing_events_for_resume(tmp_path: Path):
    path = tmp_path / "resume.jsonl"
    first = CostRecorder(path)
    first.record("observe", counters={"memory_writes": 2})
    resumed = CostRecorder(path)
    assert resumed.load_existing() == 1
    resumed.record("observe", counters={"memory_writes": 3})
    assert resumed.summary()["totals"]["memory_writes"] == 5


def test_cost_recorder_keeps_arm_scope_and_full_replay_breakdown():
    from bemula.research.maintenance import BuildCost

    recorder = CostRecorder()
    recorder.record_arm_cost(
        arm="M3_read_set",
        cost=BuildCost(
            input_events=4,
            replayed_events=3,
            prefix_replayed_events=2,
            suffix_replayed_events=1,
            checkpoint_restore_events=0,
            cost_scope="maintenance_full_execution",
        ),
    )
    summary = recorder.summary()
    assert summary["scopes"]["maintenance"]["counters"]["replayed_events"] == 3
    assert summary["operations"]["maintenance_cost"]["counters"]["prefix_replayed_events"] == 2
