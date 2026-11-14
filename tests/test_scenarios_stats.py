import json

from bemula.research import (
    BuildCost,
    BuildStep,
    DeletionScenarioKind,
    SourceEvent,
    StateComparison,
    generate_deletion_scenarios,
    paired_observation,
    summarize_paired_observations,
    write_failure_report,
    paired_bootstrap_mean_difference,
    QueryComparison,
    classify_query_comparison,
    affected_set_precision_recall,
)


def _history():
    return [
        SourceEvent("e0", "src-a", "same approval fact", timestamp="t0"),
        SourceEvent("e1", "src-b", "same approval fact", timestamp="t1"),
        SourceEvent("e2", "src-c", "office in Shanghai", timestamp="t2"),
        SourceEvent("e3", "src-d", "travel next week", timestamp="t3"),
    ]


def test_deletion_scenarios_are_deterministic_and_trace_aware(tmp_path):
    trace = [
        BuildStep(
            index=1,
            event_id="e1",
            source_id="src-b",
            operation="update",
            read_set=("src-a",),
            candidate_ids=("src-a",),
            output_ids=("fact:approval",),
        ),
        BuildStep(index=2, event_id="e2", source_id="src-c", operation="noop"),
    ]
    first = generate_deletion_scenarios(_history(), dataset="fixture", history_id="h0", trace=trace)
    second = generate_deletion_scenarios(_history(), dataset="fixture", history_id="h0", trace=trace)

    assert first.to_dict() == second.to_dict()
    kinds = {scenario.kind for scenario in first.scenarios}
    assert DeletionScenarioKind.SHARED_SUPPORT in kinds
    assert DeletionScenarioKind.ISOLATED in kinds
    assert DeletionScenarioKind.TOP_K_BOUNDARY in kinds
    assert DeletionScenarioKind.NOOP in kinds
    assert DeletionScenarioKind.RETRIEVAL_ONLY in kinds
    assert DeletionScenarioKind.LINKED_NEIGHBOR in kinds
    output = first.write_json(tmp_path / "manifest.json")
    assert json.loads(output.read_text(encoding="utf-8"))["manifest_id"] == first.manifest_id


def test_paired_stats_classify_failures_and_write_report(tmp_path):
    equal = StateComparison(True, "a", "a")
    changed = StateComparison(False, "a", "b", changed_ids=("note:1",), reason="visible state differs")
    observations = [
        paired_observation(
            scenario_id="s0", arm="M2", comparison=equal,
            cost=BuildCost(replayed_events=2, retrieved_candidates=4),
        ),
        paired_observation(
            scenario_id="s1", arm="M2", comparison=changed,
            cost=BuildCost(replayed_events=4, retrieved_candidates=8),
        ),
    ]
    summaries = summarize_paired_observations(observations, reference_replayed_events={"s0": 4, "s1": 4})
    assert summaries["M2"].scenario_count == 2
    assert summaries["M2"].exact_count == 1
    assert summaries["M2"].exact_rate == 0.5
    assert summaries["M2"].failure_counts["state_mismatch"] == 1
    # Incorrect observations are excluded from an efficiency claim; a cheap
    # wrong state must not be reported as a maintenance saving.
    assert summaries["M2"].cost_savings_vs_reference == 0.5

    report = write_failure_report(tmp_path / "failures.json", observations, summaries)
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert len(payload["observations"]) == 2
    assert payload["observations"][1]["failures"][0]["kind"] == "state_mismatch"


def test_paired_bootstrap_is_reproducible():
    first = paired_bootstrap_mean_difference([1, 2, 3], [2, 2, 2], samples=100, seed=7)
    second = paired_bootstrap_mean_difference([1, 2, 3], [2, 2, 2], samples=100, seed=7)
    assert first == second
    assert first["estimate"] == 0.0
    assert first["n"] == 3


def test_incorrect_low_cost_is_not_reported_as_saving():
    changed = StateComparison(False, "a", "b", changed_ids=("note:1",))
    observations = [
        paired_observation(
            scenario_id="s0", arm="M1", comparison=changed,
            cost=BuildCost(replayed_events=0),
        )
    ]
    summary = summarize_paired_observations(
        observations,
        reference_replayed_events={"s0": 4},
    )
    assert summary["M1"].cost_savings_vs_reference is None


def test_query_failure_classification_is_explicit():
    failures = classify_query_comparison(
        QueryComparison(False, False, ("src-a",)),
        scenario_id="s0",
        arm="M1",
    )
    assert {failure.kind.value for failure in failures} == {
        "query_mismatch",
        "evidence_mismatch",
        "old_source_exposure",
    }


def test_affected_set_precision_recall_is_explicit_about_over_invalidation():
    score = affected_set_precision_recall(["a", "b"], ["a"])
    assert score["true_positive"] == 1
    assert score["false_positive"] == 1
    assert score["false_negative"] == 0
    assert score["precision"] == 0.5
    assert score["recall"] == 1.0
