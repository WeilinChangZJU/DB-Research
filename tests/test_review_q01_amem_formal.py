import json

from scripts.review_q01_amem_formal import _load_run


def _cost(**overrides):
    value = {
        "wall_time_ms": 2.5,
        "input_events": 4,
        "replayed_events": 4,
        "llm_calls": 3,
        "input_tokens": 30,
        "output_tokens": 12,
        "embedding_calls": 4,
        "retrieved_candidates": 5,
    }
    value.update(overrides)
    return value


def test_reviewer_reports_paired_full_cost_and_components(tmp_path):
    run = tmp_path / "run"
    instances = run / "checkpoints" / "instances"
    instances.mkdir(parents=True)
    row = {
        "instance_id": "h0",
        "arms": {
            "M0_full_rebuild": {
                "exact_to_m0": True,
                "cost": _cost(),
                "metadata": {},
            },
            "M1_tombstone_filter": {
                "exact_to_m0": False,
                "cost": _cost(wall_time_ms=2.5, llm_calls=0, input_tokens=0),
                "metadata": {
                    "baseline_setup_cost": _cost(
                        wall_time_ms=10.0,
                        input_events=8,
                        replayed_events=8,
                        llm_calls=20,
                        input_tokens=200,
                        output_tokens=80,
                        embedding_calls=8,
                    )
                },
            },
        },
    }
    (instances / "h0.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = _load_run("locomo", run, expected=1)
    m0 = report["arm_summary"]["M0_full_rebuild"]
    m1 = report["arm_summary"]["M1_tombstone_filter"]

    # M0 has no paired setup, while M1's full execution includes both the
    # fresh baseline and the post-deletion maintenance operation.
    assert m0["full_llm_calls"] == 3
    assert m0["baseline_setup_llm_calls"] == 0
    assert m1["maintenance_llm_calls"] == 0
    assert m1["baseline_setup_llm_calls"] == 20
    assert m1["full_llm_calls"] == 20
    assert m1["llm_calls"] == 20
    assert m1["full_wall_time_ms"] == 12.5
    assert m1["maintenance_wall_time_ms"] == 2.5
    assert m1["baseline_setup_wall_time_ms"] == 10.0
    assert m1["full_replayed_events"] == 12

