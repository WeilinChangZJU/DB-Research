from types import SimpleNamespace

from scripts.build_q01_amem_histories import _history_payload, _mask_ordinals
from scripts.build_q01_focused_window_pilot import _window


def test_history_mask_policies_are_deterministic_and_in_bounds():
    assert _mask_ordinals(10, "ordinal_quartile_single_v1") == [2]
    assert _mask_ordinals(10, "ordinal_quartiles_pair_v1") == [2, 7]
    assert _mask_ordinals(10, "ordinal_late_single_v1") == [9]
    assert all(0 <= value < 3 for value in _mask_ordinals(3, "ordinal_quartiles_pair_v1"))


def test_history_payload_carries_explicit_mask_and_provenance(tmp_path):
    records = [
        SimpleNamespace(
            dataset="pilot",
            instance_id="h0",
            source_id="src-0",
            text="alpha",
            timestamp="t0",
            source_hash="hash-alpha",
            metadata={"speaker": "user"},
        ),
        SimpleNamespace(
            dataset="pilot",
            instance_id="h0",
            source_id="src-1",
            text="beta",
            timestamp="t1",
            source_hash="hash-beta",
            metadata={"speaker": "assistant"},
        ),
    ]
    instance = SimpleNamespace(
        dataset="pilot", instance_id="h0", source_records=records, items=["q"]
    )

    payload = _history_payload(instance, tmp_path / "source.json", "file-sha", "ordinal_late_single_v1")

    assert payload["deleted_event_ordinals"] == [1]
    assert payload["deleted_source_ids"] == ["src-1"]
    assert payload["source_file_sha256"] == "file-sha"
    assert payload["events"][0]["metadata"]["source_record_hash"] == "hash-alpha"
    assert payload["events"][0]["event_hash"]
    assert payload["history_manifest_sha256"]


def test_history_payload_preserves_query_overlay_without_gold_answer(tmp_path):
    instance = SimpleNamespace(
        dataset="pilot",
        instance_id="h1",
        source_records=[],
        items=[
            SimpleNamespace(
                item_id="q1",
                question="Where did the event occur?",
                source_ids=("src-0",),
                grading="llm_judge",
                metadata={"category": "factual"},
                gold_answer="secret answer",
            )
        ],
    )
    payload = _history_payload(instance, tmp_path / "source.json", "file-sha", "ordinal_late_single_v1")
    assert payload["items"] == [
        {
            "item_id": "q1",
            "question": "Where did the event occur?",
            "source_ids": ["src-0"],
            "grading": "llm_judge",
            "metadata": {"category": "factual"},
        }
    ]
    assert "gold_answer" not in payload["items"][0]


def test_focused_window_keeps_deleted_source_and_records_parent_manifest(tmp_path):
    parent = {
        "dataset": "pilot",
        "instance_id": "h0",
        "history_manifest_sha256": "parent-sha",
        "num_source_events": 5,
        "deleted_source_ids": ["src-3"],
        "events": [{"source_id": f"src-{i}", "content": str(i)} for i in range(5)],
        "items": [],
    }
    payload = _window(parent, tmp_path / "source.jsonl", start=2, length=3)
    assert payload["deleted_event_ordinals"] == [1]
    assert payload["deleted_source_ids"] == ["src-3"]
    assert payload["parent_history_manifest_sha256"] == "parent-sha"
    assert payload["selection"] == {"kind": "contiguous_window", "start": 2, "length": 3}
