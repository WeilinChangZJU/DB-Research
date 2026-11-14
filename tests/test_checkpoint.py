import json
from pathlib import Path

import pytest

from bemula.research.checkpoint import CheckpointStore, CheckpointValidationError


def test_checkpoint_commit_is_atomic_and_resumable(tmp_path: Path):
    store = CheckpointStore(tmp_path / "run")
    records = [{"instance_id": "inst:1", "item_id": "q0", "correct": True}]
    entry = store.commit("inst:1", records)
    assert entry["record_count"] == 1
    assert store.completed_ids() == {"inst:1"}
    assert store.load_records() == records
    out = tmp_path / "items.jsonl"
    assert store.export_items(out) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["item_id"] == "q0"
    assert store.validate() == []


def test_checkpoint_detects_tampering(tmp_path: Path):
    store = CheckpointStore(tmp_path / "run")
    store.commit("inst/unsafe", [{"x": 1}])
    path = next((tmp_path / "run" / "checkpoints" / "instances").glob("*.jsonl"))
    path.write_text('{"x": 2}\n', encoding="utf-8")
    assert store.completed_ids() == set()
    with pytest.raises(CheckpointValidationError):
        store.load_records()


def test_semantic_state_checkpoint_round_trip(tmp_path: Path):
    from bemula.research import CanonicalState

    store = CheckpointStore(tmp_path / "run")
    state = CanonicalState(state_id="state-0", metadata={"builder": "fixture"})
    entry = store.commit_state("state-0", state, metadata={"arm": "M0"})
    assert entry["state_id"] == "state-0"
    envelope = store.load_state("state-0")
    assert envelope["state"]["state_id"] == "state-0"
    assert envelope["metadata"]["arm"] == "M0"


def test_checkpoint_pins_run_fingerprint_and_rejects_mismatch(tmp_path: Path):
    store = CheckpointStore(tmp_path / "run")
    first = store.ensure_fingerprint({"model": "fixture", "seed": 7})
    assert first["sha256"] == store.load_fingerprint()["sha256"]
    assert store.ensure_fingerprint({"model": "fixture", "seed": 7})["sha256"] == first["sha256"]
    with pytest.raises(CheckpointValidationError, match="fingerprint mismatch"):
        store.ensure_fingerprint({"model": "fixture", "seed": 8})
