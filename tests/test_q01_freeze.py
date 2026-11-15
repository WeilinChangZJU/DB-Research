from pathlib import Path
import json

from bemula.research.manifest import sha256_tree
from scripts.build_q01_amem_freeze import build_freeze
from scripts.run_q01_amem_paired import _load_histories
from scripts.run_q01_amem_formal import _resolve_amem_path, _source_tree_files_only, _verify_freeze


def test_formal_source_tree_digest_matches_manifest_policy(tmp_path: Path):
    (tmp_path / "memory_layer_robust.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "module.pyc").write_bytes(b"transient")

    # The formal freeze verifier must hash source content with the same
    # exclusions used when the run manifest/receipt was produced.
    assert _source_tree_files_only(tmp_path) == sha256_tree(tmp_path)


def test_formal_uses_freeze_checkout_path_unless_overridden(tmp_path: Path, monkeypatch):
    checkout = tmp_path / "external" / "A-mem"
    checkout.mkdir(parents=True)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"official_amem": {"path": str(checkout)}}), encoding="utf-8")
    monkeypatch.delenv("BEMULA_AMEM_PATH", raising=False)
    assert _resolve_amem_path(freeze) == checkout.resolve()
    override = tmp_path / "override"
    monkeypatch.setenv("BEMULA_AMEM_PATH", str(override))
    assert _resolve_amem_path(freeze) == override.resolve()


def test_serial_history_loader_rejects_empty_and_duplicate_event_streams(tmp_path: Path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text(json.dumps({"instance_id": "h0", "events": []}) + "\n", encoding="utf-8")
    try:
        _load_histories(empty)
    except ValueError as exc:
        assert "events list" in str(exc)
    else:
        raise AssertionError("empty history should be rejected")

    duplicate = tmp_path / "duplicate.jsonl"
    row = {
        "instance_id": "h1",
        "events": [
            {"event_id": "e0", "source_id": "s0", "content": "a"},
            {"event_id": "e0", "source_id": "s1", "content": "b"},
        ],
        "deleted_source_ids": [],
    }
    duplicate.write_text(json.dumps(row) + "\n", encoding="utf-8")
    try:
        _load_histories(duplicate)
    except ValueError as exc:
        assert "duplicate event IDs" in str(exc)
    else:
        raise AssertionError("duplicate event IDs should be rejected")


def test_freeze_builder_records_commit_and_input_hashes(tmp_path: Path, monkeypatch):
    amem = tmp_path / "A-mem"
    amem.mkdir()
    (amem / "memory_layer_robust.py").write_text("pass\n", encoding="utf-8")
    (amem / "requirements.txt").write_text("openai\n", encoding="utf-8")
    locomo = tmp_path / "locomo.jsonl"
    longmem = tmp_path / "longmem.jsonl"
    locomo.write_text("{}\n", encoding="utf-8")
    longmem.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("scripts.build_q01_amem_freeze._commit", lambda root: "abc123")
    monkeypatch.setattr("scripts.run_q01_amem_formal._git_commit", lambda root: "abc123")

    receipt = build_freeze(
        locomo=locomo,
        longmemeval=longmem,
        amem=amem,
        model="deepseek-flash",
        embedding_backend="local",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        thinking="disabled",
        base_url="https://api.deepseek.com",
        retrieve_k=10,
        expected_commit="abc123",
    )
    assert receipt["official_amem"]["commit"] == "abc123"
    assert receipt["inputs"]["locomo"]["sha256"]
    assert receipt["model_config"]["local_embedding_variant"]["model"].endswith("all-MiniLM-L6-v2")


def test_formal_freeze_verifies_optional_runtime_model_settings(tmp_path: Path, monkeypatch):
    amem = tmp_path / "A-mem"
    amem.mkdir()
    robust = amem / "memory_layer_robust.py"
    requirements = amem / "requirements.txt"
    robust.write_text("pass\n", encoding="utf-8")
    requirements.write_text("openai\n", encoding="utf-8")
    locomo = tmp_path / "locomo.jsonl"
    longmem = tmp_path / "longmem.jsonl"
    locomo.write_text("{}\n", encoding="utf-8")
    longmem.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("scripts.build_q01_amem_freeze._commit", lambda root: "abc123")
    monkeypatch.setattr("scripts.run_q01_amem_formal._git_commit", lambda root: "abc123")
    receipt = build_freeze(
        locomo=locomo,
        longmemeval=longmem,
        amem=amem,
        model="deepseek-flash",
        embedding_backend="local",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        thinking="disabled",
        base_url="https://api.deepseek.com",
        retrieve_k=10,
        expected_commit="abc123",
    )
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setenv("BEMULA_THINKING", "disabled")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    assert _verify_freeze(
        freeze,
        agent_model="deepseek-flash",
        embedding_backend="local",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        retrieve_k=10,
        locomo=locomo,
        longmemeval=longmem,
        amem_path=amem,
    )["model_config"]["thinking"] == "disabled"

