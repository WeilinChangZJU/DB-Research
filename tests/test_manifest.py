from pathlib import Path

from bemula.research.manifest import build_manifest, sha256_file, sha256_tree, write_manifest


def test_hashes_and_manifest_are_deterministic(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    (root / "b.txt").write_text("beta", encoding="utf-8")
    assert sha256_file(root / "a.txt")
    first = sha256_tree(root)
    (root / "b.txt").write_text("changed", encoding="utf-8")
    assert sha256_tree(root) != first

    manifest = build_manifest(root=root, config_paths=[root / "a.txt"], data_paths=[root])
    assert manifest["schema_version"] == 1
    assert manifest["inputs"]["config"][0]["sha256"] == sha256_file(root / "a.txt")
    assert "OPENAI_API_KEY" not in manifest["environment"]
    out = tmp_path / "manifest.json"
    write_manifest(out, manifest)
    assert out.is_file()


def test_manifest_redacts_configured_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-value")
    manifest = build_manifest(root=tmp_path)
    value = manifest["environment"]["OPENAI_API_KEY"]
    assert value["configured"] is True
    assert "sk-test-secret-value" not in str(manifest)
    assert len(value["sha256"]) == 12
