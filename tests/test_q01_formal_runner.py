import json
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import run_q01_amem_formal as formal


def _prepare(monkeypatch, tmp_path: Path, *extra: str):
    freeze = tmp_path / "freeze.json"
    locomo = tmp_path / "locomo.jsonl"
    longmemeval = tmp_path / "longmemeval.jsonl"
    out_root = tmp_path / "formal-run"
    review_out = tmp_path / "review.json"
    freeze.write_text("{}\n", encoding="utf-8")
    locomo.write_text("{}\n", encoding="utf-8")
    longmemeval.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        formal,
        "_verify_freeze",
        lambda *args, **kwargs: {
            "model_config": {"model": "deepseek-flash"},
            "official_amem": {},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_q01_amem_formal.py",
            "--freeze",
            str(freeze),
            "--locomo",
            str(locomo),
            "--longmemeval",
            str(longmemeval),
            "--out-root",
            str(out_root),
            "--review-out",
            str(review_out),
            *extra,
        ],
    )
    return out_root


def test_external_blocker_requires_explicit_resume_without_starting_child(monkeypatch, tmp_path):
    out_root = _prepare(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    out_root.mkdir(parents=True)
    blocker_path = out_root / "formal_run_blocker.json"
    blocker_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "blocked_external_model_run",
                "message": "ProviderBudgetError: insufficient balance",
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fail_if_started(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("an external blocker must be retried only with --resume")

    monkeypatch.setattr(formal.subprocess, "run", fail_if_started)

    assert formal.main() == 3
    assert calls == []
    blocker = json.loads(blocker_path.read_text(encoding="utf-8"))
    assert blocker["status"] == "blocked_external_model_run"
    assert "--resume" in blocker["message"]


def test_missing_key_does_not_overwrite_external_blocker(monkeypatch, tmp_path):
    out_root = _prepare(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out_root.mkdir(parents=True)
    blocker_path = out_root / "formal_run_blocker.json"
    blocker_path.write_text(
        json.dumps({"status": "blocked_external_model_run"}), encoding="utf-8"
    )

    assert formal.main() == 2
    blocker = json.loads(blocker_path.read_text(encoding="utf-8"))
    assert blocker["status"] == "blocked_external_model_run"


def test_resume_after_external_blocker_clears_receipt_after_success(monkeypatch, tmp_path):
    out_root = _prepare(monkeypatch, tmp_path, "--resume")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    out_root.mkdir(parents=True)
    blocker_path = out_root / "formal_run_blocker.json"
    blocker_path.write_text(
        json.dumps({"status": "blocked_external_model_run"}), encoding="utf-8"
    )
    calls = []

    def fake_run(command, *args, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(formal.subprocess, "run", fake_run)

    assert formal.main() == 0
    assert len(calls) == 3  # two workload children and the final reviewer
    assert not blocker_path.exists()
