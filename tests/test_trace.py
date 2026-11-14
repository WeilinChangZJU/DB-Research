from pathlib import Path

from bemula.research.trace import TraceRecorder


def test_trace_records_spans_and_hashes(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    trace = TraceRecorder(path)
    with trace.span("memory_operation", operation="observe"):
        pass
    trace.record_llm(task="judge", model="test", prompt="secret prompt", duration_ms=1.5)
    trace.record_memory(operation="delete", source_ids=["s1"], before_hash="a", after_hash="b")
    assert TraceRecorder.hash_text("secret prompt")
    assert len(trace.events) == 3
    summary = trace.summary()
    assert summary["by_kind"]["llm_call"] == 1
    assert summary["by_kind"]["memory_operation"] == 2
    text = path.read_text(encoding="utf-8")
    assert "secret prompt" not in text
