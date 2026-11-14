from __future__ import annotations

import json
from pathlib import Path

from bemula.research.datasets import load_locomo, load_longmemeval_s
from bemula.research.backend import InMemoryBackend
from bemula.research.protocol import ResearchProtocol


def test_locomo_adapter_preserves_dia_ids_and_evidence(tmp_path: Path) -> None:
    source = tmp_path / "locomo10.json"
    source.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv-test",
                    "conversation": {
                        "speaker_a": "A",
                        "speaker_b": "B",
                        "session_1_date_time": "2024-01-01",
                        "session_1": [
                            {"speaker": "A", "dia_id": "D1:1", "text": "first"},
                            {"speaker": "B", "dia_id": "D1:2", "text": "second"},
                        ],
                    },
                    "qa": [
                        {"question": "What?", "answer": "first", "evidence": ["D1:1"], "category": 1}
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    instance = load_locomo(source)[0]
    assert instance.instance_id == "locomo:conv-test"
    assert instance.source_records[0].source_id.endswith(":D1:1")
    assert instance.source_records[0].source_hash
    assert instance.source_records[0].timestamp == "2024-01-01"
    assert instance.items[0].source_ids == (instance.source_records[0].source_id,)
    assert instance.items[0].metadata["evidence_resolved"] is True
    canonical = instance.to_protocol_instance()
    assert canonical.turns[0].source_id == instance.source_records[0].source_id
    assert canonical.items[0].related_source_ids == [instance.source_records[0].source_id]


def test_longmemeval_adapter_resolves_answer_sessions(tmp_path: Path) -> None:
    source = tmp_path / "longmemeval_s.json"
    source.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "single-session-user",
                    "question": "What is my degree?",
                    "question_date": "2024/01/02",
                    "answer": "Physics",
                    "answer_session_ids": ["s1"],
                    "haystack_dates": ["2024/01/01", "2024/01/02"],
                    "haystack_session_ids": ["s1", "s2"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "I studied Physics."}],
                        [{"role": "assistant", "content": "Okay."}],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    instance = load_longmemeval_s(source)[0]
    assert instance.dataset == "longmemeval_s"
    assert instance.items[0].gold_answer == "Physics"
    assert len(instance.items[0].source_ids) == 1
    assert instance.items[0].metadata["support_sessions_resolved"] is True
    assert instance.source_records[0].speaker == "user"
    canonical = instance.to_protocol_instance()
    assert canonical.turns[0].source_hash == instance.source_records[0].source_hash


def test_longmemeval_alias_supports_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "longmemeval.jsonl"
    row = {
        "question_id": "q1",
        "question": "Q",
        "answer": "A",
        "answer_session_ids": [],
        "haystack_dates": [],
        "haystack_session_ids": [],
        "haystack_sessions": [],
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    instance = load_longmemeval_s(source)[0]
    assert instance.instance_id.endswith(":q1")
    assert instance.items[0].source_ids == ()


def test_locomo_adapter_runs_through_research_protocol(tmp_path: Path) -> None:
    source = tmp_path / "locomo10.json"
    source.write_text(
        json.dumps(
            [{
                "sample_id": "conv-protocol",
                "conversation": {
                    "session_1_date_time": "2024-01-01",
                    "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "I like tea."}],
                },
                "qa": [{"question": "What do I like?", "answer": "tea", "evidence": ["D1:1"]}],
            }]
        ),
        encoding="utf-8",
    )
    instance = load_locomo(source)[0]
    result = ResearchProtocol(InMemoryBackend()).run_instance(
        instance, control_items=instance.items, forget_items=instance.items
    )
    assert result.instance_id == instance.instance_id
    assert result.control_queries[0].item_id == instance.items[0].item_id
    assert result.control_queries[0].evidence["source_ids"]
