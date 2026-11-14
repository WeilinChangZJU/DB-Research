"""LongMemEval-S (also written LongMenEval-S) adapter.

The official repository archive may omit the large LFS data files, so this
adapter accepts an external JSON/JSONL source path.  The cleaned JSON used in
this workspace is a list of question records with ``haystack_sessions``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .common import ResearchInstance, ResearchItem, SourceRecord, file_hash

DATASET_NAME = "longmemeval_s"
ALIASES = ("LongMemEval-S", "LongMenEval-S", "longmemeval", "longmeneval")


def _load_payload(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"LongMemEval JSONL row is not an object: {path}")
                    rows.append(value)
        return rows
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, dict):
        for key in ("data", "questions", "items"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"LongMemEval source must be a list of objects: {path}")
    return payload


def iter_longmemeval_s(
    source_path: str | Path,
    *,
    limit: int | None = None,
) -> Iterator[ResearchInstance]:
    """Yield one ResearchInstance per LongMemEval-S question."""
    path = Path(source_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LongMemEval-S source file not found: {path}")
    digest = file_hash(path)
    rows = _load_payload(path)
    for index, raw in enumerate(rows):
        if limit is not None and index >= limit:
            break
        question_id = str(raw.get("question_id") or raw.get("id") or f"question-{index:06d}")
        instance_id = f"{DATASET_NAME}:{question_id}"
        session_ids = [str(value) for value in (raw.get("haystack_session_ids") or [])]
        dates = [str(value) for value in (raw.get("haystack_dates") or [])]
        haystack = raw.get("haystack_sessions") or []
        if not isinstance(haystack, list):
            raise ValueError(f"LongMemEval record {question_id} has invalid haystack_sessions")

        sessions: list[list[SourceRecord]] = []
        session_source_ids: dict[str, list[str]] = {}
        ordinal = 0
        for session_index, messages in enumerate(haystack):
            session_id = session_ids[session_index] if session_index < len(session_ids) else f"session_{session_index}"
            timestamp = dates[session_index] if session_index < len(dates) else None
            if not isinstance(messages, list):
                messages = []
            session_records: list[SourceRecord] = []
            for turn_index, message in enumerate(messages):
                if not isinstance(message, dict):
                    continue
                speaker = str(message.get("role") or message.get("speaker") or "unknown")
                text = str(message.get("content") or message.get("text") or "")
                source_id = f"{DATASET_NAME}:{question_id}:{session_id}:{turn_index}"
                record = SourceRecord(
                    dataset=DATASET_NAME,
                    instance_id=instance_id,
                    source_id=source_id,
                    text=text,
                    speaker=speaker,
                    timestamp=timestamp,
                    ordinal=ordinal,
                    metadata={
                        "session_id": session_id,
                        "session_index": session_index,
                        "turn_index": turn_index,
                        "raw_role": message.get("role"),
                    },
                )
                ordinal += 1
                session_records.append(record)
                session_source_ids.setdefault(session_id, []).append(source_id)
            sessions.append(session_records)

        answer_sessions = [str(value) for value in (raw.get("answer_session_ids") or [])]
        support_source_ids = tuple(
            source_id
            for session_id in answer_sessions
            for source_id in session_source_ids.get(session_id, [])
        )
        item_id = f"{instance_id}:qa"
        item = ResearchItem(
            dataset=DATASET_NAME,
            instance_id=instance_id,
            item_id=item_id,
            question=str(raw.get("question") or ""),
            gold_answer=raw.get("answer", ""),
            source_ids=support_source_ids,
            grading="llm_judge",
            metadata={
                "question_id": question_id,
                "question_type": raw.get("question_type"),
                "question_date": raw.get("question_date"),
                "answer_session_ids": answer_sessions,
                "haystack_session_ids": session_ids,
                "support_sessions_resolved": all(sid in session_source_ids for sid in answer_sessions),
                "raw_record_index": index,
            },
        )
        yield ResearchInstance(
            dataset=DATASET_NAME,
            instance_id=instance_id,
            sessions=sessions,
            items=[item],
            source_path=str(path),
            source_hash=digest,
            metadata={
                "question_id": question_id,
                "dataset_aliases": list(ALIASES),
                "question_type": raw.get("question_type"),
                "question_date": raw.get("question_date"),
                "num_sessions": len(sessions),
                "num_items": 1,
                "raw_record_index": index,
            },
        )


def load_longmemeval_s(source_path: str | Path, *, limit: int | None = None) -> list[ResearchInstance]:
    return list(iter_longmemeval_s(source_path, limit=limit))


# Compatibility aliases for the spelling used in some project notes.
iter_longmemeval = iter_longmemeval_s
load_longmemeval = load_longmemeval_s
iter_longmeneval_s = iter_longmemeval_s
load_longmeneval_s = load_longmemeval_s
