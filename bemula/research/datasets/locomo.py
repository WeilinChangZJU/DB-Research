"""LoCoMo -> ResearchProtocol adapter.

The adapter accepts the official ``locomo10.json`` format.  It preserves the
LoCoMo ``dia_id`` values (for example ``D1:3``) as stable source IDs and maps
QA evidence to those IDs.  The source file is read in place and is never copied
into the Bemula repository.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from .common import ResearchInstance, ResearchItem, SourceRecord, file_hash

_SESSION_RE = re.compile(r"^session_(\d+)$")


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, dict):
        for key in ("data", "conversations", "items"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"LoCoMo source must be a JSON list: {path}")
    return payload


def _session_numbers(conversation: dict[str, Any]) -> list[int]:
    numbers = []
    for key, value in conversation.items():
        match = _SESSION_RE.match(str(key))
        if match and isinstance(value, list):
            numbers.append(int(match.group(1)))
    return sorted(set(numbers))


def _coerce_answer(answer: Any) -> Any:
    """Keep numeric/date answers intact while normalising absent answers."""
    return "" if answer is None else answer


def iter_locomo(
    source_path: str | Path,
    *,
    limit: int | None = None,
) -> Iterator[ResearchInstance]:
    """Yield LoCoMo conversations as ResearchProtocol instances."""
    path = Path(source_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LoCoMo source file not found: {path}")
    digest = file_hash(path)
    records = _load_records(path)
    for index, raw in enumerate(records):
        if limit is not None and index >= limit:
            break
        sample_id = str(raw.get("sample_id") or raw.get("id") or f"sample-{index:04d}")
        instance_id = f"locomo:{sample_id}"
        conversation = raw.get("conversation")
        if not isinstance(conversation, dict):
            raise ValueError(f"LoCoMo record {sample_id} has no conversation object")

        sessions: list[list[SourceRecord]] = []
        source_index: dict[str, SourceRecord] = {}
        ordinal = 0
        session_meta: list[dict[str, Any]] = []
        for session_number in _session_numbers(conversation):
            key = f"session_{session_number}"
            turns = conversation.get(key) or []
            timestamp = conversation.get(f"{key}_date_time")
            session_records: list[SourceRecord] = []
            for turn_index, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    continue
                dia_id = str(turn.get("dia_id") or f"D{session_number}:{turn_index + 1}")
                source_id = f"locomo:{sample_id}:{dia_id}"
                text = str(turn.get("text") or "")
                metadata = {
                    "dia_id": dia_id,
                    "session_number": session_number,
                    "turn_index": turn_index,
                    "raw_keys": sorted(str(k) for k in turn if k not in {"text"}),
                }
                record = SourceRecord(
                    dataset="locomo",
                    instance_id=instance_id,
                    source_id=source_id,
                    text=text,
                    speaker=str(turn.get("speaker") or "unknown"),
                    timestamp=str(timestamp) if timestamp is not None else None,
                    ordinal=ordinal,
                    metadata=metadata,
                )
                ordinal += 1
                session_records.append(record)
                source_index[dia_id] = record
                source_index[source_id] = record
            sessions.append(session_records)
            session_meta.append({
                "session_number": session_number,
                "session_id": f"D{session_number}",
                "timestamp": timestamp,
                "num_turns": len(session_records),
            })

        items: list[ResearchItem] = []
        for qa_index, qa in enumerate(raw.get("qa") or []):
            if not isinstance(qa, dict):
                continue
            evidence = [str(value) for value in (qa.get("evidence") or [])]
            source_ids = tuple(
                source_index[value].source_id for value in evidence if value in source_index
            )
            item_id = f"{instance_id}:qa:{qa_index}"
            items.append(
                ResearchItem(
                    dataset="locomo",
                    instance_id=instance_id,
                    item_id=item_id,
                    question=str(qa.get("question") or ""),
                    gold_answer=_coerce_answer(qa.get("answer")),
                    source_ids=source_ids,
                    grading="llm_judge",
                    metadata={
                        "category": qa.get("category"),
                        "evidence": evidence,
                        "evidence_resolved": len(source_ids) == len(evidence),
                        "qa_index": qa_index,
                    },
                )
            )

        yield ResearchInstance(
            dataset="locomo",
            instance_id=instance_id,
            sessions=sessions,
            items=items,
            source_path=str(path),
            source_hash=digest,
            metadata={
                "sample_id": sample_id,
                "speaker_a": raw.get("conversation", {}).get("speaker_a"),
                "speaker_b": raw.get("conversation", {}).get("speaker_b"),
                "session_meta": session_meta,
                "num_items": len(items),
                "raw_record_index": index,
            },
        )


def load_locomo(source_path: str | Path, *, limit: int | None = None) -> list[ResearchInstance]:
    """Load LoCoMo instances eagerly for small smoke tests or conversion."""
    return list(iter_locomo(source_path, limit=limit))
