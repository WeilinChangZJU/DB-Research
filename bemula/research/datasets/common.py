"""Shared records for external long-memory dataset adapters.

The adapters deliberately keep the raw dataset's provenance visible instead of
forcing external data into Bemula's synthetic unlearning schema.  A research
runner can map these records to its own backend while retaining stable source
IDs, content hashes, and dataset metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from bemula.schemas import Turn
import json


def content_hash(text: str) -> str:
    """Return a stable SHA-256 digest for a source text."""
    return sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    """Hash an external source file without copying it into the repository."""
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceRecord:
    """One immutable source turn with provenance metadata."""

    dataset: str
    instance_id: str
    source_id: str
    text: str
    speaker: str
    timestamp: str | None = None
    ordinal: int = 0
    source_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_hash:
            object.__setattr__(self, "source_hash", content_hash(self.text))

    def to_turn(self) -> Turn:
        """Protocol-compatible conversion to Bemula's canonical turn."""
        return Turn(
            speaker=self.speaker,
            text=self.text,
            timestamp=self.timestamp or "",
            linked_fact_ids=list(self.metadata.get("linked_fact_ids", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "instance_id": self.instance_id,
            "source_id": self.source_id,
            "text": self.text,
            "speaker": self.speaker,
            "timestamp": self.timestamp,
            "ordinal": self.ordinal,
            "source_hash": self.source_hash,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ResearchItem:
    """A question and its gold answer plus support/evidence provenance."""

    dataset: str
    instance_id: str
    item_id: str
    question: str
    gold_answer: Any
    source_ids: tuple[str, ...] = ()
    grading: str = "llm_judge"
    suite: str = "recall"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def related_source_ids(self) -> list[str]:
        """Protocol-compatible alias for evidence source IDs."""
        return list(self.source_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "instance_id": self.instance_id,
            "item_id": self.item_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "source_ids": list(self.source_ids),
            "grading": self.grading,
            "suite": self.suite,
            "metadata": self.metadata,
        }

    def to_protocol_item(self) -> Any:
        """Convert to the Pydantic item used by ResearchProtocol."""
        from ..protocol import ResearchItem as ProtocolItem

        return ProtocolItem(
            dataset=self.dataset,
            instance_id=self.instance_id,
            item_id=self.item_id,
            question=self.question,
            gold_answer="" if self.gold_answer is None else str(self.gold_answer),
            related_source_ids=list(self.source_ids),
            metadata=dict(self.metadata),
        )


@dataclass
class ResearchInstance:
    """One external-dataset evaluation unit.

    ``sessions`` is a list of source records grouped in chronological/session
    order.  ``source_path`` and ``source_hash`` point to the original file; the
    adapter never writes the large external data into the Bemula repository.
    """

    dataset: str
    instance_id: str
    sessions: list[list[SourceRecord]]
    items: list[ResearchItem]
    source_path: str
    source_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def split(self) -> str:
        return str(self.metadata.get("split", "external"))

    @property
    def turns(self) -> list[SourceRecord]:
        return self.source_records

    @property
    def control_turns(self) -> None:
        return None

    @property
    def forget_source_ids(self) -> list[str]:
        return list(self.metadata.get("forget_source_ids", []))

    @property
    def content_hash(self) -> str:
        payload = "\n".join(record.source_hash for record in self.source_records)
        return content_hash(payload)

    @property
    def source_records(self) -> list[SourceRecord]:
        return [record for session in self.sessions for record in session]

    def to_protocol_instance(self) -> Any:
        """Convert this external-dataset record to the canonical protocol model."""
        from ..protocol import ResearchInstance as ProtocolInstance, ResearchTurn

        turns = [
            ResearchTurn(
                dataset=self.dataset,
                instance_id=self.instance_id,
                source_id=record.source_id,
                speaker=record.speaker,
                text=record.text,
                timestamp=record.timestamp or "",
                source_hash=record.source_hash,
                metadata=dict(record.metadata),
            )
            for record in self.source_records
        ]
        return ProtocolInstance(
            dataset=self.dataset,
            instance_id=self.instance_id,
            turns=turns,
            items=[item.to_protocol_item() for item in self.items],
            forget_source_ids=list(self.forget_source_ids),
            metadata={
                **self.metadata,
                "source_path": self.source_path,
                "source_hash": self.source_hash,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "instance_id": self.instance_id,
            "source_path": self.source_path,
            "source_hash": self.source_hash,
            "metadata": self.metadata,
            "sessions": [
                [record.to_dict() for record in session] for session in self.sessions
            ],
            "items": [item.to_dict() for item in self.items],
        }


def dump_jsonl(instances: Iterable[ResearchInstance], path: str | Path) -> None:
    """Write adapter output as JSONL for inspection or downstream runners."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for instance in instances:
            stream.write(json.dumps(instance.to_dict(), ensure_ascii=False) + "\n")
