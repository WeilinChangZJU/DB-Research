"""Run the A-MEM research wrapper on one deterministic history.

This command is deliberately offline.  It does not import the official
A-MEM repository, load an embedding model, or create an LLM client.  The
fixture only checks the source/snapshot/trace contracts and the M0-M3 replay
plumbing before any paid model run is resumed.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from bemula.baselines.base import MemoryAdapter
from bemula.research import (
    AmemResearchWrapper,
    MaintenanceArm,
    SourceEvent,
    run_amem_all_arms,
)
from bemula.schemas import Turn


class _Note:
    def __init__(self, note_id: str, content: str, timestamp: str):
        self.id = note_id
        self.content = content
        self.timestamp = timestamp
        self.context = "offline-fixture"
        self.keywords = content.lower().split()[:4]
        self.tags = ["fixture"]
        self.links: list[int] = []


class _System:
    def __init__(self):
        self.memories: OrderedDict[str, _Note] = OrderedDict()
        self.counter = 0

    def add_note(self, content: str, time: str = "") -> str:
        note_id = f"note-{self.counter}"
        self.counter += 1
        self.memories[note_id] = _Note(note_id, content, time)
        return note_id

    def find_related_memories(self, query: str, k: int = 5):
        tokens = set(query.lower().split())
        ranked = []
        for index, note in enumerate(self.memories.values()):
            overlap = len(tokens & set(note.content.lower().split()))
            ranked.append((-overlap, index))
        ranked.sort()
        return "", [index for score, index in ranked[:k] if score < 0]


class _OfflineAmem(MemoryAdapter):
    name = "offline-amem-fixture"

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._system = _System()

    def observe(self, turn: Turn) -> None:
        # Mirror A-MEM's process_memory neighbour read without an LLM.
        self._system.find_related_memories(turn.text, k=5)
        self._system.add_note(turn.text, time=turn.timestamp)

    def query(self, question: str) -> str:
        _, indices = self._system.find_related_memories(question, k=5)
        notes = list(self._system.memories.values())
        return "\n".join(notes[index].content for index in indices)


def _factory() -> MemoryAdapter:
    return _OfflineAmem()


def _result_json(results: dict[MaintenanceArm, tuple[Any, Any]]) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm, (result, comparison) in results.items():
        arms[arm.value] = {
            "exact_to_reference": comparison.equal,
            "state_hash": result.snapshot.state_hash,
            "record_ids": [record.source_id for record in result.snapshot.records],
            "replay_start": result.metadata.get("replay_start"),
            "replayed_events": result.cost.replayed_events,
            "trace_events": len(result.wrapper.decision_events),
            "event_types": [event.event_type.value for event in result.wrapper.decision_events],
            "mismatches": {
                "missing_ids": list(comparison.missing_ids),
                "extra_ids": list(comparison.extra_ids),
                "changed_ids": list(comparison.changed_ids),
                "reason": comparison.reason,
            },
        }
    return {
        "offline": True,
        "api_calls": 0,
        "fixture": "single-history-amem-wrapper",
        "deleted_sources": ["src-a"],
        "arms": arms,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    events = [
        SourceEvent("e0", "src-a", "approval limit is ten thousand", timestamp="t0"),
        SourceEvent("e1", "src-b", "approval limit is ten thousand", timestamp="t1"),
        SourceEvent("e2", "src-c", "office is in Shanghai", timestamp="t2"),
    ]
    results = run_amem_all_arms(events, ["src-a"], _factory, retrieve_k=2)
    payload = _result_json(results)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(text, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
