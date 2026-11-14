"""Build explicit source-deletion histories for serial Q01 A-MEM runs.

Each output JSONL row is one independently checkpointable history/mask pair.
The source event stream is copied with stable IDs, timestamps, session metadata,
and hashes.  The mask policy is deterministic and recorded in each row:

* ``answer_source_first`` deletes the first source in the dataset evidence
  session/source overlay;
* ``nonanswer_source_first`` deletes the first source outside that overlay;
* ``answer_and_nonanswer`` deletes both.

For LongMemEval-S the answer evidence is session-level by construction.  The
generated mask is therefore a source-level pressure condition and is never
labelled as an exact turn-level gold dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from bemula.research.datasets import load_locomo, load_longmemeval_s


POLICY_VERSION = "q01-explicit-mask-v1"


def _event_hash(events: list[dict[str, Any]]) -> str:
    payload = json.dumps(events, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_events(instance: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, record in enumerate(instance.source_records):
        events.append(
            {
                "event_id": record.source_id,
                "source_id": record.source_id,
                "content": record.text,
                "timestamp": record.timestamp or "",
                "metadata": {
                    **dict(record.metadata),
                    "dataset": record.dataset,
                    "instance_id": record.instance_id,
                    "speaker": record.speaker,
                    "ordinal": record.ordinal,
                    "source_hash": record.source_hash,
                    "event_ordinal": index,
                },
            }
        )
    return events


def _evidence_sources(instance: Any) -> tuple[str, ...]:
    values: list[str] = []
    for item in getattr(instance, "items", ()):
        values.extend(str(value) for value in getattr(item, "source_ids", ()) or ())
    return tuple(dict.fromkeys(values))


def _mask_rows(instance: Any, events: list[dict[str, Any]], masks: Iterable[str]) -> list[dict[str, Any]]:
    source_ids = [str(event["source_id"]) for event in events]
    evidence_ids = tuple(source_id for source_id in _evidence_sources(instance) if source_id in source_ids)
    evidence_set = set(evidence_ids)
    first_answer = evidence_ids[0] if evidence_ids else source_ids[0]
    nonanswer_ids = [source_id for source_id in source_ids if source_id not in evidence_set]
    first_nonanswer = nonanswer_ids[0] if nonanswer_ids else source_ids[-1]
    selected = {
        "answer_source_first": (first_answer,),
        "nonanswer_source_first": (first_nonanswer,),
        "answer_and_nonanswer": tuple(dict.fromkeys((first_answer, first_nonanswer))),
    }
    # Preserve the dataset's session-level evidence declaration for review.
    item_meta = [dict(getattr(item, "metadata", {}) or {}) for item in getattr(instance, "items", ())]
    session_ids = sorted({str(value) for meta in item_meta for value in (meta.get("answer_session_ids") or [])})
    output: list[dict[str, Any]] = []
    base_id = str(instance.instance_id)
    for mask_name in masks:
        if mask_name not in selected:
            raise ValueError(f"unknown mask policy: {mask_name}")
        deleted = list(selected[mask_name])
        row = {
            "instance_id": f"{base_id}:mask:{mask_name}",
            "base_instance_id": base_id,
            "dataset": str(instance.dataset),
            "mask_policy_version": POLICY_VERSION,
            "mask_kind": mask_name,
            "deleted_source_ids": deleted,
            "answer_evidence_source_ids": list(evidence_ids),
            "answer_evidence_session_ids": session_ids,
            "evidence_granularity": "session" if str(instance.dataset).startswith("longmem") else "source_overlay",
            "source_path": str(instance.source_path),
            "source_file_hash": str(instance.source_hash),
            "events_hash": _event_hash(events),
            "event_count": len(events),
            "items": [
                {
                    "item_id": str(getattr(item, "item_id", "")),
                    "question": str(getattr(item, "question", "")),
                    "source_ids": list(getattr(item, "source_ids", ()) or ()),
                    "metadata": dict(getattr(item, "metadata", {}) or {}),
                }
                for item in getattr(instance, "items", ())
            ],
            "events": events,
        }
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("locomo", "longmemeval_s"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument(
        "--masks",
        nargs="+",
        default=["answer_source_first", "nonanswer_source_first", "answer_and_nonanswer"],
        choices=["answer_source_first", "nonanswer_source_first", "answer_and_nonanswer"],
    )
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    loader = load_locomo if args.dataset == "locomo" else load_longmemeval_s
    instances = loader(args.source, limit=args.limit)
    rows: list[dict[str, Any]] = []
    for instance in instances:
        events = _source_events(instance)
        rows.extend(_mask_rows(instance, events, args.masks))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    receipt = args.out.with_suffix(args.out.suffix + ".receipt.json")
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy_version": POLICY_VERSION,
                "dataset": args.dataset,
                "source_path": str(args.source.resolve()),
                "history_count": len(rows),
                "base_instance_count": len(instances),
                "masks": list(args.masks),
                "source_file_hash": instances[0].source_hash if instances else None,
                "evidence_note": (
                    "LongMemEval-S answer_session_ids are session-level overlays; "
                    "they are not exact turn dependencies."
                    if args.dataset == "longmemeval_s"
                    else "LoCoMo evidence IDs are preserved as source overlays."
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.out)
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
