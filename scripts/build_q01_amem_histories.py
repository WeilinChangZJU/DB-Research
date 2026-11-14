"""Build explicit-deletion A-MEM history JSONL from a frozen dataset adapter.

The output schema is consumed by ``run_q01_amem_paired.py``.  Deletion masks
are selected before any maintenance run by a named deterministic policy; the
policy, source file hash, source/event hashes and selected ordinals are stored
alongside every history so a result can be replayed without inferring a mask
from an answer or a post-hoc failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from bemula.research.datasets import file_hash, load_locomo, load_longmemeval_s


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mask_ordinals(count: int, policy: str) -> list[int]:
    if count < 1:
        return []
    if policy == "ordinal_quartile_single_v1":
        return [min(count - 1, count // 4)]
    if policy == "ordinal_quartiles_pair_v1":
        first = min(count - 1, count // 4)
        second = min(count - 1, (3 * count) // 4)
        return list(dict.fromkeys((first, second)))
    if policy == "ordinal_late_single_v1":
        return [count - 1]
    raise ValueError(f"unknown mask policy: {policy}")


def _history_payload(instance: Any, source_path: Path, source_file_sha256: str, policy: str) -> dict[str, Any]:
    records = list(instance.source_records)
    ordinals = _mask_ordinals(len(records), policy)
    events: list[dict[str, Any]] = []
    for ordinal, record in enumerate(records):
        metadata = {
            **dict(record.metadata),
            "dataset": record.dataset,
            "instance_id": record.instance_id,
            "source_ordinal": ordinal,
            "source_record_hash": record.source_hash,
            "source_file_sha256": source_file_sha256,
        }
        event = {
            "event_id": f"{record.source_id}:event",
            "source_id": record.source_id,
            "content": record.text,
            "timestamp": record.timestamp or "",
            "metadata": metadata,
        }
        event["event_hash"] = _stable_hash(event)
        events.append(event)
    deleted_source_ids = [events[ordinal]["source_id"] for ordinal in ordinals]
    # Preserve evaluation questions and their declared source overlays for
    # post-maintenance serving probes.  Gold answers are intentionally omitted
    # from this source-level runner so the maintenance path cannot read them.
    items = []
    for item_index, item in enumerate(instance.items):
        items.append(
            {
                "item_id": str(getattr(item, "item_id", f"{instance.instance_id}:item:{item_index}")),
                "question": str(getattr(item, "question", "")),
                "source_ids": [str(value) for value in (getattr(item, "source_ids", ()) or ())],
                "grading": str(getattr(item, "grading", "llm_judge")),
                "metadata": dict(getattr(item, "metadata", {}) or {}),
            }
        )
    payload = {
        "schema_version": 1,
        "dataset": str(instance.dataset),
        "instance_id": str(instance.instance_id),
        "source_path": str(source_path.resolve()),
        "source_file_sha256": source_file_sha256,
        "mask_policy": policy,
        "mask_version": 1,
        "deleted_event_ordinals": ordinals,
        "deleted_source_ids": deleted_source_ids,
        "num_source_events": len(events),
        "source_history_sha256": _stable_hash(events),
        "items_count": len(instance.items),
        "items": items,
        "events": events,
    }
    payload["history_manifest_sha256"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "history_manifest_sha256"}
    )
    return payload


def _load_instances(dataset: str, source: Path, limit: int | None) -> Iterable[Any]:
    if dataset == "locomo":
        return load_locomo(source, limit=limit)
    return load_longmemeval_s(source, limit=limit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("locomo", "longmemeval_s"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mask-policy",
        choices=(
            "ordinal_quartile_single_v1",
            "ordinal_quartiles_pair_v1",
            "ordinal_late_single_v1",
        ),
        default="ordinal_quartile_single_v1",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    source = args.source.expanduser().resolve()
    source_sha256 = file_hash(source)
    instances = list(_load_instances(args.dataset, source, args.limit))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        _history_payload(instance, source, source_sha256, args.mask_policy)
        for instance in instances
    ]
    args.out.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in lines),
        encoding="utf-8",
    )
    receipt = args.out.with_suffix(args.out.suffix + ".manifest.json")
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": args.dataset,
                "source_path": str(source),
                "source_file_sha256": source_sha256,
                "mask_policy": args.mask_policy,
                "history_count": len(lines),
                "history_ids": [item["instance_id"] for item in lines],
                "history_manifest_sha256": _stable_hash(lines),
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
