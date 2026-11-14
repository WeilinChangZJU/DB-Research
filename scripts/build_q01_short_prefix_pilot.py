"""Build a bounded, source-level pilot from one frozen history per dataset.

The full LoCoMo/LongMemEval-S histories are intentionally retained as the
primary inputs.  This helper derives a deterministic prefix pilot so a real
model M0--M3 smoke run can exercise the complete maintenance protocol without
spending the budget of a full-history sweep.  The selected deletion ordinal
must lie inside the prefix; all hashes and the parent manifest are recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_first(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        line = next((item for item in handle if item.strip()), "")
    if not line:
        raise ValueError(f"empty history input: {path}")
    return json.loads(line)


def _derive(parent: dict[str, Any], parent_path: Path, prefix_events: int, policy: str) -> dict[str, Any]:
    events = list(parent.get("events", []))[:prefix_events]
    if len(events) != prefix_events:
        raise ValueError(f"requested {prefix_events} events, input contains {len(events)}")
    parent_deleted = list(parent.get("deleted_source_ids", []))
    deleted = [source_id for source_id in parent_deleted if any(event.get("source_id") == source_id for event in events)]
    if not deleted:
        raise ValueError("prefix does not contain the frozen deleted source")
    ordinals = [index for index, event in enumerate(events) if event.get("source_id") in set(deleted)]
    items = list(parent.get("items", []))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset": parent["dataset"],
        "instance_id": parent["instance_id"],
        "source_path": str(parent_path.resolve()),
        "source_file_sha256": parent.get("source_file_sha256"),
        "parent_history_manifest_sha256": parent.get("history_manifest_sha256"),
        "parent_num_source_events": parent.get("num_source_events"),
        "mask_policy": policy,
        "mask_version": 1,
        "deleted_event_ordinals": ordinals,
        "deleted_source_ids": deleted,
        "num_source_events": len(events),
        "source_history_sha256": _stable_hash(events),
        "items_count": len(items),
        "items": items,
        "events": events,
    }
    payload["history_manifest_sha256"] = _stable_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--longmemeval", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix-events", type=int, default=144)
    args = parser.parse_args()
    if args.prefix_events < 1:
        raise SystemExit("--prefix-events must be >= 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for dataset, source in (("locomo", args.locomo), ("longmemeval_s", args.longmemeval)):
        source = source.expanduser().resolve()
        parent = _load_first(source)
        payload = _derive(parent, source, args.prefix_events, f"prefix_{args.prefix_events}_frozen_source_v1")
        out = args.out_dir / f"{dataset}_explicit_histories_pilot.jsonl"
        out.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "dataset": dataset,
            "source_path": str(source),
            "source_file_sha256": _hash_file(source),
            "parent_history_manifest_sha256": parent.get("history_manifest_sha256"),
            "prefix_events": args.prefix_events,
            "mask_policy": payload["mask_policy"],
            "history_count": 1,
            "history_ids": [payload["instance_id"]],
            "history_manifest_sha256": _stable_hash([payload]),
            "output_sha256": _hash_file(out),
        }
        manifest_path = out.with_suffix(out.suffix + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs.extend((out, manifest_path))
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
