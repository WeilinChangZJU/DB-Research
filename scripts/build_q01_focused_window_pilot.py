"""Build a deterministic short window around each frozen deletion source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _first(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.loads(next(line for line in f if line.strip()))


def _window(parent: dict[str, Any], source: Path, start: int, length: int) -> dict[str, Any]:
    all_events = list(parent["events"])
    events = all_events[start : start + length]
    if len(events) != length:
        raise ValueError(f"window [{start}, {start + length}) exceeds {len(all_events)} events")
    deleted = [sid for sid in parent.get("deleted_source_ids", []) if any(e.get("source_id") == sid for e in events)]
    if not deleted:
        raise ValueError(f"window [{start}, {start + length}) does not contain the frozen deletion")
    deleted_set = set(deleted)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset": parent["dataset"],
        "instance_id": parent["instance_id"],
        "source_path": str(source.resolve()),
        "source_file_sha256": parent.get("source_file_sha256"),
        "parent_history_manifest_sha256": parent.get("history_manifest_sha256"),
        "parent_num_source_events": parent.get("num_source_events"),
        "selection": {"kind": "contiguous_window", "start": start, "length": length},
        "mask_policy": f"window_{start}_{length}_frozen_source_v1",
        "mask_version": 1,
        "deleted_event_ordinals": [i for i, e in enumerate(events) if e.get("source_id") in deleted_set],
        "deleted_source_ids": deleted,
        "num_source_events": len(events),
        "source_history_sha256": _stable_hash(events),
        "items_count": len(parent.get("items", [])),
        "items": list(parent.get("items", [])),
        "events": events,
    }
    payload["history_manifest_sha256"] = _stable_hash(payload)
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--locomo", type=Path, required=True)
    p.add_argument("--longmemeval", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--window-events", type=int, default=32)
    p.add_argument("--locomo-start", type=int, default=96)
    p.add_argument("--longmemeval-start", type=int, default=129)
    args = p.parse_args()
    if args.window_events < 1 or args.locomo_start < 0 or args.longmemeval_start < 0:
        raise SystemExit("window and starts must be non-negative; window-events must be >= 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for dataset, path, start in (
        ("locomo", args.locomo, args.locomo_start),
        ("longmemeval_s", args.longmemeval, args.longmemeval_start),
    ):
        source = path.resolve()
        payload = _window(_first(source), source, start, args.window_events)
        out = args.out_dir / f"{dataset}_explicit_histories_focused_pilot.jsonl"
        out.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "dataset": dataset,
            "source_path": str(source),
            "source_file_sha256": _sha(source),
            "parent_history_manifest_sha256": payload["parent_history_manifest_sha256"],
            "selection": payload["selection"],
            "mask_policy": payload["mask_policy"],
            "history_count": 1,
            "history_ids": [payload["instance_id"]],
            "history_manifest_sha256": _stable_hash([payload]),
            "output_sha256": _sha(out),
        }
        manifest_path = Path(str(out) + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs.extend((out, manifest_path))
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
