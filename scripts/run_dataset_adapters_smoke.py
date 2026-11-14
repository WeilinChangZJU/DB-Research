"""Run an offline LoCoMo/LongMemEval-S adapter and deletion-manifest smoke.

This command validates the external-data boundary only.  It loads at most a
small number of instances, preserves source/evidence IDs, creates deterministic
deletion conditions, and writes a provenance manifest.  It never invokes an
LLM or a memory backend, so the output is not an efficacy result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from bemula.research import (
    build_manifest,
    generate_deletion_scenarios,
    source_events_from_instance,
    write_manifest,
)
from bemula.research.datasets import load_locomo, load_longmemeval_s


def _instance_summary(instance: Any) -> dict[str, Any]:
    events = source_events_from_instance(instance)
    query_source_ids = {
        item.item_id: list(item.source_ids)
        for item in instance.items
        if getattr(item, "source_ids", ())
    }
    deletion_manifest = generate_deletion_scenarios(
        events,
        dataset=instance.dataset,
        history_id=instance.instance_id,
        query_source_ids=query_source_ids,
    )
    return {
        "dataset": instance.dataset,
        "instance_id": instance.instance_id,
        "source_path": instance.source_path,
        "source_hash": instance.source_hash,
        "source_count": len(events),
        "item_count": len(instance.items),
        "resolved_evidence_items": sum(bool(item.source_ids) for item in instance.items),
        "scenario_manifest": deletion_manifest.to_dict(),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo", required=True, help="official locomo10.json path")
    parser.add_argument("--longmemeval", required=True, help="cleaned LongMemEval-S JSON/JSONL path")
    parser.add_argument("--out", default="results/q01_dataset_adapter_smoke")
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    root = Path.cwd().resolve()
    out = Path(args.out)
    locomo = load_locomo(args.locomo, limit=args.limit)
    longmemeval = load_longmemeval_s(args.longmemeval, limit=args.limit)
    summaries = {
        "locomo": [_instance_summary(item) for item in locomo],
        "longmemeval_s": [_instance_summary(item) for item in longmemeval],
    }
    _write_json(out / "summary.json", {
        "offline": True,
        "api_calls": 0,
        "limit": args.limit,
        "datasets": {
            key: {
                "instance_count": len(value),
                "source_count": sum(item["source_count"] for item in value),
                "item_count": sum(item["item_count"] for item in value),
            }
            for key, value in summaries.items()
        },
        "instances": summaries,
    })
    for dataset, instances in summaries.items():
        _write_json(out / f"{dataset}_instances.json", instances)
    manifest = build_manifest(
        root=root,
        data_paths=[args.locomo, args.longmemeval],
        execution={"status": "completed", "runner": "run_dataset_adapters_smoke", "limit": args.limit},
        extra={"adapter_summary": str(out / "summary.json")},
    )
    write_manifest(out / "manifest.json", manifest)
    print(out / "summary.json")


if __name__ == "__main__":
    main()
