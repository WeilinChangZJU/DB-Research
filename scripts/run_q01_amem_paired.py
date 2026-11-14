"""Run the source-level Q01 M0--M3 paired experiment on A-MEM.

The input is JSONL with one independent history per line::

    {"instance_id": "h0", "dataset": "pilot",
     "deleted_source_ids": ["src-2"],
     "events": [{"event_id": "e0", "source_id": "src-0",
                 "content": "...", "timestamp": "...",
                 "metadata": {"speaker": "user"}}]}

This runner is deliberately serial.  A-MEM adapters are mutable and the
first correctness pass must keep each paired history isolated; parallelism can
be added only after an exclusive worker/factory contract is measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Any

from bemula.llm import get_llm
from bemula.research import (
    CheckpointStore,
    CostRecorder,
    SourceEvent,
    TraceRecorder,
    run_amem_all_arms,
    sha256_file,
    sha256_tree,
    write_manifest,
    build_manifest,
)
from scripts.run_eval import _build_baseline


def _iter_histories(path: Path):
    # Iterate the JSONL stream instead of read_text().splitlines().  The full
    # LongMemEval-S input is hundreds of megabytes, and duplicating it as a
    # string plus a list of lines can otherwise dominate the runner's memory
    # before the first paired arm starts.
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not isinstance(item.get("instance_id"), str):
                raise ValueError(f"invalid history at {path}:{line_no}")
            raw_events = item.get("events")
            if not isinstance(raw_events, list):
                raise ValueError(f"history {item['instance_id']} must contain an events list")
            events = []
            for index, raw in enumerate(raw_events):
                if not isinstance(raw, dict):
                    raise ValueError(f"history {item['instance_id']} event {index} is not an object")
                events.append(
                    SourceEvent(
                        event_id=str(raw.get("event_id", f"e{index}")),
                        source_id=str(raw["source_id"]),
                        content=str(raw["content"]),
                        timestamp=str(raw.get("timestamp", "")),
                        metadata=dict(raw.get("metadata", {})),
                    )
                )
            deleted = tuple(dict.fromkeys(str(value) for value in item.get("deleted_source_ids", [])))
            source_ids = {event.source_id for event in events}
            unknown = sorted(set(deleted) - source_ids)
            if unknown:
                raise ValueError(f"history {item['instance_id']} deletes unknown source IDs: {unknown}")
            yield {
                "instance_id": item["instance_id"],
                "dataset": str(item.get("dataset", "q01")),
                "events": events,
                "deleted_source_ids": deleted,
                "items": list(item.get("items", [])),
            }


def _load_histories(path: Path) -> list[dict[str, Any]]:
    """Compatibility helper for tests; the main runner consumes the stream."""

    return list(_iter_histories(path))


def _summary(
    instance: dict[str, Any],
    outputs: dict[Any, tuple[Any, Any]],
    *,
    canonical_snapshots: dict[str, Any],
    query_probes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm, (result, comparison) in outputs.items():
        canonical = canonical_snapshots[arm.value]
        arms[arm.value] = {
            "exact_to_m0": comparison.equal,
            "state_hash": canonical.canonical_hash,
            "visible_state_hash": canonical.visible_state_hash,
            "runtime_state_hash": canonical.runtime_state_hash,
            "audit_trace_hash": canonical.audit_trace_hash,
            "snapshot_hash": result.snapshot.state_hash,
            "cost": result.cost.__dict__,
            "metadata": dict(result.metadata),
            "mismatches": {
                "missing_ids": list(comparison.missing_ids),
                "extra_ids": list(comparison.extra_ids),
                "changed_ids": list(comparison.changed_ids),
                "reason": comparison.reason,
            },
            "query_probe": query_probes.get(arm.value),
        }
    return {
        "schema_version": 1,
        "instance_id": instance["instance_id"],
        "dataset": instance["dataset"],
        "deleted_source_ids": list(instance["deleted_source_ids"]),
        "arms": arms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", default="amem")
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument(
        "--agent-model",
        default=os.environ.get("BEMULA_AGENT_MODEL", "deepseek-flash"),
        help="Pinned agent model name; defaults to BEMULA_AGENT_MODEL or deepseek-flash.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.retrieve_k < 1:
        raise SystemExit("--retrieve-k must be >= 1")
    args.out.mkdir(parents=True, exist_ok=True)
    history_stream = _iter_histories(args.histories)
    if args.limit is not None:
        history_stream = islice(history_stream, args.limit)
    root = Path.cwd().resolve()
    store = CheckpointStore(args.out)
    tree_digest = sha256_tree(root, exclude=(".venv", ".git", "results"))
    fingerprint = {
        "schema_version": 1,
        "runner": "run_q01_amem_paired",
        "baseline": args.baseline,
        "histories_sha256": sha256_file(args.histories),
        "retrieve_k": args.retrieve_k,
        "repository_tree_sha256": tree_digest,
        "agent_model": args.agent_model,
        "embed_backend": os.environ.get("BEMULA_EMBED_BACKEND", "api"),
    }
    store.ensure_fingerprint(fingerprint)
    done = store.completed_ids(validate=True) if args.resume else set()
    agent_llm = get_llm(model=args.agent_model)
    embed_backend = os.environ.get("BEMULA_EMBED_BACKEND", "api").strip().lower()
    embed_provider = {"api": "openai", "local": "huggingface"}.get(embed_backend, embed_backend)
    cost = CostRecorder(args.out / "cost.jsonl")
    # Preserve lifecycle cost across a resumed run; the checkpoint store and
    # cost ledger must describe the same completed history set.
    if args.resume:
        cost.load_existing()
    trace = TraceRecorder(args.out / "trace.jsonl")
    for instance in history_stream:
        instance_id = instance["instance_id"]
        if instance_id in done:
            continue
        # The factory is called once per arm by run_amem_all_arms, giving every
        # arm a fresh adapter while keeping this paired run serial.
        factory = lambda: _build_baseline(args.baseline, agent_llm, embed_provider)
        outputs = run_amem_all_arms(
            instance["events"],
            instance["deleted_source_ids"],
            factory,
            retrieve_k=args.retrieve_k,
            trace=trace,
        )
        deleted = set(instance["deleted_source_ids"])
        item_questions = [
            str(item.get("question", ""))
            for item in instance.get("items", [])
            if isinstance(item, dict) and str(item.get("question", "")).strip()
        ]
        deleted_events = [event for event in instance["events"] if event.source_id in deleted]
        question = (
            item_questions[0]
            if item_questions
            else (deleted_events[0].content if deleted_events else instance["events"][0].content)
        )[:512]
        canonical_snapshots = {
            arm.value: result.wrapper.canonical_snapshot(
                dataset=instance["dataset"], instance_id=instance["instance_id"]
            )
            for arm, (result, _comparison) in outputs.items()
        }
        query_probes: dict[str, dict[str, Any]] = {}
        for arm, (result, _comparison) in outputs.items():
            before_events = len(result.wrapper.provenance_events)
            started = time.perf_counter()
            probe = result.wrapper.query(question)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            new_events = result.wrapper.provenance_events[before_events:]
            llm_events = [
                event for event in new_events
                if getattr(getattr(event, "event_type", None), "value", str(getattr(event, "event_type", ""))) == "llm_call"
            ]
            input_tokens = sum(int(event.metadata.get("input_tokens", 0) or 0) for event in llm_events)
            output_tokens = sum(int(event.metadata.get("output_tokens", 0) or 0) for event in llm_events)
            evidence = probe.evidence
            details = dict(evidence.details)
            source_ids = sorted(set(evidence.source_ids) | set(evidence.read_set))
            deleted_exposed = bool(deleted.intersection(source_ids))
            query_probes[arm.value] = {
                "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "source_ids": source_ids,
                "read_set": sorted(set(evidence.read_set)),
                "candidate_ids": list(details.get("candidate_ids", [])),
                "selected_ids": list(details.get("selected_ids", [])),
                "context_object_ids": list(details.get("read_object_ids", [])),
                "context_source_ids": list(details.get("context_source_ids", [])),
                "evidence_complete": bool(details.get("evidence_complete", False)),
                "deleted_source_exposed": deleted_exposed,
                "llm_calls": len(llm_events),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "query_cost_observed": bool(llm_events),
            }
            counters: dict[str, int] = {
                "llm_calls": len(llm_events),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "retrieved_items": len(details.get("candidate_ids", [])),
                "memory_reads": len(evidence.read_set),
            }
            cost.record(
                "query_probe",
                scope="query_probe",
                duration_ms=elapsed_ms,
                counters=counters,
                arm=arm.value,
                query_sha256=query_probes[arm.value]["question_sha256"],
                deleted_source_exposed=deleted_exposed,
                evidence_complete=bool(details.get("evidence_complete", False)),
            )
        summary = _summary(
            instance,
            outputs,
            canonical_snapshots=canonical_snapshots,
            query_probes=query_probes,
        )
        for arm, (result, comparison) in outputs.items():
            cost.record_arm_cost(
                arm=arm.value,
                cost=result.cost,
                exact_to_reference=comparison.equal,
                phase="reference_rebuild" if arm.value.startswith("M0_") else "maintenance",
            )
            setup = result.metadata.get("baseline_setup_cost")
            if setup:
                cost.record_arm_cost(
                    arm=f"{arm.value}:baseline_setup",
                    cost=SimpleNamespace(**setup),
                    exact_to_reference=None,
                    phase="baseline_setup",
                )
        trace.record(
            "q01_instance_complete",
            instance_id=instance_id,
            deleted_source_ids=list(instance["deleted_source_ids"]),
            arm_count=len(outputs),
        )
        store.commit(instance_id, [summary])
    cost.write_summary(args.out / "cost_summary.json")
    write_manifest(
        args.out / "manifest.json",
        build_manifest(
            root=root,
            data_paths=[args.histories],
            execution={
                "runner": "run_q01_amem_paired",
                "status": "completed",
                "num_instances": len(store.completed_ids(validate=True)),
                "resume": args.resume,
                "retrieve_k": args.retrieve_k,
                "agent_model": args.agent_model,
            },
            extra={"run_fingerprint_sha256": store.load_fingerprint()["sha256"]},
        ),
    )
    print(f"Wrote paired Q01 A-MEM results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
