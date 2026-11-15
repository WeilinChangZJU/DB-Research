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
    MaintenanceArm,
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
            if not isinstance(raw_events, list) or not raw_events:
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
                        speaker=(
                            str(raw["speaker"])
                            if raw.get("speaker") is not None
                            else (str(raw.get("metadata", {}).get("speaker")) if raw.get("metadata", {}).get("speaker") is not None else None)
                        ),
                        session_id=(
                            str(raw["session_id"])
                            if raw.get("session_id") is not None
                            else (str(raw.get("metadata", {}).get("session_id")) if raw.get("metadata", {}).get("session_id") is not None else None)
                        ),
                    )
                )
            event_ids = [event.event_id for event in events]
            if len(set(event_ids)) != len(event_ids):
                raise ValueError(f"history {item['instance_id']} contains duplicate event IDs")
            raw_deleted = item.get("deleted_source_ids", [])
            if not isinstance(raw_deleted, list):
                raise ValueError(f"history {item['instance_id']} deleted_source_ids must be a list")
            deleted = tuple(dict.fromkeys(str(value) for value in raw_deleted))
            source_ids = {event.source_id for event in events}
            if len(source_ids) != len(events):
                raise ValueError(f"history {item['instance_id']} contains duplicate source IDs")
            unknown = sorted(set(deleted) - source_ids)
            if unknown:
                raise ValueError(f"history {item['instance_id']} deletes unknown source IDs: {unknown}")
            raw_items = item.get("items", [])
            if not isinstance(raw_items, list):
                raise ValueError(f"history {item['instance_id']} items must be a list")
            yield {
                "instance_id": item["instance_id"],
                "base_instance_id": str(item.get("base_instance_id", item["instance_id"])),
                "dataset": str(item.get("dataset", "q01")),
                "selection": dict(item.get("selection", {})),
                "mask_policy": item.get("mask_policy", item.get("mask_policy_version")),
                "events": events,
                "deleted_source_ids": deleted,
                "items": list(raw_items),
            }


def _load_histories(path: Path) -> list[dict[str, Any]]:
    """Compatibility helper for tests; the main runner consumes the stream."""

    return list(_iter_histories(path))


def _validate_histories(path: Path, limit: int | None = None) -> int:
    """Validate the selected JSONL prefix before constructing a provider.

    The paired runner streams rows during execution.  A separate validation
    pass prevents a malformed later row from surfacing only after earlier
    histories have already incurred model cost.  It keeps the full workload
    out of memory and returns the number of rows that will be executed.
    """

    stream = _iter_histories(path)
    if limit is not None:
        stream = islice(stream, limit)
    return sum(1 for _ in stream)


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
        query_probe = query_probes.get(arm.value)
        arms[arm.value] = {
            "exact_to_m0": comparison.equal,
            "visible_equivalent": comparison.visible_equivalent,
            "runtime_equivalent": comparison.runtime_equivalent,
            "continuation_checks_passed": (
                comparison.continuation_checks_passed
                if comparison.continuation_checks_passed is not None
                else (query_probe or {}).get("continuation_checks_passed")
            ),
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
            "query_probe": query_probe,
        }
    return {
        "schema_version": 2,
        "instance_id": instance["instance_id"],
        "base_instance_id": instance.get("base_instance_id", instance["instance_id"]),
        "dataset": instance["dataset"],
        "selection": dict(instance.get("selection", {})),
        "mask_policy": instance.get("mask_policy"),
        "deleted_source_ids": list(instance["deleted_source_ids"]),
        "arms": arms,
    }


def _probe_query(
    result: Any,
    *,
    arm: MaintenanceArm,
    question: str,
    question_id: str,
    deleted: set[str],
    trace: TraceRecorder,
) -> tuple[dict[str, Any], dict[str, int | float], float]:
    """Run one post-maintenance query on an isolated runtime fork."""

    probe_wrapper = result.wrapper
    query_isolated = False
    try:
        probe_wrapper = result.wrapper.fork(trace=trace)
        query_isolated = True
    except Exception:
        # Preserve the probe as a diagnostic, but make the missing isolation
        # explicit in the receipt and never use it as a continuation oracle.
        probe_wrapper = result.wrapper
    probe_wrapper.set_trace_context(phase="query_probe", query_id=question_id)
    before_events = len(probe_wrapper.provenance_events)
    started = time.perf_counter()
    probe = probe_wrapper.query(question)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    new_events = probe_wrapper.provenance_events[before_events:]
    llm_events = [
        event for event in new_events
        if getattr(getattr(event, "event_type", None), "value", str(getattr(event, "event_type", ""))) == "llm_call"
    ]

    def optional_total(field: str) -> int | None:
        values = [event.metadata.get(field) for event in llm_events]
        if not values or any(not isinstance(value, (int, float)) for value in values):
            return None
        total = sum(values)
        return int(total)

    input_tokens = optional_total("input_tokens")
    output_tokens = optional_total("output_tokens")
    usage_observed = bool(llm_events) and all(
        bool(event.metadata.get("usage_observed"))
        and isinstance(event.metadata.get("input_tokens"), (int, float))
        and isinstance(event.metadata.get("output_tokens"), (int, float))
        for event in llm_events
    )
    evidence = probe.evidence
    details = dict(evidence.details)
    source_ids = sorted(set(evidence.source_ids) | set(evidence.read_set))
    deleted_exposed = bool(deleted.intersection(source_ids))
    evidence_payload = evidence.model_dump(mode="json") if hasattr(evidence, "model_dump") else {
        "source_ids": list(evidence.source_ids),
        "read_set": list(evidence.read_set),
        "details": details,
    }
    record = {
        "question_id": question_id,
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "answer_sha256": hashlib.sha256(str(probe.answer).encode("utf-8")).hexdigest(),
        "evidence_sha256": hashlib.sha256(
            json.dumps(evidence_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
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
        "query_cost_observed": usage_observed,
        "query_isolated": query_isolated,
    }
    counters: dict[str, int | float] = {
        "llm_calls": len(llm_events),
        "retrieved_items": len(details.get("candidate_ids", [])),
        "memory_reads": len(evidence.read_set),
    }
    instrumentation_cost = details.get("instrumentation_cost", {})
    if isinstance(instrumentation_cost, dict):
        # These counters represent extra score reconstruction performed by
        # the audit hook.  They are intentionally kept separate from the
        # adapter's retrieval/embedding counters and included in lifecycle
        # cost summaries rather than silently folded into zero.
        for name in ("score_encode_calls", "score_matrix_rows", "score_matrix_values"):
            value = instrumentation_cost.get(name)
            if isinstance(value, (int, float)):
                counters[f"instrumentation_{name}"] = value
    record["instrumentation_cost"] = instrumentation_cost
    if input_tokens is not None:
        counters["input_tokens"] = input_tokens
    if output_tokens is not None:
        counters["output_tokens"] = output_tokens
    return record, counters, elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", default="amem")
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument(
        "--m3-mode",
        choices=("materialized", "conservative_full"),
        default="conservative_full",
        help="M3 dependency policy; conservative_full is the correctness baseline.",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=1,
        help="Number of declared items to probe per arm/history (default: 1 for cost-bounded pilots).",
    )
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
    if args.query_limit < 1:
        raise SystemExit("--query-limit must be >= 1")
    args.out.mkdir(parents=True, exist_ok=True)
    _validate_histories(args.histories, args.limit)
    history_stream = _iter_histories(args.histories)
    if args.limit is not None:
        history_stream = islice(history_stream, args.limit)
    root = Path(
        os.environ.get("BEMULA_SOURCE_ROOT", Path(__file__).resolve().parents[1])
    ).resolve()
    store = CheckpointStore(args.out)
    tree_digest = sha256_tree(root, exclude=(".venv", ".git", "results"))
    fingerprint = {
        "schema_version": 1,
        "runner": "run_q01_amem_paired",
        "baseline": args.baseline,
        "histories_sha256": sha256_file(args.histories),
        "retrieve_k": args.retrieve_k,
        "effective_serve_k": args.retrieve_k,
        "query_limit": args.query_limit,
        "m3_mode": args.m3_mode,
        "repository_tree_sha256": tree_digest,
        "agent_model": args.agent_model,
        "embed_backend": os.environ.get("BEMULA_EMBED_BACKEND", "api"),
        "freeze_sha256": os.environ.get("BEMULA_FREEZE_SHA256"),
        "official_amem_commit": os.environ.get("BEMULA_AMEM_COMMIT"),
        "official_amem_source_tree_sha256": os.environ.get(
            "BEMULA_AMEM_SOURCE_TREE_SHA256"
        ),
        "model_config_sha256": os.environ.get("BEMULA_MODEL_CONFIG_SHA256"),
    }
    store.ensure_fingerprint(fingerprint)
    done = store.completed_ids(validate=True) if args.resume else set()
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
        # The arm runner builds two independent M0 references plus one shared
        # baseline, then forks that baseline for M1--M3.  This keeps the paired
        # history serial and makes the shared setup visible in the lifecycle
        # cost ledger.
        factory = lambda: _build_baseline(
            args.baseline,
            get_llm(model=args.agent_model),
            embed_provider,
            retrieve_k=args.retrieve_k,
        )
        outputs = run_amem_all_arms(
            instance["events"],
            instance["deleted_source_ids"],
            factory,
            retrieve_k=args.retrieve_k,
            m3_mode=args.m3_mode,
            trace=trace,
            trace_context={
                "instance_id": instance_id,
                "base_instance_id": instance.get("base_instance_id", instance_id),
                "scenario_id": instance_id,
                "selection": instance.get("selection", {}),
            },
        )
        deleted = set(instance["deleted_source_ids"])
        query_items = [
            {
                "question_id": str(item.get("item_id", f"item:{index}")),
                "question": str(item.get("question", "")),
            }
            for index, item in enumerate(instance.get("items", []))
            if isinstance(item, dict) and str(item.get("question", "")).strip()
        ]
        deleted_events = [event for event in instance["events"] if event.source_id in deleted]
        if not query_items:
            query_items = [
                {
                    "question_id": "fallback",
                    "question": (
                        deleted_events[0].content
                        if deleted_events else instance["events"][0].content
                    ),
                }
            ]
        # The default is one declared item to keep a pilot affordable.  The
        # selected IDs/count are durable so a run cannot be mistaken for an
        # all-question workload evaluation.
        query_items = [
            {**item, "question": str(item["question"])[:512]}
            for item in query_items[: args.query_limit]
        ]
        canonical_snapshots = {
            arm.value: result.wrapper.canonical_snapshot(
                dataset=instance["dataset"], instance_id=instance["instance_id"]
            )
            for arm, (result, _comparison) in outputs.items()
        }
        query_probes: dict[str, dict[str, Any]] = {}
        for arm, (result, _comparison) in outputs.items():
            records: list[dict[str, Any]] = []
            for item in query_items:
                record, counters, elapsed_ms = _probe_query(
                    result,
                    arm=arm,
                    question=item["question"],
                    question_id=item["question_id"],
                    deleted=deleted,
                    trace=trace,
                )
                records.append(record)
                cost.record(
                    "query_probe",
                    scope="query_probe",
                    duration_ms=elapsed_ms,
                    counters=counters,
                    arm=arm.value,
                    query_id=item["question_id"],
                    query_sha256=record["question_sha256"],
                    deleted_source_exposed=record["deleted_source_exposed"],
                    evidence_complete=record["evidence_complete"],
                    query_cost_observed=record["query_cost_observed"],
                    query_isolated=record["query_isolated"],
                )
            query_probes[arm.value] = {
                "query_policy": "declared_items_prefix_v1",
                "query_count_requested": len(query_items),
                "records": records,
                "evidence_complete": all(record["evidence_complete"] for record in records),
                "deleted_source_exposed": any(record["deleted_source_exposed"] for record in records),
                "query_cost_observed": all(record["query_cost_observed"] for record in records),
                "query_isolated": all(record["query_isolated"] for record in records),
            }
        # Continuation is a separate oracle from the pre-query visible state.
        # Compare the same question IDs against the independent M0 probe only
        # when both probes were isolated and evidence was observed.
        m0_records = query_probes[MaintenanceArm.M0_FULL_REBUILD.value]["records"]
        m0_by_id = {record["question_id"]: record for record in m0_records}
        for arm_name, probe_group in query_probes.items():
            checks: list[bool] = []
            for record in probe_group["records"]:
                reference_record = m0_by_id.get(record["question_id"])
                passed = bool(
                    reference_record
                    and record.get("query_isolated")
                    and reference_record.get("query_isolated")
                    and record.get("answer_sha256") == reference_record.get("answer_sha256")
                    and record.get("evidence_sha256") == reference_record.get("evidence_sha256")
                    and not record.get("deleted_source_exposed")
                )
                record["answer_equal_to_m0"] = bool(
                    reference_record and record.get("answer_sha256") == reference_record.get("answer_sha256")
                )
                record["evidence_equal_to_m0"] = bool(
                    reference_record and record.get("evidence_sha256") == reference_record.get("evidence_sha256")
                )
                record["continuation_check_passed"] = passed
                checks.append(passed)
            probe_group["continuation_checks_passed"] = bool(checks) and all(checks)
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
        # The paired arms share one immutable baseline runtime.  Record that
        # build once as a lifecycle component rather than duplicating it in
        # each M1-M3 arm (or silently omitting it from the ledger).
        shared_baseline = next(
            (
                result.metadata.get("shared_baseline_cost")
                for result, _comparison in outputs.values()
                if isinstance(result.metadata.get("shared_baseline_cost"), dict)
            ),
            None,
        )
        if shared_baseline is not None:
            cost.record_arm_cost(
                arm="shared_baseline",
                cost=SimpleNamespace(**shared_baseline),
                exact_to_reference=None,
                phase="common_baseline",
            )
        reference_repeat = outputs[MaintenanceArm.M0_FULL_REBUILD][0].metadata.get("reference_repeat_cost")
        if isinstance(reference_repeat, dict):
            cost.record_arm_cost(
                arm="reference_repeat",
                cost=SimpleNamespace(**reference_repeat),
                exact_to_reference=None,
                phase="reference_repeat",
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
                "query_limit": args.query_limit,
                "agent_model": args.agent_model,
            },
            extra={"run_fingerprint_sha256": store.load_fingerprint()["sha256"]},
        ),
    )
    print(f"Wrote paired Q01 A-MEM results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
